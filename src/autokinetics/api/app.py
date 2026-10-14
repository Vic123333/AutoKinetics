"""FastAPI application: REST endpoints + static single-page frontend."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..errors import AutoKineticsError, ExternalDataError
from . import workers
from .jobs import JobTimeout, Overloaded, runner
from .schemas import (
    ArrheniusRequest,
    EquilibriumRequest,
    FitRequest,
    MechanismPayload,
    NumericsRequest,
    OrderFitRequest,
    RopRequest,
    SensitivityRequest,
    SimulationRequest,
)
from .settings import settings

log = logging.getLogger("autokinetics.api")


def _limits() -> dict:
    return {"max_species": settings.max_species, "max_reactions": settings.max_reactions,
            "max_wall_time": settings.max_wall_time}


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.info("AutoKinetics %s – %d worker process(es)", __version__, settings.workers)
    yield
    runner.shutdown()


app = FastAPI(title="AutoKinetics API", version=__version__, lifespan=lifespan,
              description="Simulation chemischer Reaktionsnetzwerke (C++/Cython-Kern).")
app.add_middleware(GZipMiddleware, minimum_size=2048)
if settings.cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"],
                       allow_headers=["*"])


@app.middleware("http")
async def timing_and_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > settings.max_upload_bytes:
        return JSONResponse({"detail": "Anfrage zu groß"}, status_code=413)
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.perf_counter() - t0:.3f}"
    return response


@app.exception_handler(AutoKineticsError)
async def _domain_error(_: Request, exc: AutoKineticsError):
    status = 502 if isinstance(exc, ExternalDataError) else 422
    return JSONResponse({"detail": str(exc)}, status_code=status)


async def _run(fn, req: dict, timeout: float | None = None):
    try:
        out = await runner.run(fn, req, _limits(), timeout=timeout)
    except Overloaded:
        raise HTTPException(503, "Server ausgelastet – bitte gleich erneut versuchen") from None
    except JobTimeout:
        raise HTTPException(504, "Zeitlimit überschritten") from None
    if not out["ok"]:
        raise HTTPException(422, out["error"])
    return out["result"]


# ----------------------------------------------------------------------------- meta
@app.get("/api/health")
def health():
    try:
        from .._ext import _core  # noqa: F401
        ext = True
    except ImportError:
        ext = False
    try:
        from .. import numerics  # noqa: F401
        num = True
    except ImportError:
        num = False
    return {"status": "ok", "version": __version__, "extension": ext, "numerics": num,
            "workers": settings.workers, "online_data": settings.allow_online_data}


# ----------------------------------------------------------------------------- library & mechanisms
@app.get("/api/examples")
def examples():
    from ..library import list_examples

    return [e.to_dict() for e in list_examples()]


@app.get("/api/examples/{example_id}")
def example(example_id: str):
    from ..io.yaml_format import safe_load_yaml
    from ..library import load_example

    mech, meta, text = load_example(example_id)
    return {"meta": meta.to_dict(), "yaml": text, "document": safe_load_yaml(text)}


def _mechanism_summary(mech) -> dict:
    from ..io.yaml_format import mechanism_to_dict

    return {
        "name": mech.name, "phase": mech.phase, "n_species": mech.n_species, "n_reactions": mech.n_reactions,
        "species": [{"name": s.name, "composition": s.composition, "charge": s.charge,
                     "has_thermo": s.thermo is not None, "thermo_model": s.thermo.model if s.thermo else None,
                     "molecular_weight": s.mw * 1e3 if s.mw else None, "note": s.note} for s in mech.species],
        "reactions": [{"equation": r.equation, "type": r.type, "reversible": r.reversible,
                       "A": r.rate.A, "b": r.rate.b, "Ea": r.rate.Ea, "order": sum(r.forward_orders().values()),
                       "duplicate": r.duplicate} for r in mech.reactions],
        "elements": mech.elements,
        "issues": [i.to_dict() for i in mech.validate()],
        "document": mechanism_to_dict(mech),
        "metadata": mech.metadata,
    }


@app.post("/api/mechanism/parse")
def parse_mechanism(payload: MechanismPayload):
    """Parse YAML / .kin / document, validate and return a normalised Cantera-style document."""
    mech = workers.build_mechanism(payload.model_dump())
    if mech.n_species > settings.max_species or mech.n_reactions > settings.max_reactions:
        raise HTTPException(422, "Mechanismus überschreitet die Größenlimits")
    return _mechanism_summary(mech)


@app.post("/api/mechanism/export", response_class=PlainTextResponse)
def export_mechanism(payload: MechanismPayload):
    from ..io.yaml_format import dump_mechanism

    return dump_mechanism(workers.build_mechanism(payload.model_dump()))


# ----------------------------------------------------------------------------- computations
@app.post("/api/simulate")
async def simulate_endpoint(req: SimulationRequest):
    return await _run(workers.job_simulate, req.model_dump())


@app.post("/api/equilibrium")
async def equilibrium_endpoint(req: EquilibriumRequest):
    return await _run(workers.job_equilibrium, req.model_dump())


@app.post("/api/analysis/sensitivity")
async def sensitivity_endpoint(req: SensitivityRequest):
    return await _run(workers.job_sensitivity, req.model_dump(), timeout=settings.max_wall_time * 2)


@app.post("/api/analysis/rop")
async def rop_endpoint(req: RopRequest):
    return await _run(workers.job_rop, req.model_dump())


@app.post("/api/analysis/fit")
async def fit_endpoint(req: FitRequest):
    return await _run(workers.job_fit, req.model_dump(), timeout=settings.max_wall_time * 2)


@app.post("/api/analysis/order")
def order_endpoint(req: OrderFitRequest):
    from ..analysis import fit_reaction_order

    if len(req.t) != len(req.c):
        raise HTTPException(422, "t und c müssen gleich lang sein")
    return fit_reaction_order(req.t, req.c, req.concentration_unit)


@app.post("/api/analysis/arrhenius")
def arrhenius_endpoint(req: ArrheniusRequest):
    from ..analysis import fit_arrhenius

    if len(req.T) != len(req.k):
        raise HTTPException(422, "T und k müssen gleich lang sein")
    return fit_arrhenius(req.T, req.k, req.modified)


@app.post("/api/numerics/analyze")
async def numerics_endpoint(req: NumericsRequest):
    return await _run(workers.job_numerics, req.model_dump(), timeout=req.budget_s * 3 + 30)


@app.get("/api/numerics/methods")
def numerics_methods():
    try:
        from ..numerics import LINEAR_SOLVERS, METHODS, NONLINEAR_SOLVERS
    except ImportError:
        return {"methods": [{"id": m, "label_de": f"SciPy {m.split(':')[1]}", "family": "scipy"}
                            for m in ("scipy:BDF", "scipy:Radau", "scipy:LSODA", "scipy:RK45")],
                "linear_solvers": [], "nonlinear_solvers": []}

    def norm(reg):
        items = reg.values() if isinstance(reg, dict) else reg
        return [i if isinstance(i, dict) else (asdict(i) if hasattr(i, "__dataclass_fields__") else dict(i))
                for i in items]

    auto = {"id": "auto", "label_de": "Automatisch (Radau5 für ≤ 40 Zustände, sonst BDF)",
            "description_de": "Wählt anhand der Systemgröße; die Numerik-Analyse liefert eine problemangepasste Empfehlung.",
            "family": "Automatisch", "order": None, "implicit": True, "adaptive": True}
    return {"methods": [auto, *norm(METHODS)], "linear_solvers": norm(LINEAR_SOLVERS),
            "nonlinear_solvers": norm(NONLINEAR_SOLVERS)}


# ----------------------------------------------------------------------------- open data
@app.get("/api/species/lookup")
def species_lookup(q: str = Query(min_length=1, max_length=200), online: bool = True):
    from ..data import lookup_species

    info = lookup_species(q, online=online and settings.allow_online_data)
    return info.to_dict()


@app.get("/api/thermo/nasa")
def nasa_lookup(name: str | None = None, formula: str | None = None):
    from ..composition import parse_formula
    from ..data import nasa_db
    from ..data.lookup import thermo_candidates

    if name:
        hit = nasa_db.by_name(name)
        if hit is None:
            raise HTTPException(404, f"'{name}' nicht in der NASA-Datenbank")
        return {"name": hit[0]["name"], "thermo": hit[1].to_dict(), "source": hit[1].source}
    if formula:
        comp, charge = parse_formula(formula)
        return thermo_candidates(comp, charge)
    raise HTTPException(422, "name oder formula angeben")


# ----------------------------------------------------------------------------- frontend
if settings.static_dir and (settings.static_dir / "index.html").exists():
    static_dir = settings.static_dir
    app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        candidate = (static_dir / path).resolve()
        if path and candidate.is_file() and static_dir.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(static_dir / "index.html")
