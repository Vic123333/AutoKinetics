"""Command line interface: ``autokinetics serve|run|example|validate|lookup|examples``."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .errors import AutoKineticsError


def _write_csv(result, path: Path, unit: str) -> None:
    factor = {"mol/L": 1e-3, "mol/m3": 1.0, "mmol/L": 1.0}[unit]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t [s]", "T [K]", "P [Pa]"] + [f"{s} [{unit}]" for s in result.species_names])
        for k in range(len(result.t)):
            w.writerow([f"{result.t[k]:.10g}", f"{result.T[k]:.8g}", f"{result.P[k]:.8g}"]
                       + [f"{v * factor:.10g}" for v in result.concentrations[k]])


def _print_summary(result) -> None:
    print(f"Erfolg: {result.success} – {result.message}")
    print(f"Rechenzeit: {result.wall_time:.3f} s, Punkte: {len(result.t)}, Statistik: {result.stats}")
    for key, val in result.metrics.items():
        if val:
            print(f"  {key}: {val}")
    if "element_conservation_max" in result.diagnostics:
        print(f"  Elementerhaltung (max. rel. Abweichung): {result.diagnostics['element_conservation_max']:.2e}")


def cmd_serve(args) -> int:
    import os

    import uvicorn

    if args.workers:
        os.environ["AUTOKINETICS_WORKERS"] = str(args.workers)
    uvicorn.run("autokinetics.api.app:app", host=args.host, port=args.port, log_level="info")
    return 0


def _load_dict(path: str) -> dict:
    from .io.yaml_format import safe_load_yaml

    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text) if path.endswith(".json") else safe_load_yaml(text)


def cmd_run(args) -> int:
    from .io import load_mechanism
    from .presets import options_from_dict, reactor_spec_from_dict
    from .simulation import simulate

    mech = load_mechanism(args.mechanism)
    preset = mech.metadata or {}
    reactor = _load_dict(args.reactor) if args.reactor else preset.get("reactor")
    sim = dict(preset.get("simulation") or {})
    if args.t_end:
        sim["t_end"] = args.t_end
    if args.method:
        sim["method"] = args.method
    if not reactor or "t_end" not in sim:
        print("Reaktor (--reactor) und Simulationsdauer (--t-end) angeben", file=sys.stderr)
        return 2
    result = simulate(mech, reactor_spec_from_dict(reactor, mech), options_from_dict(sim))
    _print_summary(result)
    if args.out:
        _write_csv(result, Path(args.out), args.unit)
        print(f"Ergebnis geschrieben: {args.out}")
    return 0 if result.success else 1


def cmd_example(args) -> int:
    from .library import load_example
    from .presets import options_from_dict, reactor_spec_from_dict
    from .simulation import simulate

    mech, meta, _ = load_example(args.id)
    sim = dict(meta.preset["simulation"])
    if args.method:
        sim["method"] = args.method
    result = simulate(mech, reactor_spec_from_dict(meta.preset["reactor"], mech), options_from_dict(sim))
    print(f"{meta.title} ({meta.id})")
    _print_summary(result)
    if args.out:
        _write_csv(result, Path(args.out), args.unit)
    return 0 if result.success else 1


def cmd_examples(_args) -> int:
    from .library import list_examples

    for e in list_examples():
        print(f"{e.id:30s} {e.category:24s} {e.n_species:4d} Spezies {e.n_reactions:5d} Reaktionen  {e.title}")
    return 0


def cmd_validate(args) -> int:
    from .io import load_mechanism

    mech = load_mechanism(args.mechanism)
    issues = mech.validate()
    print(f"{mech.name}: {mech.n_species} Spezies, {mech.n_reactions} Reaktionen, Phase {mech.phase}")
    for i in issues:
        print(f"  [{i.level}] {i.message}")
    return 1 if any(i.level == "error" for i in issues) else 0


def cmd_lookup(args) -> int:
    from .data import lookup_species

    info = lookup_species(args.query, online=not args.offline)
    print(json.dumps({k: v for k, v in info.to_dict().items() if k not in ("thermo", "thermo_candidates")},
                     indent=2, ensure_ascii=False, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="autokinetics", description="Simulation chemischer Reaktionsnetzwerke")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="Web-Oberfläche + REST-API starten")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--workers", type=int, default=0, help="Rechenprozesse (Standard: CPU/2)")
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("run", help="Mechanismus-Datei simulieren")
    s.add_argument("mechanism")
    s.add_argument("--reactor", help="Reaktor-Definition (YAML/JSON); Standard: Preset der Datei")
    s.add_argument("--t-end", type=float)
    s.add_argument("--method")
    s.add_argument("--out", help="CSV-Ausgabe")
    s.add_argument("--unit", default="mol/L", choices=["mol/L", "mol/m3", "mmol/L"])
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("example", help="Beispiel aus der Bibliothek rechnen")
    s.add_argument("id")
    s.add_argument("--method")
    s.add_argument("--out")
    s.add_argument("--unit", default="mol/L", choices=["mol/L", "mol/m3", "mmol/L"])
    s.set_defaults(fn=cmd_example)
    sub.add_parser("examples", help="Beispiele auflisten").set_defaults(fn=cmd_examples)
    s = sub.add_parser("validate", help="Mechanismus prüfen (Element-/Ladungsbilanz, Thermo …)")
    s.add_argument("mechanism")
    s.set_defaults(fn=cmd_validate)
    s = sub.add_parser("lookup", help="Stoffdaten nachschlagen (PubChem, NIST, NASA)")
    s.add_argument("query")
    s.add_argument("--offline", action="store_true")
    s.set_defaults(fn=cmd_lookup)
    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except AutoKineticsError as e:
        print(f"Fehler: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
