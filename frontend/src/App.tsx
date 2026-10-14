import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "./lib/api";
import { applyReactorPreset, loadLocal, emptyProject, projectFromDocument, reactorPayload, saveLocal, simulationPayload, toDocument } from "./lib/project";
import type { ExampleMeta, Issue, MethodInfo, Project, SimulationResult } from "./lib/types";
import { Alert } from "./components/ui";
import { getThemeChoice, setThemeChoice, type ThemeChoice } from "./components/theme";
import { MechanismView } from "./views/MechanismView";
import { ReactorView } from "./views/ReactorView";
import { ResultsView } from "./views/ResultsView";
import { AnalysisView } from "./views/AnalysisView";
import { NumericsView } from "./views/NumericsView";
import { NetworkView } from "./views/NetworkView";
import { DataView } from "./views/DataView";
import { ExamplesDialog } from "./views/ExamplesDialog";

type Tab = "mechanism" | "reactor" | "results" | "analysis" | "numerics" | "network" | "data";
const TABS: { id: Tab; label: string }[] = [
  { id: "mechanism", label: "Mechanismus" },
  { id: "reactor", label: "Reaktor & Löser" },
  { id: "results", label: "Ergebnisse" },
  { id: "analysis", label: "Analyse" },
  { id: "numerics", label: "Numerik" },
  { id: "network", label: "Netzwerk" },
  { id: "data", label: "Stoffdaten" },
];

export interface Health {
  version: string;
  extension: boolean;
  numerics: boolean;
  online_data: boolean;
}

export interface AppContext {
  project: Project;
  update: (fn: (p: Project) => void) => void;
  setProject: (p: Project) => void;
  issues: Issue[];
  validating: boolean;
  result: SimulationResult | null;
  runSimulation: () => Promise<void>;
  simulating: boolean;
  simError: string | null;
  cancelSimulation: () => void;
  methods: { methods: MethodInfo[]; linear_solvers: MethodInfo[]; nonlinear_solvers: MethodInfo[] };
  health: Health | null;
  goto: (t: Tab) => void;
  notify: (msg: string, kind?: "success" | "error" | "info") => void;
}

export function mechanismPayload(p: Project) {
  return { document: toDocument(p) };
}

export default function App() {
  const [project, setProjectState] = useState<Project>(() => loadLocal() ?? emptyProject());
  const [tab, setTab] = useState<Tab>("mechanism");
  const [issues, setIssues] = useState<Issue[]>([]);
  const [validating, setValidating] = useState(false);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [simulating, setSimulating] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);
  const [methods, setMethods] = useState<AppContext["methods"]>({ methods: [], linear_solvers: [], nonlinear_solvers: [] });
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [showExamples, setShowExamples] = useState(false);
  const [toast, setToast] = useState<{ msg: string; kind: "success" | "error" | "info" } | null>(null);
  const [theme, setTheme] = useState<ThemeChoice>(getThemeChoice);
  const abortRef = useRef<AbortController | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => setThemeChoice(theme), [theme]);

  const update = useCallback((fn: (p: Project) => void) => {
    setProjectState((prev) => {
      const next = structuredClone(prev);
      fn(next);
      return next;
    });
  }, []);
  const setProject = useCallback((p: Project) => {
    setProjectState(p);
    setResult(null);
    setSimError(null);
  }, []);

  useEffect(() => {
    const id = setTimeout(() => saveLocal(project), 500);
    return () => clearTimeout(id);
  }, [project]);

  const notify = useCallback((msg: string, kind: "success" | "error" | "info" = "info") => {
    setToast({ msg, kind });
    window.setTimeout(() => setToast(null), 4000);
  }, []);

  useEffect(() => {
    api.get<Health>("/api/health").then(setHealth).catch((e) => setHealthError((e as Error).message));
    api.get<AppContext["methods"]>("/api/numerics/methods").then(setMethods).catch(() => undefined);
    if (!loadLocal()) setShowExamples(true);
  }, []);

  // Debounced validation of the mechanism by the backend.
  const docKey = useMemo(() => JSON.stringify(toDocument(project)), [project.species, project.reactions, project.units, project.phase, project.name]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!project.species.length) {
      setIssues([]);
      return;
    }
    const ctrl = new AbortController();
    const id = setTimeout(() => {
      setValidating(true);
      api.post<{ issues: Issue[] }>("/api/mechanism/parse", { document: JSON.parse(docKey) }, ctrl.signal)
        .then((r) => setIssues(r.issues))
        .catch((e) => {
          if ((e as Error).name !== "AbortError") setIssues([{ level: "error", message: (e as Error).message, reaction: null, species: null }]);
        })
        .finally(() => setValidating(false));
    }, 450);
    return () => {
      clearTimeout(id);
      ctrl.abort();
    };
  }, [docKey, project.species.length]);

  const runSimulation = useCallback(async () => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setSimulating(true);
    setSimError(null);
    try {
      const res = await api.post<SimulationResult>("/api/simulate", {
        mechanism: mechanismPayload(project), reactor: reactorPayload(project), simulation: simulationPayload(project),
        concentration_unit: project.reactor.concentration_unit === "mol/m3" ? "mol/m3" : project.reactor.concentration_unit,
        include_heat_release: project.phase === "ideal-gas" && project.reactor.energy !== "isothermal",
      }, ctrl.signal);
      setResult(res);
      setTab("results");
      if (!res.success) notify(`Integration unvollständig: ${res.message}`, "error");
    } catch (e) {
      if ((e as Error).name !== "AbortError") setSimError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setSimulating(false);
    }
  }, [project, notify]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        void runSimulation();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [runSimulation]);

  const ctx: AppContext = {
    project, update, setProject, issues, validating, result, runSimulation, simulating, simError,
    cancelSimulation: () => abortRef.current?.abort(), methods, health, goto: setTab, notify,
  };

  const loadExample = async (ex: ExampleMeta) => {
    const data = await api.get<{ meta: ExampleMeta; document: Record<string, unknown> }>(`/api/examples/${ex.id}`);
    setProject(projectFromDocument(data.document, data.meta));
    setShowExamples(false);
    setTab("mechanism");
    notify(`Beispiel „${data.meta.title}" geladen`, "success");
  };

  const importFile = async (file: File) => {
    const text = await file.text();
    try {
      if (file.name.endsWith(".akp.json")) {
        const p = JSON.parse(text) as Project;
        if (p.version !== 1) throw new Error("Unbekanntes Projektformat");
        setProject(p);
      } else {
        const body = file.name.endsWith(".kin") ? { kin: text } : { yaml: text };
        const parsed = await api.post<{ document: Record<string, unknown>; metadata: Record<string, any>; name: string }>("/api/mechanism/parse", body);
        const p = projectFromDocument(parsed.document);
        p.name = file.name.replace(/\.(ya?ml|kin|json)$/i, "");
        if (parsed.metadata?.reactor) applyReactorPreset(p, parsed.metadata.reactor);
        setProject(p);
        for (const w of parsed.metadata?.import_warnings ?? []) notify(w, "info");
      }
      setTab("mechanism");
      notify(`„${file.name}" importiert`, "success");
    } catch (e) {
      notify(`Import fehlgeschlagen: ${(e as Error).message}`, "error");
    }
  };

  const download = (name: string, content: string, type: string) => {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportYaml = async () => {
    try {
      const text = await api.post<string>("/api/mechanism/export", mechanismPayload(project));
      download(`${slug(project.name)}.yaml`, text, "application/x-yaml");
    } catch (e) {
      notify(`Export fehlgeschlagen: ${(e as Error).message}`, "error");
    }
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img src="/favicon.svg" alt="" width={26} height={26} />
          <span>AutoKinetics</span>
        </div>
        <input
          className="project-name"
          value={project.name}
          aria-label="Projektname"
          onChange={(e) => update((p) => void (p.name = e.target.value))}
        />
        <nav className="top-actions" aria-label="Projekt">
          <button className="ghost" onClick={() => { if (confirm("Neues leeres Projekt anlegen? Ungespeicherte Änderungen gehen verloren.")) setProject(emptyProject()); }}>Neu</button>
          <button className="ghost" onClick={() => setShowExamples(true)}>Beispiele</button>
          <button className="ghost" onClick={() => fileRef.current?.click()}>Importieren</button>
          <input ref={fileRef} type="file" hidden accept=".yaml,.yml,.kin,.json" onChange={(e) => { const f = e.target.files?.[0]; if (f) void importFile(f); e.target.value = ""; }} />
          <button className="ghost" onClick={() => void exportYaml()} disabled={!project.species.length}>YAML-Export</button>
          <button className="ghost" onClick={() => download(`${slug(project.name)}.akp.json`, JSON.stringify(project, null, 2), "application/json")}>Projekt speichern</button>
          <select aria-label="Farbschema" value={theme} onChange={(e) => setTheme(e.target.value as ThemeChoice)}>
            <option value="auto">Auto</option>
            <option value="light">Hell</option>
            <option value="dark">Dunkel</option>
          </select>
          <button className="primary" onClick={() => void runSimulation()} disabled={simulating || !project.species.length} title="Strg+Enter">
            {simulating ? "Rechnet …" : "▶ Simulieren"}
          </button>
        </nav>
      </header>
      <nav className="tabs" role="tablist" aria-label="Bereiche">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} className={tab === t.id ? "active" : ""} onClick={() => setTab(t.id)}>
            {t.label}
            {t.id === "mechanism" && issues.some((i) => i.level === "error") && <span className="badge-dot" aria-label="Fehler im Mechanismus" />}
            {t.id === "results" && result && <span className="badge-count">✓</span>}
          </button>
        ))}
      </nav>
      <main className="content">
        {healthError && <Alert kind="error">{healthError}</Alert>}
        {health && !health.extension && <Alert kind="warning">Die C++-Erweiterung ist nicht gebaut – Simulationen sind nicht möglich.</Alert>}
        {tab === "mechanism" && <MechanismView ctx={ctx} />}
        {tab === "reactor" && <ReactorView ctx={ctx} />}
        {tab === "results" && <ResultsView ctx={ctx} />}
        {tab === "analysis" && <AnalysisView ctx={ctx} />}
        {tab === "numerics" && <NumericsView ctx={ctx} />}
        {tab === "network" && <NetworkView ctx={ctx} />}
        {tab === "data" && <DataView ctx={ctx} />}
      </main>
      <footer className="statusbar">
        <span>{project.species.length} Spezies · {project.reactions.length} Reaktionen · {project.phase === "ideal-gas" ? "Gasphase" : "Flüssigphase"}</span>
        <span>{health ? `Backend ${health.version} · C++ ${health.extension ? "✓" : "✗"} · Numerik ${health.numerics ? "✓" : "–"}` : "Backend …"}</span>
      </footer>
      {showExamples && <ExamplesDialog onClose={() => setShowExamples(false)} onLoad={loadExample} />}
      {toast && (
        <div className={`toast toast-${toast.kind}`} role="status">
          {toast.msg}
        </div>
      )}
    </div>
  );
}

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9äöüß]+/gi, "_").replace(/^_|_$/g, "") || "mechanismus";
