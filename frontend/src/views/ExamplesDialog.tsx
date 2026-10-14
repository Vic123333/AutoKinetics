import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import type { ExampleMeta } from "../lib/types";
import { Alert, Spinner } from "../components/ui";

const QUALITY: Record<string, string> = { literature: "Literaturwerte", illustrative: "illustrativ", mixed: "gemischt" };

export function ExamplesDialog({ onClose, onLoad }: { onClose: () => void; onLoad: (e: ExampleMeta) => Promise<void> }) {
  const [examples, setExamples] = useState<ExampleMeta[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    ref.current?.showModal();
    api.get<ExampleMeta[]>("/api/examples").then(setExamples).catch((e) => setErr((e as Error).message));
  }, []);
  const categories = examples ? [...new Set(examples.map((e) => e.category))] : [];
  return (
    <dialog ref={ref} className="dialog" onClose={onClose} aria-labelledby="ex-title">
      <header className="card-head">
        <h2 id="ex-title">Beispielbibliothek</h2>
        <button className="ghost" onClick={onClose} aria-label="Schließen">✕</button>
      </header>
      {err && <Alert kind="error">{err}</Alert>}
      {!examples && !err && <Spinner label="lade …" />}
      {categories.map((c) => (
        <section key={c}>
          <h3>{c}</h3>
          <div className="example-grid">
            {examples!.filter((e) => e.category === c).map((e) => (
              <button key={e.id} className="example" disabled={!!loading} onClick={async () => { setLoading(e.id); try { await onLoad(e); } catch (x) { setErr((x as Error).message); } finally { setLoading(null); } }}>
                <span className="example-title">{e.title}</span>
                <span className="example-meta">{e.phase === "ideal-gas" ? "Gas" : "Flüssig"} · {e.n_species} Spezies · {e.n_reactions} Reaktionen · {QUALITY[e.parameter_quality] ?? e.parameter_quality}</span>
                <span className="example-summary">{e.summary}</span>
                {loading === e.id && <Spinner label="lade …" />}
              </button>
            ))}
          </div>
        </section>
      ))}
    </dialog>
  );
}
