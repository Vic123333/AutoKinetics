import { useState } from "react";
import { mechanismPayload, type AppContext } from "../App";
import { api } from "../lib/api";
import { reactorPayload, simulationPayload } from "../lib/project";
import { Alert, Card, Field, NumInput, Spinner, useElapsed } from "../components/ui";

export function NumericsView({ ctx }: { ctx: AppContext }) {
  const [budget, setBudget] = useState(20);
  const [report, setReport] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const elapsed = useElapsed(busy);
  const run = async () => {
    setBusy(true);
    setErr(null);
    try {
      setReport(await api.post("/api/numerics/analyze", { mechanism: mechanismPayload(ctx.project), reactor: reactorPayload(ctx.project), simulation: simulationPayload(ctx.project), budget_s: budget }));
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="stack">
      <Card title="Automatische Löseroptimierung">
        <p className="muted">Stabilitäts-, Konsistenz- und Konvergenzanalyse am aktuellen Problem und Auswahl des günstigsten Verfahrens.</p>
        {ctx.health && !ctx.health.numerics && <Alert kind="warning">Numerik-Modul nicht verfügbar.</Alert>}
        <div className="row">
          <Field label="Zeitbudget [s]"><NumInput value={budget} min={1} onChange={setBudget} /></Field>
          <button className="primary" disabled={busy} onClick={() => void run()}>Analyse starten</button>
          {busy && <Spinner label={`analysiert … ${elapsed.toFixed(0)} s`} />}
        </div>
        {err && <Alert kind="error">{err}</Alert>}
      </Card>
      {report && <Card title="Bericht"><pre className="code">{JSON.stringify(report, null, 2).slice(0, 20000)}</pre></Card>}
    </div>
  );
}
