import { useState } from "react";
import type { AppContext } from "../App";
import { api } from "../lib/api";
import { newSpecies } from "../lib/project";
import { Alert, Card, Field, Spinner, fmtNum } from "../components/ui";

export function DataView({ ctx }: { ctx: AppContext }) {
  const [q, setQ] = useState("");
  const [info, setInfo] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [target, setTarget] = useState("");
  const search = async () => {
    if (!q.trim()) return;
    setBusy(true);
    setErr(null);
    setInfo(null);
    try {
      setInfo(await api.get(`/api/species/lookup?q=${encodeURIComponent(q.trim())}`));
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const apply = (name: string, thermo: any, source?: string) =>
    ctx.update((p) => {
      let s = p.species.find((x) => x.name === name);
      if (!s) {
        s = newSpecies(name);
        p.species.push(s);
      }
      if (info?.composition && Object.keys(info.composition).length) {
        s.composition = info.composition;
        s.charge = info.charge ?? 0;
        s.formula = info.formula ?? s.formula;
      }
      if (thermo) s.thermo = { ...thermo, note: source ?? "" };
      s.extra = { ...(s.extra ?? {}), ...(info?.pubchem ? { pubchem: info.pubchem.cid } : {}), ...(info?.cas ? { cas: info.cas } : {}) };
    });

  return (
    <div className="stack">
      <Card title="Stoffdaten nachschlagen">
        <p className="muted">Identität aus <b>PubChem</b>, Gasphasen-Thermochemie aus dem <b>NIST Chemistry WebBook</b>, offline aus der <b>NASA-Thermodatenbank</b> (McBride et al., 748 Spezies). Ergebnisse werden zwischengespeichert.</p>
        {ctx.health && !ctx.health.online_data && <Alert kind="info">Server im Offline-Modus – nur NASA-Datenbank verfügbar.</Alert>}
        <div className="row">
          <Field label="Name, Summenformel oder CAS" wide>
            <input value={q} placeholder="z. B. Ethanol, CO2, 64-17-5" onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && void search()} />
          </Field>
          <button className="primary" onClick={() => void search()} disabled={busy}>Suchen</button>
          {busy && <Spinner label="frage PubChem/NIST …" />}
        </div>
        {err && <Alert kind="error">{err}</Alert>}
      </Card>
      {info && (
        <Card title={info.name ?? info.query}>
          {info.warnings?.map((w: string) => <Alert key={w} kind="warning">{w}</Alert>)}
          <div className="tiles">
            <div className="tile"><div className="tile-label">Summenformel</div><div className="tile-value">{info.formula ?? "–"}</div></div>
            <div className="tile"><div className="tile-label">Molare Masse</div><div className="tile-value">{info.molecular_weight ? `${fmtNum(info.molecular_weight, 6)} g/mol` : "–"}</div></div>
            <div className="tile"><div className="tile-label">ΔfH°(298 K, Gas)</div><div className="tile-value">{info.dHf_298 !== null ? `${fmtNum(info.dHf_298 / 1e3, 5)} kJ/mol` : "–"}</div></div>
            <div className="tile"><div className="tile-label">S°(298 K)</div><div className="tile-value">{info.S_298 !== null ? `${fmtNum(info.S_298, 5)} J/(mol·K)` : "–"}</div></div>
            <div className="tile"><div className="tile-label">c_p(298 K)</div><div className="tile-value">{info.cp_298 !== null ? `${fmtNum(info.cp_298, 4)} J/(mol·K)` : "–"}</div></div>
          </div>
          <p>
            {info.cas && <>CAS {info.cas} · </>}
            {info.pubchem && <a href={info.pubchem.url} target="_blank" rel="noreferrer">PubChem CID {info.pubchem.cid}</a>}
            {info.cas && <> · <a href={`https://webbook.nist.gov/cgi/cbook.cgi?ID=C${info.cas.replace(/-/g, "")}&Units=SI&Mask=1`} target="_blank" rel="noreferrer">NIST WebBook</a></>}
            {info.pubchem?.smiles && <> · SMILES <code>{info.pubchem.smiles}</code></>}
          </p>
          {info.thermo_source && <p>Thermodaten: {info.thermo_source}</p>}
          <div className="row">
            <Field label="Übernehmen als Spezies">
              <input value={target} placeholder={info.formula ?? info.query} onChange={(e) => setTarget(e.target.value)} list="species-names" />
              <datalist id="species-names">{ctx.project.species.map((s) => <option key={s.key} value={s.name} />)}</datalist>
            </Field>
            <button className="primary" onClick={() => { const n = target || info.formula || info.query; apply(n, info.thermo, info.thermo_source); ctx.notify(`${n} übernommen`, "success"); }}>In Mechanismus übernehmen</button>
          </div>
          {info.thermo_candidates?.length > 1 && (
            <>
              <h3>Isomere in der NASA-Datenbank</h3>
              <table className="grid compact">
                <thead><tr><th>Eintrag</th><th className="num">ΔfH° [kJ/mol]</th><th className="num">S° [J/(mol·K)]</th><th /></tr></thead>
                <tbody>{info.thermo_candidates.map((c: any) => (
                  <tr key={c.name}><td>{c.name}</td><td className="num">{fmtNum(c.dHf_298 / 1e3, 5)}</td><td className="num">{fmtNum(c.S_298, 5)}</td>
                    <td><button className="ghost small" onClick={() => { const n = target || info.formula || c.name; apply(n, c.thermo, c.source); ctx.notify(`${n}: Thermodaten ${c.name} übernommen`, "success"); }}>diese Daten verwenden</button></td></tr>
                ))}</tbody>
              </table>
            </>
          )}
        </Card>
      )}
    </div>
  );
}
