import { forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY, type SimulationLinkDatum, type SimulationNodeDatum } from "d3-force";
import { useEffect, useMemo, useRef, useState } from "react";
import type { AppContext } from "../App";
import { parseEquation } from "../lib/chem";
import { Card } from "../components/ui";

interface Node extends SimulationNodeDatum {
  id: string;
  kind: "species" | "reaction";
  label: string;
}
interface Link extends SimulationLinkDatum<Node> {
  role: "in" | "out";
  nu: number;
}

export function NetworkView({ ctx }: { ctx: AppContext }) {
  const { project, result } = ctx;
  const [positions, setPositions] = useState<Map<string, { x: number; y: number }>>(new Map());
  const [hover, setHover] = useState<string | null>(null);
  const W = 1000, H = 640;
  const aggregated = project.reactions.length > 60;
  const tooBig = false;
  const MAX_EDGES = 140;

  const { nodes, links } = useMemo(() => {
    const nodes: Node[] = project.species.map((s) => ({ id: `s:${s.name}`, kind: "species", label: s.name }));
    const links: Link[] = [];
    if (aggregated) {
      // species graph: edge reactant -> product, weight = number of reactions linking them
      const w = new Map<string, number>();
      for (const r of project.reactions) {
        const p = parseEquation(r.equation);
        if (!p) continue;
        for (const a of Object.keys(p.reactants)) for (const b of Object.keys(p.products)) {
          if (a === b) continue;
          const key = `${a}\u0000${b}`;
          w.set(key, (w.get(key) ?? 0) + 1);
        }
      }
      const top = [...w.entries()].sort((x, y) => y[1] - x[1]).slice(0, MAX_EDGES);
      const usedNames = new Set(top.flatMap(([k]) => k.split("\u0000")));
      for (const [k, n] of top) {
        const [a, b] = k.split("\u0000");
        if (project.species.some((s) => s.name === a) && project.species.some((s) => s.name === b)) links.push({ source: `s:${a}`, target: `s:${b}`, role: "out", nu: n });
      }
      return { nodes: nodes.filter((n) => usedNames.has(n.label)), links };
    }
    project.reactions.forEach((r, j) => {
      const p = parseEquation(r.equation);
      if (!p) return;
      nodes.push({ id: `r:${j}`, kind: "reaction", label: `R${j + 1}: ${r.equation}` });
      for (const [n, nu] of Object.entries(p.reactants)) if (project.species.some((s) => s.name === n)) links.push({ source: `s:${n}`, target: `r:${j}`, role: "in", nu });
      for (const [n, nu] of Object.entries(p.products)) if (project.species.some((s) => s.name === n)) links.push({ source: `r:${j}`, target: `s:${n}`, role: "out", nu });
    });
    return { nodes, links };
  }, [project.species, project.reactions, aggregated]);

  const simRef = useRef<ReturnType<typeof forceSimulation<Node>> | null>(null);
  useEffect(() => {
    if (tooBig) return;
    const ns = nodes.map((n) => ({ ...n }));
    const ls = links.map((l) => ({ ...l }));
    const sim = forceSimulation<Node>(ns)
      .force("link", forceLink<Node, Link>(ls).id((d) => d.id).distance(aggregated ? 90 : 70).strength(aggregated ? 0.25 : 0.6))
      .force("charge", forceManyBody().strength(aggregated ? -320 : -380))
      .force("collide", forceCollide<Node>((d) => (d.kind === "species" ? 34 : 12)))
      .force("x", forceX(W / 2).strength(0.04))
      .force("y", forceY(H / 2).strength(0.06))
      .stop();
    for (let i = 0; i < 300; i++) sim.tick();
    simRef.current = sim;
    // fit the layout into the drawing area (auto-zoom)
    const xs = ns.map((n) => n.x ?? 0), ys = ns.map((n) => n.y ?? 0);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const pad = 50;
    const sc = Math.min((W - 2 * pad) / Math.max(x1 - x0, 1), (H - 2 * pad) / Math.max(y1 - y0, 1), 3);
    const ox = (W - sc * (x1 - x0)) / 2, oy = (H - sc * (y1 - y0)) / 2;
    setPositions(new Map(ns.map((n) => [n.id, { x: ox + sc * ((n.x ?? 0) - x0), y: oy + sc * ((n.y ?? 0) - y0) }])));
  }, [nodes, links, tooBig, aggregated]);

  if (!project.reactions.length) return <Card title="Reaktionsnetzwerk"><p>Noch keine Reaktionen.</p></Card>;
  if (tooBig) return <Card title="Reaktionsnetzwerk"><p>Das Netzwerk ist mit {project.reactions.length} Reaktionen zu groß für eine lesbare Darstellung (Grenze 400). Nutzen Sie die Reaktionsflussanalyse unter „Analyse".</p></Card>;

  const finalConc = result ? new Map(result.species.map((s, i) => [s, result.concentrations[i].at(-1) ?? 0])) : null;
  const maxC = finalConc ? Math.max(...finalConc.values(), 1e-300) : 1;
  const related = hover ? new Set(links.flatMap((l) => {
    const s = typeof l.source === "string" ? l.source : (l.source as Node).id;
    const t = typeof l.target === "string" ? l.target : (l.target as Node).id;
    return s === hover || t === hover ? [s, t] : [];
  })) : null;

  return (
    <Card title="Reaktionsnetzwerk" actions={<span className="muted">{aggregated
      ? `Verdichtet: Pfeil Edukt → Produkt, Strichstärke ∝ Anzahl Reaktionen (stärkste ${MAX_EDGES} Verbindungen)`
      : "Kreise: Spezies · Quadrate: Reaktionen"}{finalConc ? " · Kreisfläche ∝ Endkonzentration" : ""} · Maus darüber hebt Nachbarn hervor</span>}>
      <svg viewBox={`0 0 ${W} ${H}`} className="network" role="img" aria-label="Reaktionsnetzwerk als Graph">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" className="net-arrow" />
          </marker>
        </defs>
        {links.map((l, k) => {
          const s = positions.get(typeof l.source === "string" ? l.source : (l.source as Node).id);
          const t = positions.get(typeof l.target === "string" ? l.target : (l.target as Node).id);
          if (!s || !t) return null;
          const tid = typeof l.target === "string" ? l.target : (l.target as Node).id;
          const sid = typeof l.source === "string" ? l.source : (l.source as Node).id;
          const dim = related && !(related.has(sid) && related.has(tid));
          const r = tid.startsWith("s:") ? 18 : 6;
          const dx = t.x - s.x, dy = t.y - s.y, d = Math.hypot(dx, dy) || 1;
          const width = aggregated ? Math.min(1 + l.nu * 0.8, 6) : l.nu > 1 ? 2.5 : 1.4;
          return <line key={k} x1={s.x} y1={s.y} x2={t.x - (dx / d) * r} y2={t.y - (dy / d) * r} className={`net-link ${l.role} ${dim ? "dim" : ""}`} markerEnd="url(#arrow)" strokeWidth={width}><title>{`${sid.slice(2)} → ${tid.slice(2)}${aggregated ? `: ${l.nu} Reaktion(en)` : ""}`}</title></line>;
        })}
        {nodes.map((n) => {
          const p = positions.get(n.id);
          if (!p) return null;
          const dim = related && !related.has(n.id) && hover !== n.id;
          if (n.kind === "reaction")
            return (
              <rect key={n.id} x={p.x - 5} y={p.y - 5} width={10} height={10} rx={2} className={`net-rxn ${dim ? "dim" : ""}`} onMouseEnter={() => setHover(n.id)} onMouseLeave={() => setHover(null)}>
                <title>{n.label}</title>
              </rect>
            );
          const c = finalConc?.get(n.label) ?? 0;
          const r = finalConc ? 8 + 14 * Math.sqrt(Math.max(c, 0) / maxC) : 16;
          return (
            <g key={n.id} className={dim ? "dim" : ""} onMouseEnter={() => setHover(n.id)} onMouseLeave={() => setHover(null)}>
              <circle cx={p.x} cy={p.y} r={r} className="net-species" />
              <text x={p.x} y={p.y + r + 12} textAnchor="middle" className="net-label">{n.label}</text>
              <title>{n.label}{finalConc ? `: ${c.toExponential(3)} ${result?.concentration_unit}` : ""}</title>
            </g>
          );
        })}
      </svg>
    </Card>
  );
}
