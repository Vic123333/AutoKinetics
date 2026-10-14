import Plotly from "plotly.js-dist-min";
import type { Config, Data, Layout } from "plotly.js";
import { useEffect, useRef } from "react";
import { chartTheme, type ChartTheme } from "../lib/palette";
import { useDarkMode } from "./theme";

interface Props {
  data: Data[];
  layout?: Partial<Layout>;
  height?: number;
  title?: string;
  ariaLabel: string;
}

export function themedLayout(theme: ChartTheme, layout: Partial<Layout> = {}): Partial<Layout> {
  const axis = (a: Partial<Layout["xaxis"]> | undefined) => ({
    gridcolor: theme.grid, linecolor: theme.axis, zerolinecolor: theme.axis, tickfont: { color: theme.muted, size: 11 },
    title: { font: { color: theme.ink2, size: 12 }, ...(a?.title as object) }, automargin: true, ...a,
  });
  return {
    ...layout,
    paper_bgcolor: theme.surface,
    plot_bgcolor: theme.surface,
    font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: theme.ink, size: 12 },
    margin: { l: 64, r: 16, t: 16, b: 48, ...layout.margin },
    hovermode: layout.hovermode ?? "x unified",
    hoverlabel: { bgcolor: theme.surface, bordercolor: theme.axis, font: { color: theme.ink } },
    legend: { orientation: "h", y: -0.18, font: { color: theme.ink2 }, ...layout.legend },
    xaxis: axis(layout.xaxis),
    yaxis: axis(layout.yaxis),
  } as Partial<Layout>;
}

const CONFIG: Partial<Config> = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
  toImageButtonOptions: { format: "svg", filename: "autokinetics" },
  locale: "de",
};

export function Plot({ data, layout, height = 380, ariaLabel }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const dark = useDarkMode();
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const theme = chartTheme(dark);
    void Plotly.react(el, data, { ...themedLayout(theme, layout), height }, CONFIG);
  }, [data, layout, height, dark]);
  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el) Plotly.purge(el);
    };
  }, []);
  return <div ref={ref} className="plot" role="img" aria-label={ariaLabel} />;
}
