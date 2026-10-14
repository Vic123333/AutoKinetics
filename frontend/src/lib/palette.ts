// Categorical palette (validated order, see dataviz reference palette). Never cycled:
// at most 8 series are coloured at once.
export const SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
export const SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
export const MAX_SERIES = 8;

export interface ChartTheme {
  dark: boolean;
  surface: string;
  ink: string;
  ink2: string;
  muted: string;
  grid: string;
  axis: string;
  series: string[];
  diverging: [string, string];
}

export function chartTheme(dark: boolean): ChartTheme {
  return dark
    ? { dark, surface: "#1a1a19", ink: "#f0efec", ink2: "#c3c2b7", muted: "#898781", grid: "#2c2c2a", axis: "#383835", series: SERIES_DARK, diverging: ["#3987e5", "#e66767"] }
    : { dark, surface: "#fcfcfb", ink: "#0b0b0b", ink2: "#52514e", muted: "#898781", grid: "#e1e0d9", axis: "#c3c2b7", series: SERIES_LIGHT, diverging: ["#2a78d6", "#e34948"] };
}

/** Stable colour per entity: the slot follows the entity's position in the selection order
 *  it was first assigned, not its rank in the current data. */
export class ColorAssigner {
  private slots = new Map<string, number>();
  color(name: string, theme: ChartTheme): string | null {
    if (!this.slots.has(name)) {
      const used = new Set(this.slots.values());
      const free = [...Array(MAX_SERIES).keys()].find((i) => !used.has(i));
      if (free === undefined) return null;
      this.slots.set(name, free);
    }
    return theme.series[this.slots.get(name)!];
  }
  release(keep: string[]) {
    for (const k of [...this.slots.keys()]) if (!keep.includes(k)) this.slots.delete(k);
  }
}
