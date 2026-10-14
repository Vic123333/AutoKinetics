import { useEffect, useState, type ReactNode } from "react";

export function Card({ title, actions, children, className }: { title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`card ${className ?? ""}`}>
      {(title || actions) && (
        <header className="card-head">
          {title && <h2>{title}</h2>}
          {actions && <div className="card-actions">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

export function Field({ label, hint, children, wide }: { label: ReactNode; hint?: ReactNode; children: ReactNode; wide?: boolean }) {
  return (
    <label className={`field ${wide ? "wide" : ""}`}>
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

/** Numeric input that keeps the user's text while typing (e.g. "1e-" or "0.") and only commits valid numbers. */
export function NumInput({ value, onChange, min, step, ariaLabel, disabled, allowEmpty, placeholder }: {
  value: number | null | undefined; onChange: (v: number) => void; min?: number; step?: number | "any";
  ariaLabel?: string; disabled?: boolean; allowEmpty?: boolean; placeholder?: string;
}) {
  const [text, setText] = useState(fmt(value));
  useEffect(() => {
    if (Number(text) !== value) setText(fmt(value));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);
  const n = Number(text.replace(",", "."));
  const invalid = text.trim() === "" ? !allowEmpty : !Number.isFinite(n) || (min !== undefined && n < min);
  return (
    <input
      className={`num ${invalid ? "invalid" : ""}`}
      inputMode="decimal"
      value={text}
      aria-label={ariaLabel}
      aria-invalid={invalid}
      disabled={disabled}
      placeholder={placeholder}
      step={step}
      onChange={(e) => {
        setText(e.target.value);
        const v = Number(e.target.value.replace(",", "."));
        if (e.target.value.trim() !== "" && Number.isFinite(v) && (min === undefined || v >= min)) onChange(v);
        else if (allowEmpty && e.target.value.trim() === "") onChange(NaN);
      }}
    />
  );
}

/** Text-or-number input for rate parameters that may carry units (e.g. "15 kcal/mol"). */
export function QtyInput({ value, onChange, ariaLabel }: { value: number | string; onChange: (v: number | string) => void; ariaLabel?: string }) {
  const [text, setText] = useState(fmt(value));
  useEffect(() => {
    if (String(text) !== String(value) && Number(text) !== value) setText(fmt(value));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);
  return (
    <input
      className="num"
      value={text}
      aria-label={ariaLabel}
      onChange={(e) => {
        setText(e.target.value);
        const v = Number(e.target.value.replace(",", "."));
        onChange(e.target.value.trim() !== "" && Number.isFinite(v) ? v : e.target.value);
      }}
    />
  );
}

export function fmt(v: number | string | null | undefined): string {
  if (v === null || v === undefined || (typeof v === "number" && Number.isNaN(v))) return "";
  if (typeof v === "string") return v;
  if (v === 0) return "0";
  const a = Math.abs(v);
  return a >= 1e5 || a < 1e-3 ? v.toExponential(4).replace(/\.?0+e/, "e") : String(+v.toPrecision(8));
}

export function fmtNum(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(digits - 1).replace("e", "·10^").replace(/\^\+?(-?)0*(\d)/, "^$1$2");
  return v.toLocaleString("de-DE", { maximumSignificantDigits: digits });
}

export function Spinner({ label }: { label: string }) {
  return (
    <span className="spinner" role="status">
      <span className="spinner-dot" aria-hidden="true" /> {label}
    </span>
  );
}

const ICON_PATHS: Record<string, string> = {
  error: "M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13Zm-.75 3h1.5v4.5h-1.5Zm0 6h1.5V12h-1.5Z",
  warning: "M8 1.2 15.2 14H.8Zm-.75 4.3v4.2h1.5V5.5Zm0 5.4v1.5h1.5v-1.5Z",
  info: "M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13Zm-.75 5.5h1.5v4.5h-1.5Zm0-2.5h1.5V6h-1.5Z",
  success: "M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13Zm-1 9.6L4 8.1l1.06-1.06L7 8.98l3.94-3.94L12 6.1Z",
};

export function Icon({ kind }: { kind: keyof typeof ICON_PATHS | string }) {
  return (
    <svg className={`icon icon-${kind}`} width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
      <path d={ICON_PATHS[kind]} fill="currentColor" fillRule="evenodd" />
    </svg>
  );
}

export function Alert({ kind, children, onClose }: { kind: "error" | "warning" | "info" | "success"; children: ReactNode; onClose?: () => void }) {
  const icon = <Icon kind={kind} />;
  const label = { error: "Fehler", warning: "Warnung", info: "Hinweis", success: "OK" }[kind];
  return (
    <div className={`alert alert-${kind}`} role={kind === "error" ? "alert" : "status"}>
      {icon}
      <span className="sr-only">{label}: </span>
      <div className="alert-body">{children}</div>
      {onClose && (
        <button className="ghost small" onClick={onClose} aria-label="Schließen">
          ✕
        </button>
      )}
    </div>
  );
}

export function useElapsed(running: boolean): number {
  const [t, setT] = useState(0);
  useEffect(() => {
    if (!running) return;
    const start = performance.now();
    setT(0);
    const id = setInterval(() => setT((performance.now() - start) / 1000), 200);
    return () => clearInterval(id);
  }, [running]);
  return t;
}

/** Format a duration in seconds with an appropriate unit. */
export function fmtTime(s: number | null | undefined): string {
  if (s === null || s === undefined || !Number.isFinite(s)) return "–";
  const a = Math.abs(s);
  if (a === 0) return "0 s";
  if (a < 1e-6) return `${fmtNum(s * 1e9)} ns`;
  if (a < 1e-3) return `${fmtNum(s * 1e6)} µs`;
  if (a < 1) return `${fmtNum(s * 1e3)} ms`;
  if (a < 600) return `${fmtNum(s)} s`;
  if (a < 7200) return `${fmtNum(s / 60)} min`;
  if (a < 3 * 86400) return `${fmtNum(s / 3600)} h`;
  return `${fmtNum(s / 86400)} d`;
}
