import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api } from "../../lib/api";
import { pct, signedPts } from "../../lib/format";
import { useDataset } from "../../lib/queries";
import type { ScoredRow } from "../../lib/types";
import { useDebounced } from "../../lib/useDebounced";
import { ErrorBanner } from "../ErrorBanner";

function Figure({ label, value, tone = "text-ink" }: { label: string; value: string; tone?: string }) {
  return (
    <div>
      <p className="text-sm text-ink-muted">{label}</p>
      <p className={`mt-1 font-serif text-2xl ${tone}`}>{value}</p>
    </div>
  );
}

export function WhatIfDrawer({ row, onClose }: { row: ScoredRow; onClose: () => void }) {
  const profile = useDataset().data;
  const [changes, setChanges] = useState<Record<string, string>>({});
  const debounced = useDebounced(changes, 300);
  const result = useQuery({
    queryKey: ["whatif", row.row_id, debounced],
    queryFn: () => api.whatIf(row.row_id, debounced),
    placeholderData: keepPreviousData,
  });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const columns = Object.fromEntries((profile?.columns ?? []).map((c) => [c.name, c]));
  const base = result.data?.features ?? {};
  const set = (name: string, value: string) => setChanges((prev) => ({ ...prev, [name]: value }));
  const reset = (name: string) =>
    setChanges((prev) => {
      const next = { ...prev };
      delete next[name];
      return next;
    });
  const delta = result.data?.delta ?? 0;

  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-ink/30" onClick={onClose}>
      <aside
        role="dialog"
        aria-modal="true"
        aria-labelledby="whatif-title"
        className="h-full w-full overflow-y-auto border-l border-rule bg-paper p-6 sm:max-w-md"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="eyebrow">What if</p>
            <h2 id="whatif-title" className="mt-2 text-2xl">
              {row.label}
            </h2>
          </div>
          <button className="btn-secondary px-3 py-1" onClick={onClose} autoFocus>
            Close
          </button>
        </div>

        {result.data && (
          <div className="mt-6 grid grid-cols-3 gap-3 border-y border-rule py-4 text-center" aria-live="polite">
            <Figure label="Now" value={pct(result.data.baseline)} />
            <Figure label="With changes" value={pct(result.data.scenario)} />
            <Figure
              label="Change"
              value={signedPts(delta)}
              tone={delta < 0 ? "text-gain" : delta > 0 ? "text-verdict" : "text-ink"}
            />
          </div>
        )}
        <p className="mt-2 text-xs text-ink-muted">
          Scored by the final model trained on all rows, so "Now" can differ slightly from the list's out-of-fold
          score.
        </p>
        {result.error && (
          <div className="mt-4">
            <ErrorBanner message={result.error.message} />
          </div>
        )}

        <form className="mt-6 space-y-4" onSubmit={(e) => e.preventDefault()}>
          {Object.keys(base).map((name) => {
            const column = columns[name];
            const original = base[name];
            const value = changes[name] ?? (original === null ? "" : String(original));
            const options = column?.kind === "categorical" ? column.top_values.map((v) => v.value) : [];
            const inputId = `whatif-${name}`;
            return (
              <div key={name}>
                <div className="flex items-baseline justify-between text-sm">
                  <label htmlFor={inputId}>{name}</label>
                  {changes[name] !== undefined && (
                    <button type="button" className="text-ink-muted underline" onClick={() => reset(name)}>
                      reset
                    </button>
                  )}
                </div>
                {options.length > 0 ? (
                  <select id={inputId} className="field mt-1 w-full" value={value} onChange={(e) => set(name, e.target.value)}>
                    {!options.includes(value) && <option value={value}>{value || "(missing)"}</option>}
                    {options.map((o) => (
                      <option key={o} value={o}>
                        {o}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    id={inputId}
                    className="field mt-1 w-full"
                    inputMode={column?.kind === "numeric" ? "decimal" : undefined}
                    value={value}
                    onChange={(e) => set(name, e.target.value)}
                  />
                )}
              </div>
            );
          })}
        </form>
      </aside>
    </div>
  );
}
