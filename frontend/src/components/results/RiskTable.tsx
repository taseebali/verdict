import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../../lib/api";
import { pct } from "../../lib/format";
import type { ScoredRow, Source } from "../../lib/types";
import { ErrorBanner } from "../ErrorBanner";

const PAGE_SIZE = 25;

interface Props {
  source: Source;
  threshold: number;
  onSelect?: (row: ScoredRow) => void;
}

export function RiskTable({ source, threshold, onSelect }: Props) {
  const [page, setPage] = useState(0);
  const rows = useQuery({
    queryKey: ["rows", source, page],
    queryFn: () => api.rows(source, page * PAGE_SIZE, PAGE_SIZE),
    placeholderData: keepPreviousData,
  });

  if (rows.isError) return <ErrorBanner message={rows.error.message} />;
  if (!rows.data) return <p className="text-ink-muted">Loading rows…</p>;

  const pages = Math.max(1, Math.ceil(rows.data.total / PAGE_SIZE));
  const showActual = source === "training";

  return (
    <div className="space-y-4">
      <p className="text-sm text-ink-muted">
        Sorted by risk. Red means above your cutoff.
        {onSelect && " Select a row to test what would change its score."}
      </p>
      <table className="w-full text-sm">
        <thead className="hidden md:table-header-group">
          <tr className="border-b border-rule text-left text-ink-muted">
            <th scope="col" className="py-2 pr-3 font-medium">#</th>
            <th scope="col" className="py-2 pr-3 font-medium">Record</th>
            <th scope="col" className="py-2 pr-3 font-medium">Risk</th>
            <th scope="col" className="py-2 pr-3 font-medium">Why</th>
            {showActual && <th scope="col" className="py-2 font-medium">Actual</th>}
          </tr>
        </thead>
        <tbody>
          {rows.data.rows.map((row, i) => {
            const flagged = row.probability > threshold;
            const interactive = Boolean(onSelect);
            return (
              <tr
                key={row.row_id}
                tabIndex={interactive ? 0 : undefined}
                aria-label={interactive ? `Explore ${row.label}` : undefined}
                onClick={() => onSelect?.(row)}
                onKeyDown={(e) => {
                  if (onSelect && (e.key === "Enter" || e.key === " ")) {
                    e.preventDefault();
                    onSelect(row);
                  }
                }}
                className={`block border-b border-rule py-3 md:table-row md:py-0 ${
                  interactive ? "cursor-pointer hover:bg-paper-raised" : ""
                }`}
              >
                <td className="hidden py-2 pr-3 text-ink-faint md:table-cell">{page * PAGE_SIZE + i + 1}</td>
                <td className="inline font-medium md:table-cell md:py-2 md:pr-3">{row.label}</td>
                <td className="inline pl-3 md:table-cell md:py-2 md:pl-0 md:pr-3">
                  <span
                    className={`inline-block min-w-[3.5rem] rounded px-2 py-0.5 text-center font-semibold ${
                      flagged ? "bg-verdict text-paper" : "border border-rule text-ink"
                    }`}
                  >
                    {pct(row.probability)}
                  </span>
                </td>
                <td className="block pt-1 text-ink-muted md:table-cell md:py-2 md:pr-3">
                  {row.reasons.length ? row.reasons.map((r) => `${r.feature} = ${r.value}`).join(" · ") : "—"}
                </td>
                {showActual && (
                  <td className="hidden py-2 text-ink-muted md:table-cell">{row.actual ? "✓ yes" : "–"}</td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="flex items-center justify-between gap-3 text-sm">
        <button className="btn-secondary" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
          ← Previous
        </button>
        <span className="text-ink-muted">
          Page {page + 1} of {pages}
        </span>
        <button className="btn-secondary" disabled={page + 1 >= pages} onClick={() => setPage((p) => p + 1)}>
          Next →
        </button>
      </div>
    </div>
  );
}
