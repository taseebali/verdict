import { count, formatCell } from "../../lib/format";
import type { ColumnKind, DatasetProfile } from "../../lib/types";

const KIND_LABEL: Record<ColumnKind, string> = {
  numeric: "number",
  categorical: "category",
  identifier: "ID — not used",
};

export function DataPreview({ profile, onContinue }: { profile: DatasetProfile; onContinue: () => void }) {
  const { columns } = profile;
  return (
    <section className="space-y-6 border-t border-rule pt-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="eyebrow">Loaded</p>
          <h2 className="mt-2 text-2xl">{profile.name}</h2>
          <p className="text-ink-muted">
            {count(profile.rows)} rows · {columns.length} columns
          </p>
        </div>
        <button className="btn-primary" onClick={onContinue}>
          Continue →
        </button>
      </div>

      <ul className="flex flex-wrap gap-2" aria-label="Columns">
        {columns.map((c) => (
          <li
            key={c.name}
            className={`rounded border px-2 py-1 text-sm ${
              c.kind === "identifier" ? "border-dashed border-rule text-ink-faint" : "border-rule bg-paper-raised"
            }`}
          >
            {c.name}{" "}
            <span className="text-ink-faint">
              · {KIND_LABEL[c.kind]}
              {c.missing_pct > 0 ? ` · ${c.missing_pct}% missing` : ""}
            </span>
          </li>
        ))}
      </ul>

      <div className="overflow-x-auto rounded border border-rule bg-paper-raised">
        <table className="min-w-full text-sm">
          <caption className="sr-only">First {profile.preview.length} rows</caption>
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c.name} scope="col" className="whitespace-nowrap border-b border-rule px-3 py-2 text-left font-semibold">
                  {c.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {profile.preview.map((row, i) => (
              <tr key={i} className="border-b border-rule last:border-0">
                {columns.map((c) => (
                  <td key={c.name} className="whitespace-nowrap px-3 py-1.5 text-ink-muted">
                    {formatCell(row[c.name])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
