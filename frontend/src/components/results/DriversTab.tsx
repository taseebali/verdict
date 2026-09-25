import { pct } from "../../lib/format";
import type { TrainSummary } from "../../lib/types";

function ImportanceBars({ items }: { items: { feature: string; score: number }[] }) {
  const max = Math.max(...items.map((i) => i.score), 1e-9);
  return (
    <ul className="mt-4 space-y-3">
      {items.map((item) => (
        <li key={item.feature}>
          <div className="flex justify-between gap-3 text-sm">
            <span>{item.feature}</span>
            <span className="text-ink-muted">{item.score.toFixed(3)}</span>
          </div>
          <div className="mt-1 h-1.5 rounded bg-rule" aria-hidden="true">
            <div className="h-full rounded bg-ink" style={{ width: `${(item.score / max) * 100}%` }} />
          </div>
        </li>
      ))}
    </ul>
  );
}

export function DriversTab({ summary }: { summary: TrainSummary }) {
  const outcome = `${summary.target} = ${summary.positive_class}`;
  const important = summary.importance.filter((i) => i.score > 0).slice(0, 10);
  return (
    <div className="grid gap-10 lg:grid-cols-[3fr_2fr]">
      <section>
        <h2 className="text-2xl">Where {outcome} concentrates</h2>
        {summary.drivers.length === 0 ? (
          <p className="mt-4 text-ink-muted">No single segment stands out strongly in this data.</p>
        ) : (
          <ol className="mt-4 space-y-4">
            {summary.drivers.map((d) => (
              <li key={d.feature} className="rounded border border-rule bg-paper-raised p-4">
                <p className="font-serif text-lg">{d.segment}</p>
                <p className="mt-1 text-ink-muted">
                  <span className="font-semibold text-verdict">{pct(d.rate)}</span> have {outcome}, vs{" "}
                  {pct(d.overall)} overall ({d.lift.toFixed(1)}×). {pct(d.share)} of rows are in this group.
                </p>
              </li>
            ))}
          </ol>
        )}
        <p className="mt-4 text-sm text-ink-faint">
          Segments are measured directly on your data. They show where the outcome concentrates, not what causes it.
        </p>
      </section>
      <section>
        <h2 className="text-2xl">What the model leans on</h2>
        <p className="mt-1 text-sm text-ink-muted">
          Drop in ROC AUC when a column is shuffled (permutation importance on the final model).
        </p>
        {important.length ? (
          <ImportanceBars items={important} />
        ) : (
          <p className="mt-4 text-ink-muted">No column moves the score on its own.</p>
        )}
      </section>
    </div>
  );
}
