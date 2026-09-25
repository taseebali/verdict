import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, XAxis, YAxis } from "recharts";
import { count, methodLabel, pct } from "../../lib/format";
import type { CurvePoint, TrainSummary } from "../../lib/types";

function Stat({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div>
      <dt className="text-sm text-ink-muted">{label}</dt>
      <dd className="mt-1 font-serif text-3xl">{value}</dd>
      <dd className="text-sm text-ink-faint">{note}</dd>
    </div>
  );
}

function ConfusionMatrix({ point, outcome }: { point: CurvePoint; outcome: string }) {
  const rows: [string, number, number][] = [
    ["Flagged", point.tp, point.fp],
    ["Not flagged", point.fn, point.tn],
  ];
  return (
    <table className="w-full text-sm">
      <caption className="pb-2 text-left text-sm text-ink-muted">Confusion matrix at the current cutoff</caption>
      <thead>
        <tr className="text-left">
          <td />
          <th scope="col" className="px-3 py-2 font-medium">
            Actually {outcome}
          </th>
          <th scope="col" className="px-3 py-2 font-medium">
            Actually not
          </th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([label, a, b]) => (
          <tr key={label} className="border-t border-rule">
            <th scope="row" className="px-3 py-2 text-left font-medium">
              {label}
            </th>
            <td className="px-3 py-2">{count(a)}</td>
            <td className="px-3 py-2">{count(b)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function RocChart({ points }: { points: [number, number][] }) {
  const data = points.map(([fpr, tpr]) => ({ fpr, tpr }));
  return (
    <figure>
      <figcaption className="text-sm text-ink-muted">ROC curve (out-of-fold) · dashed line = coin flip</figcaption>
      <div className="mt-2 h-56">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
            <CartesianGrid stroke="#e3dccf" />
            <XAxis dataKey="fpr" type="number" domain={[0, 1]} tickFormatter={(v) => pct(Number(v))} stroke="#736b5f" fontSize={12} />
            <YAxis dataKey="tpr" type="number" domain={[0, 1]} tickFormatter={(v) => pct(Number(v))} stroke="#736b5f" fontSize={12} width={44} />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#736b5f" strokeDasharray="4 4" />
            <Line type="monotone" dataKey="tpr" stroke="#a4262c" dot={false} strokeWidth={1.5} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

export function UnderTheHood({ summary, point }: { summary: TrainSummary; point: CurvePoint }) {
  const outcome = `${summary.target} = ${summary.positive_class}`;
  return (
    <div className="grid gap-10 lg:grid-cols-2">
      <section className="space-y-8">
        <dl className="grid grid-cols-2 gap-6">
          <Stat label="ROC AUC (out-of-fold)" value={summary.roc_auc.toFixed(3)} note="0.5 is a coin flip, 1.0 is perfect" />
          <Stat label="Base rate" value={pct(summary.base_rate, 1)} note={`of rows have ${outcome}`} />
          <Stat
            label={`Precision at ${Math.round(point.threshold * 100)}%`}
            value={pct(point.precision, 1)}
            note="of flagged rows are real cases"
          />
          <Stat label="Recall" value={pct(point.recall, 1)} note="of real cases get flagged" />
        </dl>
        <ConfusionMatrix point={point} outcome={outcome} />
      </section>
      <section className="space-y-6">
        <RocChart points={summary.roc_points} />
        <div className="space-y-3 text-sm text-ink-muted">
          <p>
            <span className="font-medium text-ink">Model:</span> {methodLabel(summary.method)}, predicting{" "}
            <span className="text-ink">{outcome}</span> vs everything else.
          </p>
          <p>
            <span className="font-medium text-ink">Honest scores:</span> 5-fold stratified out-of-fold predictions — no
            row is scored by a model that saw it.
          </p>
          <p>
            <span className="font-medium text-ink">Features ({summary.features.length}):</span>{" "}
            {summary.features.join(", ")}
          </p>
          {summary.identifiers.length > 0 && (
            <p>
              <span className="font-medium text-ink">Excluded as identifiers:</span> {summary.identifiers.join(", ")}
            </p>
          )}
          {summary.rows_skipped > 0 && (
            <p>
              <span className="font-medium text-ink">Skipped:</span> {count(summary.rows_skipped)} rows with a blank
              outcome.
            </p>
          )}
        </div>
        <div className="rounded border border-rule bg-paper-raised p-4 text-sm">
          <p className="font-medium">How the recommendation is computed</p>
          <p className="mt-2 font-mono text-xs">
            net = true cases flagged × success rate × value of a save − rows flagged × cost per action
          </p>
          <p className="mt-2 text-ink-muted">
            Verdict tries every cutoff from 0% to 100% on the out-of-fold scores and recommends the one with the
            highest net. If no cutoff pays, it recommends acting on nobody. The recommended cutoff is chosen on the
            same scores used to report its net, so treat the net as slightly optimistic.
          </p>
        </div>
      </section>
    </div>
  );
}
