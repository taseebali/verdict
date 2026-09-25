import { count, money, pct, plainMoney } from "../../lib/format";
import type { Costs } from "../../lib/types";

interface Props {
  fileName: string;
  outcome: string;
  flagged: number;
  total: number;
  net: number;
  thresholdPct: number;
  costs: Costs;
  expected: boolean;
}

export function Headline({ fileName, outcome, flagged, total, net, thresholdPct, costs, expected }: Props) {
  const nobody = flagged === 0;
  return (
    <section aria-live="polite">
      <p className="eyebrow">The verdict · {fileName}</p>
      <h1 className="mt-3 text-4xl leading-tight sm:text-5xl">
        {nobody ? (
          <>Don't act on anyone.</>
        ) : (
          <>
            Act on <span className="text-verdict">{count(flagged)}</span> of {count(total)} rows.
          </>
        )}
        <br />
        {nobody && net <= 0 ? (
          <span className="text-ink-muted">At these costs, acting doesn't pay.</span>
        ) : (
          <>
            {expected ? "Expected net" : "Net"}:{" "}
            <span className={net >= 0 ? "text-gain" : "text-verdict"}>{money(net)}</span>.
          </>
        )}
      </h1>
      <p className="mt-4 text-ink-muted">
        Flagging rows above a {thresholdPct}% risk of <span className="text-ink">{outcome}</span> ·{" "}
        {plainMoney(costs.action_cost)} per action · {plainMoney(costs.saved_value)} per save ·{" "}
        {pct(costs.success_rate)} success rate
      </p>
    </section>
  );
}
