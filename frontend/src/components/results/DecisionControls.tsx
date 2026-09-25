import { useId, useState } from "react";
import type { Costs } from "../../lib/types";

interface NumberFieldProps {
  label: string;
  value: number;
  min: number;
  max?: number;
  prefix?: string;
  suffix?: string;
  onChange: (value: number) => void;
}

function NumberField({ label, value, min, max, prefix, suffix, onChange }: NumberFieldProps) {
  const [text, setText] = useState(String(value));
  const id = useId();
  return (
    <div>
      <label htmlFor={id} className="text-sm text-ink-muted">
        {label}
      </label>
      <div className="focus-ring mt-1 flex items-center rounded border border-rule bg-paper-raised">
        {prefix && <span className="pl-3 text-ink-muted">{prefix}</span>}
        <input
          id={id}
          inputMode="decimal"
          value={text}
          className="w-full bg-transparent px-2 py-2 outline-none"
          onChange={(e) => {
            setText(e.target.value);
            const n = Number(e.target.value);
            const valid =
              e.target.value.trim() !== "" && Number.isFinite(n) && n >= min && (max === undefined || n <= max);
            if (valid) onChange(n);
          }}
        />
        {suffix && <span className="pr-3 text-ink-muted">{suffix}</span>}
      </div>
    </div>
  );
}

interface Props {
  costs: Costs;
  onCosts: (costs: Costs) => void;
  thresholdPct: number;
  recommendedPct: number;
  onThreshold: (pct: number) => void;
  onUseRecommended: () => void;
}

export function DecisionControls({ costs, onCosts, thresholdPct, recommendedPct, onThreshold, onUseRecommended }: Props) {
  const sliderId = useId();
  return (
    <div className="grid gap-8 lg:grid-cols-2">
      <div className="grid gap-4 sm:grid-cols-3">
        <NumberField
          label="Cost per action"
          prefix="$"
          value={costs.action_cost}
          min={0}
          onChange={(v) => onCosts({ ...costs, action_cost: v })}
        />
        <NumberField
          label="Value of one save"
          prefix="$"
          value={costs.saved_value}
          min={1}
          onChange={(v) => onCosts({ ...costs, saved_value: v })}
        />
        <NumberField
          label="Action success rate"
          suffix="%"
          value={Math.round(costs.success_rate * 100)}
          min={1}
          max={100}
          onChange={(v) => onCosts({ ...costs, success_rate: v / 100 })}
        />
      </div>
      <div>
        <label htmlFor={sliderId} className="flex items-baseline justify-between text-sm">
          <span className="text-ink-muted">Risk cutoff</span>
          <span className="font-medium">{thresholdPct}%</span>
        </label>
        <input
          id={sliderId}
          type="range"
          min={0}
          max={100}
          step={1}
          value={thresholdPct}
          onChange={(e) => onThreshold(Number(e.target.value))}
          className="mt-3 w-full accent-verdict"
        />
        <div className="mt-2 flex items-center justify-between text-sm text-ink-muted">
          <span>Recommended: {recommendedPct}%</span>
          <button
            type="button"
            className="underline hover:text-ink disabled:no-underline disabled:opacity-50"
            disabled={thresholdPct === recommendedPct}
            onClick={onUseRecommended}
          >
            Use recommended
          </button>
        </div>
      </div>
    </div>
  );
}
