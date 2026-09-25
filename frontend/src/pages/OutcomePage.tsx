import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ErrorBanner } from "../components/ErrorBanner";
import { api } from "../lib/api";
import { count, methodLabel, pct } from "../lib/format";
import { keys, resetResults, useDataset } from "../lib/queries";
import type { Method, TrainSummary, ValueCount } from "../lib/types";

const METHODS: Method[] = ["random_forest", "logistic_regression"];
const POSITIVE_WORDS = ["1", "yes", "true", "y"];

/** Likely "event" value: 1/yes/true if present, otherwise the rarest value. */
function defaultPositive(values: ValueCount[]): string {
  const named = values.find((v) => POSITIVE_WORDS.includes(v.value.toLowerCase()));
  return named?.value ?? values[values.length - 1]?.value ?? "";
}

export function OutcomePage() {
  const profile = useDataset().data;
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const previous = queryClient.getQueryData<TrainSummary>(keys.summary);

  const [target, setTarget] = useState(previous?.target ?? profile?.target_suggestions[0] ?? "");
  const [positive, setPositive] = useState(previous?.positive_class ?? "");
  const [method, setMethod] = useState<Method>(previous?.method ?? "random_forest");
  const [excluded, setExcluded] = useState<Set<string>>(new Set());

  const train = useMutation({
    mutationFn: api.train,
    onSuccess: (summary) => {
      resetResults(queryClient);
      queryClient.setQueryData(keys.summary, summary);
      navigate("/results");
    },
  });

  if (!profile) return null;

  const byName = Object.fromEntries(profile.columns.map((c) => [c.name, c]));
  const suggestions = profile.target_suggestions;
  const others = profile.columns.filter((c) => c.kind !== "identifier" && !suggestions.includes(c.name));
  const values = byName[target]?.top_values ?? [];
  const positiveValue = values.some((v) => v.value === positive) ? positive : defaultPositive(values);
  const features = profile.columns.filter((c) => c.kind !== "identifier" && c.name !== target);

  const chooseTarget = (name: string) => {
    setTarget(name);
    setPositive("");
  };
  const toggleFeature = (name: string) =>
    setExcluded((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });

  return (
    <div className="space-y-10">
      <header className="max-w-3xl">
        <p className="eyebrow">Step 2 · Outcome</p>
        <h1 className="mt-3 text-4xl">What do you want to predict?</h1>
        <p className="mt-3 text-ink-muted">
          Pick the column that records what happened. Suggested columns have 2–20 distinct values.
        </p>
      </header>

      <fieldset>
        <legend className="sr-only">Outcome column</legend>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {suggestions.map((name) => {
            const column = byName[name];
            const selected = target === name;
            return (
              <label
                key={name}
                className={`focus-ring cursor-pointer rounded border p-4 ${
                  selected ? "border-ink bg-paper-raised" : "border-rule hover:bg-paper-raised"
                }`}
              >
                <input
                  type="radio"
                  name="target"
                  className="sr-only"
                  checked={selected}
                  onChange={() => chooseTarget(name)}
                />
                <span className="block font-serif text-lg">{name}</span>
                <span className="mt-1 block text-sm text-ink-muted">
                  {column.unique} values · {column.top_values.slice(0, 3).map((v) => v.value).join(", ")}
                </span>
              </label>
            );
          })}
        </div>
        {others.length > 0 && (
          <label className="mt-4 flex flex-wrap items-center gap-3 text-sm">
            <span className="text-ink-muted">Or another column</span>
            <select
              className="field"
              value={suggestions.includes(target) ? "" : target}
              onChange={(e) => chooseTarget(e.target.value)}
            >
              <option value="">Choose…</option>
              {others.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name}
                </option>
              ))}
            </select>
          </label>
        )}
      </fieldset>

      {target && (
        <section className="border-t border-rule pt-8">
          <h2 className="text-2xl">Which outcome do you want to catch?</h2>
          {values.length === 0 ? (
            <p className="mt-2 text-ink-muted">
              This column has more than 20 distinct values, so Verdict can't treat it as an outcome. Pick another.
            </p>
          ) : (
            <fieldset className="mt-4 flex flex-wrap gap-3">
              <legend className="sr-only">Outcome to catch</legend>
              {values.map((v) => {
                const selected = positiveValue === v.value;
                return (
                  <label
                    key={v.value}
                    className={`focus-ring cursor-pointer rounded border px-4 py-2 ${
                      selected ? "border-verdict bg-paper-raised" : "border-rule hover:bg-paper-raised"
                    }`}
                  >
                    <input
                      type="radio"
                      name="positive"
                      className="sr-only"
                      checked={selected}
                      onChange={() => setPositive(v.value)}
                    />
                    <span className="font-medium">
                      {target} = {v.value}
                    </span>{" "}
                    <span className="text-sm text-ink-muted">
                      · {count(v.count)} rows ({pct(v.count / profile.rows)})
                    </span>
                  </label>
                );
              })}
            </fieldset>
          )}
        </section>
      )}

      <details className="border-t border-rule pt-6">
        <summary className="cursor-pointer font-medium">Advanced: model and features</summary>
        <div className="mt-4 space-y-5">
          <fieldset>
            <legend className="text-sm text-ink-muted">Model</legend>
            <div className="mt-2 flex flex-wrap gap-2">
              {METHODS.map((m) => (
                <button
                  key={m}
                  type="button"
                  aria-pressed={method === m}
                  onClick={() => setMethod(m)}
                  className={method === m ? "btn-primary" : "btn-secondary"}
                >
                  {methodLabel(m)}
                </button>
              ))}
            </div>
          </fieldset>
          <fieldset>
            <legend className="text-sm text-ink-muted">
              Features used ({features.filter((c) => !excluded.has(c.name)).length} of {features.length}) — click to
              exclude
            </legend>
            <div className="mt-2 flex flex-wrap gap-2">
              {features.map((c) => {
                const on = !excluded.has(c.name);
                return (
                  <button
                    key={c.name}
                    type="button"
                    aria-pressed={on}
                    onClick={() => toggleFeature(c.name)}
                    className={`rounded border px-2 py-1 text-sm ${
                      on ? "border-rule bg-paper-raised" : "border-dashed border-rule text-ink-faint line-through"
                    }`}
                  >
                    {c.name}
                  </button>
                );
              })}
            </div>
          </fieldset>
        </div>
      </details>

      {train.error && <ErrorBanner message={train.error.message} />}

      <div className="flex flex-wrap items-center gap-4">
        <button
          className="btn-primary px-6 py-3 text-base"
          disabled={!target || !positiveValue || train.isPending}
          onClick={() =>
            train.mutate({
              target,
              positive_class: positiveValue,
              method,
              excluded: [...excluded].filter((name) => name !== target),
            })
          }
        >
          {train.isPending ? "Reaching a verdict…" : "Reach a verdict"}
        </button>
        {train.isPending && (
          <p className="text-sm text-ink-muted" role="status">
            Scoring every row with 5-fold cross-validation…
          </p>
        )}
      </div>
    </div>
  );
}
