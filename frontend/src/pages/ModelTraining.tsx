import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { apiClient } from "../lib/api";
import { Chip } from "../components/Chip";
import { EmptyState } from "../components/EmptyState";
import { Link } from "react-router-dom";

export function ModelTraining() {
  const datasetQuery = useQuery({ queryKey: ["dataset"], queryFn: apiClient.getCurrentDataset, retry: false });
  const [target, setTarget] = useState<string>("");
  const [excludedFeatures, setExcludedFeatures] = useState<Set<string>>(new Set());

  const trainMutation = useMutation({ mutationFn: apiClient.train });

  if (datasetQuery.isError) {
    return (
      <EmptyState
        title="No dataset loaded"
        description="Load a dataset first."
        action={<Link to="/data" className="text-xs font-medium text-accent">Go to Data Explorer &rarr;</Link>}
      />
    );
  }

  const dataset = datasetQuery.data;
  const allColumns = dataset ? [...dataset.numeric_columns, ...dataset.categorical_columns] : [];
  const candidateFeatures = allColumns.filter((c) => c !== target);
  const selectedFeatureCount = candidateFeatures.length - excludedFeatures.size;

  return (
    <div>
      <div className="mb-6">
        <div className="text-[11px] uppercase tracking-wide text-stone-500 mb-1">Model training</div>
        <div className="text-2xl font-semibold tracking-tight text-ink">Train a classifier</div>
        {dataset && (
          <div className="text-xs text-stone-400 data-value mt-0.5">
            {dataset.rows.toLocaleString()} rows &middot; {dataset.columns} columns
          </div>
        )}
      </div>

      <div className="mb-5">
        <div className="text-[11px] text-stone-500 mb-1.5">Target column</div>
        <select
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          className="border border-black/10 rounded-md px-2.5 py-2 text-xs bg-white w-64"
        >
          <option value="">Select…</option>
          {allColumns.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </div>

      {target && (
        <div className="mb-5">
          <div className="text-[11px] text-stone-500 mb-1.5 data-value">
            Features &middot; {candidateFeatures.length - excludedFeatures.size} of {candidateFeatures.length} selected
          </div>
          <div className="flex flex-wrap gap-1.5">
            {candidateFeatures.map((f) => (
              <Chip
                key={f}
                label={f}
                selected={!excludedFeatures.has(f)}
                onClick={() =>
                  setExcludedFeatures((prev) => {
                    const next = new Set(prev);
                    if (next.has(f)) next.delete(f);
                    else next.add(f);
                    return next;
                  })
                }
              />
            ))}
          </div>
        </div>
      )}

      <button
        disabled={!target || selectedFeatureCount === 0 || trainMutation.isPending}
        onClick={() =>
          trainMutation.mutate({
            target,
            features: candidateFeatures.filter((f) => !excludedFeatures.has(f)),
            method: "random_forest",
          })
        }
        className="bg-ink text-canvas-100 rounded-lg px-4 py-2 text-xs font-medium active:scale-[0.98] transition-transform disabled:opacity-40"
      >
        {trainMutation.isPending ? "Training…" : "Train model"}
      </button>

      {trainMutation.isError && (
        <div className="bg-red-50 border border-red-100 text-red-700 text-xs rounded-lg px-4 py-3 mt-4">
          {(trainMutation.error as Error).message}
        </div>
      )}

      {trainMutation.data && (
        <div className="mt-6 pt-5 border-t border-black/5">
          <div className="flex gap-6 mb-5">
            {Object.entries(trainMutation.data.metrics).map(([key, value]) => (
              <div key={key}>
                <div className="text-[10px] text-stone-400 uppercase tracking-wide">{key}</div>
                <div className="text-sm font-semibold text-ink data-value">{(value as number).toFixed(3)}</div>
              </div>
            ))}
          </div>

          <div className="text-xs font-medium text-ink mb-2">Feature importance</div>
          <div className="flex flex-col gap-2">
            {Object.entries(trainMutation.data.feature_importance)
              .slice(0, 8)
              .map(([feature, value]) => {
                const maxVal = Math.max(...Object.values(trainMutation.data!.feature_importance).map(Math.abs));
                const width = maxVal > 0 ? (Math.abs(value as number) / maxVal) * 100 : 0;
                return (
                  <div key={feature}>
                    <div className="flex justify-between text-[11px] text-stone-600 mb-0.5">
                      <span>{feature}</span>
                      <span className="data-value">{(value as number).toFixed(4)}</span>
                    </div>
                    <div className="h-[5px] bg-stone-100 rounded">
                      <div className="h-full bg-accent rounded" style={{ width: `${width}%` }} />
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      )}
    </div>
  );
}
