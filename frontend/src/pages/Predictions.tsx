import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { apiClient } from "../lib/api";
import { EmptyState } from "../components/EmptyState";
import { Link } from "react-router-dom";

export function Predictions() {
  const datasetQuery = useQuery({ queryKey: ["dataset"], queryFn: apiClient.getCurrentDataset, retry: false });
  const categoriesQuery = useQuery({
    queryKey: ["categories"],
    queryFn: apiClient.getCategories,
    enabled: !datasetQuery.isError,
  });
  const [values, setValues] = useState<Record<string, string>>({});

  const sampleMutation = useMutation({
    mutationFn: apiClient.getSampleRow,
    onSuccess: (data) => {
      const next: Record<string, string> = {};
      for (const [key, value] of Object.entries(data.features)) {
        next[key] = String(value);
      }
      setValues(next);
    },
  });

  const predictMutation = useMutation({ mutationFn: apiClient.predict });

  if (datasetQuery.isError) {
    return (
      <EmptyState
        title="No dataset loaded"
        description="Load and train a model first."
        action={<Link to="/data" className="text-xs font-medium text-accent">Go to Data Explorer &rarr;</Link>}
      />
    );
  }

  const dataset = datasetQuery.data;
  const featureColumns = dataset ? [...dataset.numeric_columns, ...dataset.categorical_columns] : [];
  const categories = categoriesQuery.data?.categories ?? {};

  return (
    <div>
      <div className="mb-6 flex items-start justify-between">
        <div>
          <div className="text-[11px] uppercase tracking-wide text-stone-500 mb-1">Predictions</div>
          <div className="text-2xl font-semibold tracking-tight text-ink">Run a prediction</div>
        </div>
        <button
          onClick={() => sampleMutation.mutate()}
          disabled={sampleMutation.isPending}
          className="border border-black/10 bg-white rounded-lg px-3 py-1.5 text-xs font-medium hover:bg-stone-50 active:scale-[0.98] transition-transform disabled:opacity-40"
        >
          {sampleMutation.isPending ? "Loading…" : "Load a sample row"}
        </button>
      </div>

      <div className="grid grid-cols-4 gap-3 mb-5">
        {featureColumns.map((f) => {
          const options = categories[f];
          return (
            <div key={f}>
              <label className="text-[11px] text-stone-500 mb-1 block">{f}</label>
              {options ? (
                <select
                  value={values[f] ?? ""}
                  onChange={(e) => setValues((prev) => ({ ...prev, [f]: e.target.value }))}
                  className="border border-black/10 rounded-md px-2.5 py-1.5 text-xs w-full bg-white"
                >
                  <option value="">Select…</option>
                  {options.map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  value={values[f] ?? ""}
                  onChange={(e) => setValues((prev) => ({ ...prev, [f]: e.target.value }))}
                  className="border border-black/10 rounded-md px-2.5 py-1.5 text-xs w-full bg-white"
                />
              )}
            </div>
          );
        })}
      </div>

      <button
        onClick={() => {
          const features: Record<string, number | string> = {};
          for (const [key, value] of Object.entries(values)) {
            const numeric = Number(value);
            features[key] = value !== "" && !Number.isNaN(numeric) ? numeric : value;
          }
          predictMutation.mutate({ features });
        }}
        disabled={predictMutation.isPending}
        className="bg-ink text-canvas-100 rounded-lg px-4 py-2 text-xs font-medium active:scale-[0.98] transition-transform disabled:opacity-40"
      >
        {predictMutation.isPending ? "Predicting…" : "Predict"}
      </button>

      {predictMutation.isError && (
        <div className="bg-red-50 border border-red-100 text-red-700 text-xs rounded-lg px-4 py-3 mt-4">
          {(predictMutation.error as Error).message}
        </div>
      )}

      {predictMutation.data && (
        <div className="mt-6 pt-5 border-t border-black/5 flex gap-6">
          <div>
            <div className="text-[10px] text-stone-400 uppercase tracking-wide">Prediction</div>
            <div className="text-2xl font-semibold text-ink data-value">{predictMutation.data.prediction}</div>
          </div>
          <div>
            <div className="text-[10px] text-stone-400 uppercase tracking-wide">Probability</div>
            <div className="text-2xl font-semibold text-ink data-value">
              {(predictMutation.data.probability * 100).toFixed(1)}%
            </div>
          </div>
          <div>
            <div className="text-[10px] text-stone-400 uppercase tracking-wide">Confidence</div>
            <div className="text-2xl font-semibold text-ink data-value">
              {(predictMutation.data.confidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
