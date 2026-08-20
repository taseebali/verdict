import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { apiClient } from "../lib/api";
import { StatTile } from "../components/StatTile";

export function DataExplorer() {
  const queryClient = useQueryClient();
  const [error, setError] = useState<string | null>(null);

  const demoMutation = useMutation({
    mutationFn: apiClient.loadDemo,
    onSuccess: () => {
      setError(null);
      queryClient.invalidateQueries({ queryKey: ["dataset"] });
    },
    onError: (e: Error) => setError(e.message),
  });

  const uploadMutation = useMutation({
    mutationFn: apiClient.uploadCsv,
    onSuccess: () => {
      setError(null);
      queryClient.invalidateQueries({ queryKey: ["dataset"] });
    },
    onError: (e: Error) => setError(e.message),
  });

  const summary = demoMutation.data ?? uploadMutation.data;

  return (
    <div>
      <div className="mb-6">
        <div className="text-[11px] uppercase tracking-wide text-stone-500 mb-1">Data</div>
        <div className="text-2xl font-semibold tracking-tight text-ink">Data Explorer</div>
      </div>

      <div className="flex gap-3 mb-6">
        <button
          onClick={() => demoMutation.mutate()}
          disabled={demoMutation.isPending}
          className="bg-ink text-canvas-100 rounded-lg px-4 py-2 text-xs font-medium active:scale-[0.98] transition-transform disabled:opacity-50"
        >
          {demoMutation.isPending ? "Loading…" : "Use demo dataset"}
        </button>
        <label className="border border-black/10 bg-white rounded-lg px-4 py-2 text-xs font-medium cursor-pointer hover:bg-stone-50 transition-colors">
          Upload CSV
          <input
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) uploadMutation.mutate(file);
            }}
          />
        </label>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-100 text-red-700 text-xs rounded-lg px-4 py-3 mb-6">
          {error}
        </div>
      )}

      {summary && (
        <div className="grid grid-cols-4 gap-3.5 mb-6">
          <StatTile label="Rows" value={summary.rows.toLocaleString()} />
          <StatTile label="Columns" value={String(summary.columns)} />
          <StatTile label="Numeric" value={String(summary.numeric_columns.length)} />
          <StatTile label="Categorical" value={String(summary.categorical_columns.length)} />
        </div>
      )}

      {summary && summary.warnings.length > 0 && (
        <div className="bg-white rounded-2xl p-4 border border-black/5 shadow-tile text-xs text-stone-600">
          {summary.warnings.map((w) => (
            <div key={w}>{w}</div>
          ))}
        </div>
      )}
    </div>
  );
}
