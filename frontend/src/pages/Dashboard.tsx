import { useQuery } from "@tanstack/react-query";
import { apiClient, ApiError } from "../lib/api";
import { HeroStat } from "../components/HeroStat";
import { StatTile } from "../components/StatTile";
import { EmptyState } from "../components/EmptyState";
import { Link } from "react-router-dom";

export function Dashboard() {
  const datasetQuery = useQuery({
    queryKey: ["dataset"],
    queryFn: apiClient.getCurrentDataset,
    retry: false,
  });
  const isNotFound = datasetQuery.error instanceof ApiError && datasetQuery.error.status === 404;
  const auditQuery = useQuery({
    queryKey: ["audit-logs"],
    queryFn: apiClient.getAuditLogs,
    enabled: !datasetQuery.isError,
  });

  if (datasetQuery.isError && isNotFound) {
    return (
      <EmptyState
        title="No dataset loaded"
        description="Load a dataset from the Data Explorer page to get started."
        action={
          <Link to="/data" className="text-xs font-medium text-accent">
            Go to Data Explorer &rarr;
          </Link>
        }
      />
    );
  }

  if (datasetQuery.isError) {
    return (
      <div className="bg-red-50 border border-red-100 text-red-700 text-xs rounded-lg px-4 py-3">
        {(datasetQuery.error as Error).message || "Failed to load dataset."}
      </div>
    );
  }

  const dataset = datasetQuery.data;
  const predictionsCount = auditQuery.data?.length ?? 0;

  return (
    <div>
      <div className="mb-6">
        <div className="text-[11px] uppercase tracking-wide text-stone-500 mb-1">Verdict</div>
        <div className="text-2xl font-semibold tracking-tight text-ink">Overview</div>
      </div>

      <div className="grid grid-cols-3 gap-3.5 mb-4">
        <HeroStat label="Dataset rows" value={dataset ? dataset.rows.toLocaleString() : "—"} />
        <StatTile label="Columns" value={dataset ? String(dataset.columns) : "—"} />
        <StatTile label="Predictions logged" value={String(predictionsCount)} />
      </div>

      {dataset && dataset.warnings.length > 0 && (
        <div className="bg-white rounded-2xl p-4 border border-black/5 shadow-tile text-xs text-stone-600">
          {dataset.warnings.map((w) => (
            <div key={w}>{w}</div>
          ))}
        </div>
      )}
    </div>
  );
}
