import { useQuery } from "@tanstack/react-query";
import { apiClient } from "../lib/api";
import { EmptyState } from "../components/EmptyState";
import { SkeletonBlock } from "../components/SkeletonBlock";

export function AuditLogs() {
  const auditQuery = useQuery({ queryKey: ["audit-logs"], queryFn: apiClient.getAuditLogs });

  return (
    <div>
      <div className="mb-6">
        <div className="text-[11px] uppercase tracking-wide text-stone-500 mb-1">Audit</div>
        <div className="text-2xl font-semibold tracking-tight text-ink">Prediction log</div>
      </div>

      {auditQuery.isLoading && (
        <div className="flex flex-col gap-2">
          <SkeletonBlock className="h-9 w-full" />
          <SkeletonBlock className="h-9 w-full" />
          <SkeletonBlock className="h-9 w-full" />
        </div>
      )}

      {auditQuery.isError && (
        <div className="bg-red-50 border border-red-100 text-red-700 text-xs rounded-lg px-4 py-3">
          {(auditQuery.error as Error).message || "Failed to load audit logs."}
        </div>
      )}

      {auditQuery.data && auditQuery.data.length === 0 && (
        <EmptyState title="No predictions yet" description="Predictions you make will be logged here." />
      )}

      {auditQuery.data && auditQuery.data.length > 0 && (
        <div className="bg-white rounded-2xl border border-black/5 shadow-tile overflow-hidden">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-black/5 text-stone-500 uppercase text-[10px] tracking-wide">
                <th className="text-left px-4 py-2.5 font-medium">Timestamp</th>
                <th className="text-left px-4 py-2.5 font-medium">Model</th>
                <th className="text-left px-4 py-2.5 font-medium">Prediction</th>
                <th className="text-left px-4 py-2.5 font-medium">Probability</th>
                <th className="text-left px-4 py-2.5 font-medium">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {auditQuery.data.map((record) => (
                <tr key={record.record_id} className="border-b border-black/5 last:border-0">
                  <td className="px-4 py-2.5 text-stone-500 data-value">{new Date(record.timestamp).toLocaleString()}</td>
                  <td className="px-4 py-2.5 text-ink">{record.model}</td>
                  <td className="px-4 py-2.5 text-ink data-value">{record.prediction}</td>
                  <td className="px-4 py-2.5 text-ink data-value">{(record.probability * 100).toFixed(1)}%</td>
                  <td className="px-4 py-2.5 text-ink">{record.confidence_level}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
