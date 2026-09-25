import type { ReactNode } from "react";
import { Navigate } from "react-router-dom";
import { useDataset, useSummary } from "../lib/queries";

export function PageLoading() {
  return (
    <p className="text-ink-muted" role="status">
      Loading…
    </p>
  );
}

export function RequireDataset({ children }: { children: ReactNode }) {
  const dataset = useDataset();
  if (dataset.isPending) return <PageLoading />;
  if (dataset.isError) return <Navigate to="/" replace state={{ notice: "no-data" }} />;
  return <>{children}</>;
}

export function RequireModel({ children }: { children: ReactNode }) {
  const dataset = useDataset();
  const summary = useSummary(dataset.isSuccess);
  if (dataset.isPending || (dataset.isSuccess && summary.isPending)) return <PageLoading />;
  if (dataset.isError) return <Navigate to="/" replace state={{ notice: "no-data" }} />;
  if (summary.isError) return <Navigate to="/outcome" replace />;
  return <>{children}</>;
}
