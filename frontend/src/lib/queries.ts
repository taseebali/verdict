import { useQuery, type QueryClient } from "@tanstack/react-query";
import { api } from "./api";

export const keys = {
  dataset: ["dataset"] as const,
  summary: ["summary"] as const,
};

export function useDataset() {
  return useQuery({ queryKey: keys.dataset, queryFn: api.currentDataset, retry: false });
}

export function useSummary(enabled = true) {
  return useQuery({ queryKey: keys.summary, queryFn: api.summary, retry: false, enabled });
}

/** Forget every cached result derived from the previous model. */
export function resetResults(queryClient: QueryClient) {
  for (const key of ["summary", "decision", "rows", "newDecision", "whatif"]) {
    queryClient.removeQueries({ queryKey: [key] });
  }
}
