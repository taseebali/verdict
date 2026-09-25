import type {
  Costs,
  DatasetProfile,
  DecisionResponse,
  NewDecisionResponse,
  RowsResponse,
  ScoreResponse,
  Source,
  TrainRequest,
  TrainSummary,
  WhatIfResponse,
} from "./types";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function messageFrom(detail: unknown, status: number): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d: { msg?: string }) => d.msg ?? "Invalid input").join("; ");
  return `Request failed (${status})`;
}

async function send<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ApiError(messageFrom(body?.detail, response.status), response.status);
  }
  return response.json() as Promise<T>;
}

const postJson = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

const postFile = (file: File): RequestInit => {
  const form = new FormData();
  form.append("file", file);
  return { method: "POST", body: form };
};

export const api = {
  loadDemo: () => send<DatasetProfile>("/api/datasets/demo", { method: "POST" }),
  uploadCsv: (file: File) => send<DatasetProfile>("/api/datasets/upload", postFile(file)),
  currentDataset: () => send<DatasetProfile>("/api/datasets/current"),
  train: (request: TrainRequest) => send<TrainSummary>("/api/train", postJson(request)),
  summary: () => send<TrainSummary>("/api/results/summary"),
  decision: (costs: Costs) => send<DecisionResponse>("/api/results/decision", postJson(costs)),
  rows: (source: Source, offset: number, limit: number) =>
    send<RowsResponse>(`/api/results/rows?source=${source}&offset=${offset}&limit=${limit}`),
  whatIf: (rowId: number, changes: Record<string, string>) =>
    send<WhatIfResponse>("/api/results/whatif", postJson({ row_id: rowId, changes })),
  scoreFile: (file: File) => send<ScoreResponse>("/api/results/score", postFile(file)),
  newDecision: (threshold: number, costs: Costs) =>
    send<NewDecisionResponse>("/api/results/new/decision", postJson({ threshold, ...costs })),
  exportUrl: (source: Source, threshold: number) =>
    `/api/results/export.csv?source=${source}&threshold=${threshold}`,
};

/** The server forgot this visitor (expired session or never loaded data). */
export const isSessionGone = (error: unknown) =>
  error instanceof ApiError && error.status === 404 && error.message.startsWith("No dataset");
