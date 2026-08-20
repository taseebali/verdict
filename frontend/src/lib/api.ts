import type {
  DatasetSummary,
  TrainRequest,
  TrainResponse,
  PredictRequest,
  PredictResponse,
  WhatIfRequest,
  WhatIfResponse,
  AuditRecord,
  SampleRowResponse,
  CategoriesResponse,
} from "./types";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new ApiError(body.detail ?? `Request failed: ${response.status}`, response.status);
  }
  return response.json();
}

export const apiClient = {
  loadDemo: () => request<DatasetSummary>("/api/datasets/demo", { method: "POST" }),

  uploadCsv: async (file: File): Promise<DatasetSummary> => {
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch(`${BASE_URL}/api/datasets/upload`, { method: "POST", body: formData });
    if (!response.ok) {
      const body = await response.json().catch(() => ({ detail: response.statusText }));
      throw new ApiError(body.detail ?? `Upload failed: ${response.status}`, response.status);
    }
    return response.json();
  },

  getCurrentDataset: () => request<DatasetSummary>("/api/datasets/current"),

  train: (req: TrainRequest) =>
    request<TrainResponse>("/api/train", { method: "POST", body: JSON.stringify(req) }),

  predict: (req: PredictRequest) =>
    request<PredictResponse>("/api/predict", { method: "POST", body: JSON.stringify(req) }),

  whatif: (req: WhatIfRequest) =>
    request<WhatIfResponse>("/api/whatif", { method: "POST", body: JSON.stringify(req) }),

  getAuditLogs: () => request<AuditRecord[]>("/api/audit-logs"),

  getSampleRow: () => request<SampleRowResponse>("/api/datasets/sample-row"),

  getCategories: () => request<CategoriesResponse>("/api/datasets/categories"),
};
