export interface DatasetSummary {
  rows: number;
  columns: number;
  numeric_columns: string[];
  categorical_columns: string[];
  missing_pct: number;
  warnings: string[];
}

export interface TrainRequest {
  target: string;
  features: string[] | null;
  method: string;
}

export interface TrainResponse {
  model_name: string;
  metrics: Record<string, number>;
  feature_importance: Record<string, number>;
}

export interface SampleRowResponse {
  features: Record<string, unknown>;
}

export interface CategoriesResponse {
  categories: Record<string, string[]>;
}

export interface PredictRequest {
  features: Record<string, number | string>;
}

export interface PredictResponse {
  prediction: number;
  probability: number;
  confidence: number;
}

export interface WhatIfRequest {
  baseline_features: Record<string, number | string>;
  scenario_features: Record<string, number | string>;
}

export interface WhatIfResponse {
  baseline: PredictResponse;
  scenario: PredictResponse;
  delta_probability: number;
}

export interface AuditRecord {
  timestamp: string;
  prediction: number;
  probability: number;
  confidence: number;
  confidence_level: string;
  model: string;
  threshold: number;
  recommended_action: string | null;
  record_id: number;
  features?: Record<string, unknown>;
}
