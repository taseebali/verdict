export type ColumnKind = "numeric" | "categorical" | "identifier";

export interface ValueCount {
  value: string;
  count: number;
}

export interface ColumnProfile {
  name: string;
  kind: ColumnKind;
  missing_pct: number;
  unique: number;
  top_values: ValueCount[];
}

export interface DatasetProfile {
  name: string;
  rows: number;
  columns: ColumnProfile[];
  preview: Record<string, unknown>[];
  target_suggestions: string[];
}

export type Method = "random_forest" | "logistic_regression";

export interface TrainRequest {
  target: string;
  positive_class: string;
  method: Method;
  excluded: string[];
}

export interface Driver {
  feature: string;
  segment: string;
  rate: number;
  overall: number;
  share: number;
  lift: number;
}

export interface TrainSummary {
  dataset_name: string;
  target: string;
  positive_class: string;
  method: Method;
  rows_scored: number;
  rows_skipped: number;
  base_rate: number;
  roc_auc: number;
  roc_points: [number, number][];
  features: string[];
  identifiers: string[];
  importance: { feature: string; score: number }[];
  drivers: Driver[];
}

export interface Costs {
  action_cost: number;
  saved_value: number;
  success_rate: number;
}

export interface CurvePoint {
  threshold: number;
  flagged: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number;
  recall: number;
  net: number;
}

export interface DecisionResponse {
  curve: CurvePoint[];
  recommended: CurvePoint;
}

export type Source = "training" | "new";

export interface Reason {
  feature: string;
  value: string;
  impact: number;
}

export interface ScoredRow {
  row_id: number;
  label: string;
  probability: number;
  actual: boolean | null;
  reasons: Reason[];
}

export interface RowsResponse {
  total: number;
  source: Source;
  rows: ScoredRow[];
}

export interface WhatIfResponse {
  baseline: number;
  scenario: number;
  delta: number;
  features: Record<string, string | number | boolean | null>;
}

export interface ScoreResponse {
  rows_scored: number;
  source: "new";
  name: string;
}

export interface NewDecisionResponse {
  flagged: number;
  expected_net: number;
}
