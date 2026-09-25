import type { Method } from "./types";

export const pct = (value: number, digits = 0) => `${(value * 100).toFixed(digits)}%`;

export const count = (value: number) => value.toLocaleString("en-US");

export const money = (value: number) => {
  const sign = value < 0 ? "−" : "+";
  return `${sign}$${Math.round(Math.abs(value)).toLocaleString("en-US")}`;
};

export const plainMoney = (value: number) => `$${value.toLocaleString("en-US")}`;

export const compactMoney = (value: number) => {
  const sign = value < 0 ? "−" : "";
  const abs = Math.abs(value);
  return abs >= 1000 ? `${sign}$${Math.round(abs / 1000)}k` : `${sign}$${Math.round(abs)}`;
};

export const signedPts = (delta: number) =>
  `${delta >= 0 ? "+" : "−"}${Math.abs(delta * 100).toFixed(1)} pts`;

export const methodLabel = (method: Method) =>
  method === "random_forest" ? "Random Forest" : "Logistic Regression";

export const formatCell = (value: unknown) => {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toLocaleString("en-US") : value.toFixed(2);
  }
  return String(value);
};
