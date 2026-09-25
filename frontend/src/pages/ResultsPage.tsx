import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { ErrorBanner } from "../components/ErrorBanner";
import { PageLoading } from "../components/Guards";
import { DecisionControls } from "../components/results/DecisionControls";
import { DriversTab } from "../components/results/DriversTab";
import { Headline } from "../components/results/Headline";
import { NetCurve } from "../components/results/NetCurve";
import { RiskTable } from "../components/results/RiskTable";
import { ScoreNewFile } from "../components/results/ScoreNewFile";
import { UnderTheHood } from "../components/results/UnderTheHood";
import { WhatIfDrawer } from "../components/results/WhatIfDrawer";
import { api } from "../lib/api";
import { useSummary } from "../lib/queries";
import type { Costs, ScoredRow, ScoreResponse, Source } from "../lib/types";
import { useDebounced } from "../lib/useDebounced";

const TABS = [
  { id: "list", label: "At-risk list" },
  { id: "drivers", label: "What drives it" },
  { id: "hood", label: "Under the hood" },
] as const;
type TabId = (typeof TABS)[number]["id"];

const DEFAULT_COSTS: Costs = { action_cost: 20, saved_value: 500, success_rate: 0.3 };

export function ResultsPage() {
  const summary = useSummary().data;
  const [costs, setCosts] = useState<Costs>(DEFAULT_COSTS);
  const debouncedCosts = useDebounced(costs, 400);
  const [chosenPct, setChosenPct] = useState<number | null>(null);
  const [tab, setTab] = useState<TabId>(TABS[0].id);
  const [source, setSource] = useState<Source>("training");
  const [newFile, setNewFile] = useState<ScoreResponse | null>(null);
  const [newFileVersion, setNewFileVersion] = useState(0);
  const [selected, setSelected] = useState<ScoredRow | null>(null);

  const decision = useQuery({
    queryKey: ["decision", debouncedCosts],
    queryFn: () => api.decision(debouncedCosts),
    placeholderData: keepPreviousData,
  });

  const recommendedPct = decision.data ? Math.round(decision.data.recommended.threshold * 100) : 50;
  const thresholdPct = chosenPct ?? recommendedPct;
  const threshold = thresholdPct / 100;

  const newDecision = useQuery({
    queryKey: ["newDecision", newFileVersion, threshold, debouncedCosts],
    queryFn: () => api.newDecision(threshold, debouncedCosts),
    enabled: source === "new" && newFile !== null,
    placeholderData: keepPreviousData,
  });

  if (!summary) return null;
  if (decision.isError) return <ErrorBanner message={decision.error.message} />;
  if (!decision.data) return <PageLoading />;

  const point = decision.data.curve[thresholdPct];
  const outcome = `${summary.target} = ${summary.positive_class}`;
  const onNew = source === "new" && newFile !== null;
  const headline = onNew
    ? {
        fileName: newFile.name,
        flagged: newDecision.data?.flagged ?? 0,
        total: newFile.rows_scored,
        net: newDecision.data?.expected_net ?? 0,
      }
    : { fileName: summary.dataset_name, flagged: point.flagged, total: summary.rows_scored, net: point.net };

  return (
    <div className="space-y-10">
      {onNew && !newDecision.data ? (
        <PageLoading />
      ) : (
        <Headline {...headline} outcome={outcome} thresholdPct={thresholdPct} costs={costs} expected={onNew} />
      )}

      <section className="space-y-6 border-y border-rule py-8">
        <DecisionControls
          costs={costs}
          onCosts={setCosts}
          thresholdPct={thresholdPct}
          recommendedPct={recommendedPct}
          onThreshold={setChosenPct}
          onUseRecommended={() => setChosenPct(null)}
        />
        <NetCurve curve={decision.data.curve} current={point} recommended={decision.data.recommended} />
      </section>

      <section>
        <div role="tablist" aria-label="Result views" className="flex gap-1 border-b border-rule">
          {TABS.map((t) => (
            <button
              key={t.id}
              role="tab"
              id={`tab-${t.id}`}
              aria-selected={tab === t.id}
              aria-controls={`panel-${t.id}`}
              onClick={() => setTab(t.id)}
              className={`-mb-px whitespace-nowrap border-b-2 px-3 py-2 text-sm font-medium ${
                tab === t.id ? "border-verdict text-ink" : "border-transparent text-ink-muted hover:text-ink"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`} className="pt-6">
          {tab === "list" && (
            <div className="space-y-6">
              <div className="flex flex-wrap items-center justify-between gap-3">
                {newFile ? (
                  <div role="group" aria-label="Rows to show" className="flex gap-1">
                    {(["training", "new"] as const).map((s) => (
                      <button
                        key={s}
                        type="button"
                        aria-pressed={source === s}
                        onClick={() => setSource(s)}
                        className={source === s ? "btn-primary" : "btn-secondary"}
                      >
                        {s === "training" ? "Your history (out-of-fold)" : `New file: ${newFile.name}`}
                      </button>
                    ))}
                  </div>
                ) : (
                  <span />
                )}
                <a className="btn-secondary" href={api.exportUrl(source, threshold)} download>
                  Export CSV
                </a>
              </div>
              <RiskTable
                key={`${source}-${newFileVersion}`}
                source={source}
                threshold={threshold}
                version={newFileVersion}
                onSelect={source === "training" ? setSelected : undefined}
              />
              <details className="border-t border-rule pt-6">
                <summary className="cursor-pointer font-medium">Score a new file</summary>
                <p className="mt-2 text-sm text-ink-muted">
                  Upload current records with the same columns (the outcome column can be missing). They're scored by
                  the model trained on your history.
                </p>
                <div className="mt-4">
                  <ScoreNewFile
                    onScored={(result) => {
                      setNewFile(result);
                      setSource("new");
                      setNewFileVersion((v) => v + 1);
                    }}
                  />
                </div>
              </details>
            </div>
          )}
          {tab === "drivers" && <DriversTab summary={summary} />}
          {tab === "hood" && <UnderTheHood summary={summary} point={point} />}
        </div>
      </section>
      {selected && <WhatIfDrawer row={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
