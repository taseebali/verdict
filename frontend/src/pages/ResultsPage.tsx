import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { ErrorBanner } from "../components/ErrorBanner";
import { PageLoading } from "../components/Guards";
import { DecisionControls } from "../components/results/DecisionControls";
import { Headline } from "../components/results/Headline";
import { NetCurve } from "../components/results/NetCurve";
import { UnderTheHood } from "../components/results/UnderTheHood";
import { api } from "../lib/api";
import { useSummary } from "../lib/queries";
import type { Costs, ScoreResponse, Source } from "../lib/types";
import { useDebounced } from "../lib/useDebounced";

const TABS = [{ id: "hood", label: "Under the hood" }] as const;
type TabId = (typeof TABS)[number]["id"];

const DEFAULT_COSTS: Costs = { action_cost: 20, saved_value: 500, success_rate: 0.3 };

export function ResultsPage() {
  const summary = useSummary().data;
  const [costs, setCosts] = useState<Costs>(DEFAULT_COSTS);
  const debouncedCosts = useDebounced(costs, 400);
  const [chosenPct, setChosenPct] = useState<number | null>(null);
  const [tab, setTab] = useState<TabId>(TABS[0].id);
  const [source] = useState<Source>("training");
  const [newFile] = useState<ScoreResponse | null>(null);

  const decision = useQuery({
    queryKey: ["decision", debouncedCosts],
    queryFn: () => api.decision(debouncedCosts),
    placeholderData: keepPreviousData,
  });

  const recommendedPct = decision.data ? Math.round(decision.data.recommended.threshold * 100) : 50;
  const thresholdPct = chosenPct ?? recommendedPct;
  const threshold = thresholdPct / 100;

  const newDecision = useQuery({
    queryKey: ["newDecision", newFile?.name, threshold, debouncedCosts],
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
      <Headline {...headline} outcome={outcome} thresholdPct={thresholdPct} costs={costs} expected={onNew} />

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
        <div role="tablist" aria-label="Result views" className="flex gap-1 overflow-x-auto border-b border-rule">
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
          {tab === "hood" && <UnderTheHood summary={summary} point={point} />}
        </div>
      </section>
    </div>
  );
}
