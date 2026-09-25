import { Link, useLocation } from "react-router-dom";
import { useDataset, useSummary } from "../lib/queries";

const STEPS = [
  { path: "/", label: "Data" },
  { path: "/outcome", label: "Outcome" },
  { path: "/results", label: "Verdict" },
];

export function StepBar() {
  const { pathname } = useLocation();
  const dataset = useDataset();
  const summary = useSummary(dataset.isSuccess);
  const reachable = [true, dataset.isSuccess, summary.isSuccess];

  return (
    <nav aria-label="Progress">
      <ol className="flex items-center gap-1 text-sm sm:gap-2">
        {STEPS.map((step, i) => {
          const current = pathname === step.path;
          const tone = current
            ? "bg-ink text-paper"
            : reachable[i]
              ? "text-ink hover:bg-paper-raised"
              : "text-ink-faint";
          const className = `flex items-center gap-1.5 rounded px-2.5 py-1 ${tone}`;
          const content = (
            <>
              <span>{i + 1}</span>
              <span>{step.label}</span>
            </>
          );
          return (
            <li key={step.path} className="flex items-center gap-1 sm:gap-2">
              {i > 0 && (
                <span aria-hidden="true" className="text-ink-faint">
                  ·
                </span>
              )}
              {reachable[i] && !current ? (
                <Link to={step.path} className={className}>
                  {content}
                </Link>
              ) : (
                <span className={className} aria-current={current ? "step" : undefined}>
                  {content}
                </span>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
