import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useLocation, useNavigate } from "react-router-dom";
import { DataPreview } from "../components/data/DataPreview";
import { ErrorBanner } from "../components/ErrorBanner";
import { UploadDrop } from "../components/UploadDrop";
import { api } from "../lib/api";
import { keys, resetResults, useDataset } from "../lib/queries";
import type { DatasetProfile } from "../lib/types";

export function DataPage() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const location = useLocation();
  const notice = (location.state as { notice?: string } | null)?.notice;
  const expired = new URLSearchParams(location.search).get("expired") === "1";
  const dataset = useDataset();

  const onLoaded = (profile: DatasetProfile) => {
    resetResults(queryClient);
    queryClient.setQueryData(keys.dataset, profile);
  };
  const demo = useMutation({ mutationFn: api.loadDemo, onSuccess: onLoaded });
  const upload = useMutation({ mutationFn: api.uploadCsv, onSuccess: onLoaded });
  const error = demo.error ?? upload.error;
  const busy = demo.isPending || upload.isPending;

  return (
    <div className="space-y-10">
      <section className="max-w-3xl">
        <p className="eyebrow">Step 1 · Data</p>
        <h1 className="mt-3 text-4xl leading-tight sm:text-5xl">
          Upload customer records.
          <br />
          Get a verdict on who to act on.
        </h1>
        <p className="mt-4 text-lg text-ink-muted">
          Verdict learns from your history, scores every row honestly, and tells you how many to act on for the
          best return.
        </p>
      </section>

      {(notice === "no-data" || expired) && (
        <ErrorBanner message="No data loaded — your session may have expired (sessions last 1 hour). Load it again to continue." />
      )}

      <section className="grid gap-4 sm:grid-cols-[auto_1fr]">
        <button className="btn-primary px-6 py-4 text-base" onClick={() => demo.mutate()} disabled={busy}>
          {demo.isPending ? "Loading demo…" : "Try the demo dataset"}
        </button>
        <UploadDrop onFile={(file) => upload.mutate(file)} busy={upload.isPending} />
      </section>
      <p className="text-sm text-ink-muted">
        Your data stays in memory for this session only (1 hour) and is never saved.
      </p>

      {error && <ErrorBanner message={error.message} />}
      {dataset.data && <DataPreview profile={dataset.data} onContinue={() => navigate("/outcome")} />}
    </div>
  );
}
