import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../../lib/api";
import type { ScoreResponse } from "../../lib/types";
import { ErrorBanner } from "../ErrorBanner";
import { UploadDrop } from "../UploadDrop";

export function ScoreNewFile({ onScored }: { onScored: (result: ScoreResponse) => void }) {
  const queryClient = useQueryClient();
  const score = useMutation({
    mutationFn: api.scoreFile,
    onSuccess: (result) => {
      queryClient.removeQueries({ queryKey: ["rows", "new"] });
      queryClient.removeQueries({ queryKey: ["newDecision"] });
      onScored(result);
    },
  });
  return (
    <div className="space-y-3">
      <UploadDrop
        label="Score a new file with this model"
        busy={score.isPending}
        onFile={(file) => score.mutate(file)}
      />
      {score.error && <ErrorBanner message={score.error.message} />}
    </div>
  );
}
