import { useId, useState } from "react";

interface Props {
  onFile: (file: File) => void;
  busy: boolean;
  label?: string;
}

export function UploadDrop({ onFile, busy, label = "Upload a CSV" }: Props) {
  const [dragging, setDragging] = useState(false);
  const inputId = useId();

  return (
    <label
      htmlFor={inputId}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const file = e.dataTransfer.files[0];
        if (file && !busy) onFile(file);
      }}
      className={`focus-ring flex cursor-pointer flex-col justify-center rounded border border-dashed px-6 py-4 transition-colors ${
        dragging ? "border-verdict bg-paper-raised" : "border-ink-faint hover:bg-paper-raised"
      }`}
    >
      <span className="font-medium">{busy ? "Reading file…" : label}</span>
      <span className="text-sm text-ink-muted">
        Drop a file here or click to choose · CSV up to 20 MB
      </span>
      <input
        id={inputId}
        type="file"
        accept=".csv,text/csv"
        className="sr-only"
        disabled={busy}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          e.target.value = "";
        }}
      />
    </label>
  );
}
