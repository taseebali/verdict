export function Chip({ label, selected, onClick }: { label: string; selected: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`border rounded-md px-2 py-1 text-[11px] transition-colors ${
        selected
          ? "bg-white border-black/10 text-zinc-700"
          : "bg-stone-100 border-transparent text-stone-400"
      }`}
    >
      {label}
    </button>
  );
}
