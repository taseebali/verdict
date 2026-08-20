export function StatTile({ label, value, delta }: { label: string; value: string; delta?: string }) {
  return (
    <div className="bg-white rounded-2xl p-[18px] border border-black/5 shadow-tile">
      <div className="text-[11px] text-stone-500 uppercase tracking-wide mb-2">{label}</div>
      <div className="text-2xl font-semibold text-ink data-value">{value}</div>
      {delta && <div className="text-[11px] text-green-600 mt-1">{delta}</div>}
    </div>
  );
}
