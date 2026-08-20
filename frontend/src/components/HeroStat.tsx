export function HeroStat({ label, value, suffix }: { label: string; value: string; suffix?: string }) {
  return (
    <div className="bg-black/[0.03] rounded-[20px] p-1.5">
      <div className="bg-white rounded-2xl p-5 h-full border border-accent/10 shadow-tile-accent">
        <div className="text-[11px] text-stone-500 uppercase tracking-wide mb-2">{label}</div>
        <div className="text-4xl font-bold tracking-tight text-ink data-value">
          {value}
          {suffix && <span className="text-lg text-stone-400">{suffix}</span>}
        </div>
      </div>
    </div>
  );
}
