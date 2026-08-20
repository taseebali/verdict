import type { ReactNode } from "react";

export function EmptyState({ title, description, action }: { title: string; description: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center text-center py-20 border border-dashed border-black/10 rounded-2xl">
      <div className="text-sm font-medium text-ink mb-1">{title}</div>
      <div className="text-xs text-stone-500 mb-4 max-w-xs">{description}</div>
      {action}
    </div>
  );
}
