import { NavLink } from "react-router-dom";
import { LayoutDashboard, Database, Brain, Target, ClipboardList } from "lucide-react";
import type { LucideIcon } from "lucide-react";

const NAV_ITEMS: { to: string; label: string; icon: LucideIcon }[] = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/data", label: "Data Explorer", icon: Database },
  { to: "/training", label: "Model Training", icon: Brain },
  { to: "/predictions", label: "Predictions", icon: Target },
  { to: "/audit", label: "Audit Logs", icon: ClipboardList },
];

export function Sidebar() {
  return (
    <nav className="w-16 bg-sidebar flex flex-col items-center py-4 gap-5 shrink-0">
      <div className="w-5 h-5 rounded-md bg-gradient-to-br from-accent-light to-accent shadow-[0_0_16px_rgba(99,102,241,0.5)]" />
      <div className="flex flex-col gap-4 mt-2">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            title={item.label}
            className={({ isActive }) =>
              `w-7 h-7 rounded-lg border flex items-center justify-center transition-colors ${
                isActive
                  ? "bg-accent/15 border-accent/30 text-accent"
                  : "border-transparent text-stone-400 hover:bg-white/5 hover:text-stone-200"
              }`
            }
          >
            <item.icon size={17} strokeWidth={1.5} />
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
