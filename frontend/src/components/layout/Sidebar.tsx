import { NavLink } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard" },
  { to: "/data", label: "Data Explorer" },
  { to: "/training", label: "Model Training" },
  { to: "/predictions", label: "Predictions" },
  { to: "/audit", label: "Audit Logs" },
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
              `w-7 h-7 rounded-lg border transition-colors ${
                isActive
                  ? "bg-accent/15 border-accent/30"
                  : "border-transparent hover:bg-white/5"
              }`
            }
          />
        ))}
      </div>
    </nav>
  );
}
