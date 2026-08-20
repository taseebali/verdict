import type { ReactNode } from "react";
import { Sidebar } from "./Sidebar";

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen bg-canvas relative overflow-hidden">
      <div className="absolute -top-32 -right-20 w-[420px] h-[420px] rounded-full bg-[radial-gradient(circle,rgba(99,102,241,0.10),transparent_70%)] pointer-events-none" />
      <Sidebar />
      <main className="flex-1 p-8 relative z-10">{children}</main>
    </div>
  );
}
