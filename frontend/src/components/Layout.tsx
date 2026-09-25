import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { StepBar } from "./StepBar";

export function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b border-rule">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-4 sm:px-6">
          <Link to="/" className="font-serif text-2xl tracking-tight">
            Verdict
          </Link>
          <StepBar />
        </div>
      </header>
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6 sm:py-12">{children}</main>
      <footer className="border-t border-rule">
        <div className="mx-auto flex max-w-6xl flex-wrap justify-between gap-2 px-4 py-6 text-sm text-ink-muted sm:px-6">
          <span>Built by Taseeb Ali</span>
          <a className="underline hover:text-ink" href="https://github.com/taseebali/verdict">
            Source on GitHub
          </a>
        </div>
      </footer>
    </div>
  );
}
