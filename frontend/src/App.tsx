import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/queryClient";
import { AppShell } from "./components/layout/AppShell";
import { Dashboard } from "./pages/Dashboard";
import { DataExplorer } from "./pages/DataExplorer";
import { ModelTraining } from "./pages/ModelTraining";
import { Predictions } from "./pages/Predictions";
import { AuditLogs } from "./pages/AuditLogs";

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppShell>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/data" element={<DataExplorer />} />
            <Route path="/training" element={<ModelTraining />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/audit" element={<AuditLogs />} />
          </Routes>
        </AppShell>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
