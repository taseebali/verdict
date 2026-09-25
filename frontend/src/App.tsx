import { QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { RequireDataset } from "./components/Guards";
import { queryClient } from "./lib/queryClient";
import { DataPage } from "./pages/DataPage";
import { OutcomePage } from "./pages/OutcomePage";

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<DataPage />} />
            <Route
              path="/outcome"
              element={
                <RequireDataset>
                  <OutcomePage />
                </RequireDataset>
              }
            />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
