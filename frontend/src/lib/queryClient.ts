import { MutationCache, QueryCache, QueryClient } from "@tanstack/react-query";
import { isSessionGone } from "./api";

function onError(error: unknown) {
  if (isSessionGone(error) && window.location.pathname !== "/") {
    queryClient.clear();
    window.location.replace("/?expired=1");
  }
}

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false },
  },
  queryCache: new QueryCache({ onError }),
  mutationCache: new MutationCache({ onError }),
});
