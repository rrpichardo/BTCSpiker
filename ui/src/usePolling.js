import { useEffect, useRef, useState } from "react";

// Polls fetchFn(signal) every intervalMs. Fires immediately on mount, then
// pauses while the tab is hidden and fires immediately again on return.
export function usePolling(fetchFn, intervalMs) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isPaused, setIsPaused] = useState(document.visibilityState === "hidden");

  // Ref so the effect below doesn't need to restart if the caller passes a
  // fresh fetchFn closure every render.
  const fetchFnRef = useRef(fetchFn);
  fetchFnRef.current = fetchFn;

  useEffect(() => {
    let cancelled = false;
    let timeoutId = null;
    let controller = null;
    let generation = 0;

    function clearScheduledTick() {
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
    }

    function scheduleNextTick() {
      clearScheduledTick();
      if (!cancelled && document.visibilityState !== "hidden") {
        timeoutId = setTimeout(tick, intervalMs);
      }
    }

    async function tick() {
      if (cancelled || document.visibilityState === "hidden") return;

      const requestGeneration = ++generation;
      const requestController = new AbortController();
      controller = requestController;
      setIsRefreshing(true);

      try {
        const result = await fetchFnRef.current(requestController.signal);
        if (cancelled || requestGeneration !== generation) return;
        setData(result);
        setError(null);
        setLastUpdated(new Date());
      } catch (err) {
        if (
          cancelled ||
          requestGeneration !== generation ||
          err.name === "AbortError"
        ) {
          return;
        }
        setError(err);
      } finally {
        if (cancelled || requestGeneration !== generation) return;
        controller = null;
        setIsLoading(false);
        setIsRefreshing(false);
        scheduleNextTick();
      }
    }

    function handleVisibilityChange() {
      if (document.visibilityState === "hidden") {
        generation += 1;
        clearScheduledTick();
        controller?.abort();
        controller = null;
        setIsPaused(true);
        setIsRefreshing(false);
      } else {
        setIsPaused(false);
        tick();
      }
    }

    document.addEventListener("visibilitychange", handleVisibilityChange);
    if (document.visibilityState === "hidden") {
      setIsPaused(true);
    } else {
      tick();
    }

    return () => {
      cancelled = true;
      generation += 1;
      clearScheduledTick();
      controller?.abort();
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [intervalMs]);

  return {
    data,
    error,
    lastUpdated,
    isLoading,
    isRefreshing,
    isPaused,
  };
}
