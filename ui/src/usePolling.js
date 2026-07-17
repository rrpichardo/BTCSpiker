import { useEffect, useRef, useState } from "react";

// Polls fetchFn(signal) every intervalMs. Fires immediately on mount, then
// pauses while the tab is hidden and fires immediately again on return.
export function usePolling(fetchFn, intervalMs) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  // Ref so the effect below doesn't need to restart if the caller passes a
  // fresh fetchFn closure every render.
  const fetchFnRef = useRef(fetchFn);
  fetchFnRef.current = fetchFn;

  useEffect(() => {
    let cancelled = false;
    let intervalId = null;
    let controller = null;

    async function tick() {
      controller = new AbortController();
      try {
        const result = await fetchFnRef.current(controller.signal);
        if (cancelled) return; // ignore stale response after unmount
        setData(result);
        setError(null);
        setLastUpdated(new Date());
      } catch (err) {
        if (cancelled || err.name === "AbortError") return;
        setError(err);
      }
    }

    function startInterval() {
      if (intervalId !== null) return;
      intervalId = setInterval(tick, intervalMs);
    }

    function stopInterval() {
      if (intervalId !== null) {
        clearInterval(intervalId);
        intervalId = null;
      }
    }

    function handleVisibilityChange() {
      if (document.visibilityState === "hidden") {
        stopInterval();
      } else {
        tick();
        startInterval();
      }
    }

    tick();
    startInterval();
    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      cancelled = true;
      stopInterval();
      if (controller) controller.abort();
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [intervalMs]);

  return { data, error, lastUpdated };
}
