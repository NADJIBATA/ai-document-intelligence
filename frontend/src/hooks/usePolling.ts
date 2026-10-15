import { useEffect, useRef } from "react";

/** Call `callback` every `intervalMs` while `enabled` is true (and the tab is visible). */
export function usePolling(callback: () => void, intervalMs: number, enabled: boolean): void {
  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  useEffect(() => {
    if (!enabled) {
      return;
    }
    const timer = window.setInterval(() => {
      if (document.visibilityState !== "hidden") {
        callbackRef.current();
      }
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [intervalMs, enabled]);
}
