import { useCallback, useEffect, useRef, useState } from "react";

import { isAbortError } from "../lib/errors";

export interface ApiState<T> {
  data: T | undefined;
  error: unknown;
  /** True during the first load (no data yet). */
  loading: boolean;
  /** Fetch again, keeping the current data on screen meanwhile. */
  reload: () => void;
  setData: (data: T | undefined) => void;
}

/**
 * Load data with `loader` when the component mounts and whenever `deps` change.
 * Responses of outdated requests are ignored, and requests are aborted on unmount.
 * Pass `null` as loader to skip loading.
 */
export function useApi<T>(
  loader: ((signal: AbortSignal) => Promise<T>) | null,
  deps: readonly unknown[],
): ApiState<T> {
  const [data, setData] = useState<T | undefined>(undefined);
  const [error, setError] = useState<unknown>(undefined);
  const [loading, setLoading] = useState<boolean>(loader !== null);
  const [version, setVersion] = useState(0);
  const loaderRef = useRef(loader);
  loaderRef.current = loader;
  const depsKey = JSON.stringify(deps);

  useEffect(() => {
    const currentLoader = loaderRef.current;
    if (currentLoader === null) {
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    currentLoader(controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) {
          setData(result);
          setError(undefined);
        }
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted && !isAbortError(reason)) {
          setError(reason);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [depsKey, version]);

  // New dependencies mean other data: forget what is on screen.
  const previousKey = useRef(depsKey);
  if (previousKey.current !== depsKey) {
    previousKey.current = depsKey;
    setData(undefined);
    setError(undefined);
    setLoading(loader !== null);
  }

  const reload = useCallback(() => setVersion((value) => value + 1), []);
  return { data, error, loading, reload, setData };
}
