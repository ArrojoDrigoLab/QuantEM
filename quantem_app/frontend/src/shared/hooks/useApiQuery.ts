/**
 * Hook for fetching data from the API with loading and error states.
 */

import { useEffect, useState, useCallback, useRef, type DependencyList } from "react";

export interface UseApiQueryResult<T> {
  data: T | null;
  loading: boolean;
  refetching: boolean;
  error: Error | null;
  settledRequestVersion: number;
  refetch: () => Promise<void>;
}

export function useApiQuery<T>(
  fn: () => Promise<T>,
  deps: DependencyList = []
): UseApiQueryResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [refetching, setRefetching] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);
  const [settledRequestVersion, setSettledRequestVersion] = useState(0);
  const requestIdRef = useRef(0);

  // Use ref to store the latest function without causing re-renders
  const fnRef = useRef(fn);
  useEffect(() => {
    fnRef.current = fn;
  }, [fn]);

  const run = useCallback(async (isRefetch = false) => {
    const requestId = ++requestIdRef.current;
    // Only set loading on initial load, use refetching for subsequent updates
    if (isRefetch) {
      setRefetching(true);
    } else {
      setLoading(true);
    }
    setError(null);
    try {
      const result = await fnRef.current();
      if (requestId !== requestIdRef.current) {
        return;
      }
      setData(result);
    } catch (err) {
      if (requestId !== requestIdRef.current) {
        return;
      }
      console.error("[useApiQuery] Query error:", err);
      setError(err instanceof Error ? err : new Error("Unknown error"));
      if (!isRefetch) {
        setData(null);
      }
    } finally {
      if (requestId === requestIdRef.current) {
        setSettledRequestVersion((current) => current + 1);
        if (isRefetch) {
          setRefetching(false);
        } else {
          setLoading(false);
        }
      }
    }
  }, []);

  useEffect(() => {
    void run(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run, ...deps]);

  const refetch = useCallback(() => {
    return run(true);
  }, [run]);

  return { data, loading, refetching, error, settledRequestVersion, refetch };
}
