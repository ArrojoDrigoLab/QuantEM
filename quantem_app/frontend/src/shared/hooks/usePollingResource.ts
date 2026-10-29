import { useCallback, useEffect, useRef, useState } from "react";

export function canRetryPollingError(error: unknown): boolean {
  const status = error instanceof Error && "status" in error ? error.status : null;
  return typeof status !== "number" || status < 400 || status >= 500 || status === 408 || status === 429;
}

/** Serialize requests and ignore late responses from a previous resource.
 * Connection failures retain the last good data and keep retrying. */
export function usePollingResource<T>(
  key: string | null,
  load: (key: string) => Promise<T>,
  isSettled: (value: T) => boolean,
  intervalMs = 1000,
) {
  const [snapshot, setSnapshot] = useState<{ key: string; data: T | null; error: Error | null } | null>(null);
  const functions = useRef({ load, isSettled });
  useEffect(() => { functions.current = { load, isSettled }; });
  const refreshRef = useRef<() => Promise<void>>(async () => {});
  useEffect(() => {
    if (!key) { refreshRef.current = async () => {}; return; }
    let disposed = false;
    let settled = false;
    let failures = 0;
    let retryable = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let pending: Promise<void> | null = null;
    const poll = (): Promise<void> => {
      if (disposed) return Promise.resolve();
      if (pending) return pending;
      clearTimeout(timer);
      pending = (async () => {
        try {
          const data = await Promise.resolve().then(() => disposed ? null : functions.current.load(key));
          if (disposed || data === null) return;
          settled = functions.current.isSettled(data);
          failures = 0;
          retryable = true;
          setSnapshot({ key, data, error: null });
        } catch (cause) {
          if (disposed) return;
          settled = false;
          failures += 1;
          retryable = canRetryPollingError(cause);
          setSnapshot((previous) => ({ key, data: previous?.key === key ? previous.data : null,
            error: cause instanceof Error ? cause : new Error("Could not refresh. Retrying…") }));
        } finally {
          pending = null;
          if (!disposed && !settled && retryable) {
            const delay = failures ? Math.min(30000, intervalMs * 2 ** Math.min(failures - 1, 5)) : intervalMs;
            timer = setTimeout(() => void poll(), delay);
          }
        }
      })();
      return pending;
    };
    refreshRef.current = poll;
    const reconnect = () => { void poll(); };
    window.addEventListener("online", reconnect);
    window.addEventListener("focus", reconnect);
    void poll();
    return () => {
      disposed = true;
      clearTimeout(timer);
      window.removeEventListener("online", reconnect);
      window.removeEventListener("focus", reconnect);
    };
  }, [key, intervalMs]);
  const data = snapshot?.key === key ? snapshot.data : null;
  return {
    data,
    error: snapshot?.key === key ? snapshot.error : null,
    retrying: snapshot?.key === key && snapshot.error !== null && canRetryPollingError(snapshot.error),
    settled: data !== null && isSettled(data),
    refresh: useCallback(() => refreshRef.current(), []),
  };
}
