/**
 * Hook for API mutations (POST, PUT, PATCH, DELETE) with loading and error states.
 */

import { useState, useCallback } from "react";

export interface UseApiMutationResult<TReq, TRes> {
  mutate: (data: TReq) => Promise<TRes | undefined>;
  loading: boolean;
  error: Error | null;
  reset: () => void;
}

export interface UseApiMutationOptions<TRes> {
  onSuccess?: (result: TRes) => void;
  onError?: (error: Error) => void;
}

export function useApiMutation<TReq, TRes>(
  fn: (data: TReq) => Promise<TRes>,
  options?: UseApiMutationOptions<TRes>
): UseApiMutationResult<TReq, TRes> {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);

  const mutate = useCallback(
    async (data: TReq): Promise<TRes | undefined> => {
      setLoading(true);
      setError(null);
      try {
        const result = await fn(data);
        if (options?.onSuccess) {
          options.onSuccess(result);
        }
        return result;
      } catch (err) {
        const nextError = err instanceof Error ? err : new Error("Unknown error");
        setError(nextError);
        if (options?.onError) {
          options.onError(nextError);
        }
        return undefined;
      } finally {
        setLoading(false);
      }
    },
    [fn, options]
  );

  const reset = useCallback(() => {
    setError(null);
    setLoading(false);
  }, []);

  return { mutate, loading, error, reset };
}








