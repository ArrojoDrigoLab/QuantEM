/**
 * The experiment list, with each experiment's datasets nested.
 *
 * One request for the whole grouping catalogue, because that is what the server
 * sends and because every consumer wants both halves at once: the library's
 * filter, the import form's two pickers, and the selection bar.
 *
 * Retains the last loaded catalogue on failures and exposes a retryable error.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { getExperiments } from "@/shared/api/assets";
import type { Dataset, Experiment } from "@/shared/types/common";

export interface ExperimentCatalogue {
  experiments: Experiment[];
  /** True until the first attempt settles, either way. */
  loading: boolean;
  error: string | null;
  reload: () => Promise<void>;
}

export function useExperiments(): ExperimentCatalogue {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const reload = useCallback(async () => {
    try {
      const rows = await getExperiments();
      if (!mountedRef.current) return;
      setExperiments(rows);
      setError(null);
    } catch {
      if (!mountedRef.current) return;
      setError("Experiments could not be loaded. Your saved images have not been removed.");
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  return { experiments, loading, error, reload };
}

/** The datasets of one shared experiment, or none for per-image experiments. */
export function datasetsFor(
  experiments: Experiment[],
  experimentId: string
): Dataset[] {
  if (!experimentId) return [];
  return experiments.find((row) => row.id === experimentId)?.datasets ?? [];
}
