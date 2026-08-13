/**
 * The experiment list, with each experiment's datasets nested.
 *
 * One request for the whole grouping catalogue, because that is what the server
 * sends and because every consumer wants both halves at once: the library's
 * filter, the import form's two pickers, and the selection bar.
 *
 * **A failure here is silent on purpose.** Organising is optional; if the list
 * cannot be fetched, the screens that use it fall back to "no experiments yet",
 * which is a state they already have to render correctly. An error banner over
 * the library because an optional catalogue did not load would be worse than
 * the missing feature.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { getExperiments } from "@/shared/api/assets";
import type { Dataset, Experiment } from "@/shared/types/common";

export interface ExperimentCatalogue {
  experiments: Experiment[];
  /** True until the first attempt settles, either way. */
  loading: boolean;
  reload: () => Promise<void>;
}

export function useExperiments(): ExperimentCatalogue {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
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
    } catch {
      // See the module docstring: an unreachable optional catalogue is "there
      // are no experiments", not an error the library has to report.
      if (!mountedRef.current) return;
      setExperiments([]);
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  return { experiments, loading, reload };
}

/** The datasets of one shared experiment, or none for per-image experiments. */
export function datasetsFor(
  experiments: Experiment[],
  experimentId: string
): Dataset[] {
  if (!experimentId) return [];
  return experiments.find((row) => row.id === experimentId)?.datasets ?? [];
}
