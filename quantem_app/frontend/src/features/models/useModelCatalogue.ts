/**
 * The model catalogue, for screens that need to know whether a run can work.
 *
 * Deliberately tolerant: `GET /api/models/` failing must never take a screen
 * down, because every screen that asks this question has a job to do without
 * the answer. A missing catalogue resolves to "unknown" runnability everywhere,
 * which suppresses the claim rather than the feature.
 */

import { getModelCatalogue } from "@/shared/api/finetune";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import type { ModelCatalogue } from "@/shared/types/finetune";

export function useModelCatalogue() {
  const { data, error, loading, refetch } = useApiQuery<ModelCatalogue | null>(
    () => getModelCatalogue(),
    []
  );
  return {
    catalogue: data ?? null,
    catalogueError: error ?? null,
    catalogueLoading: loading,
    refetchCatalogue: refetch,
  };
}
