/**
 * Ground-truth composition for every segmentation the adaptation drew crops
 * from.
 *
 * Crops are gathered across every image with the same organelle segmented, so
 * the split has to be summed over each distinct `segmentation_id` in the crop
 * list rather than just the one the wizard is pointed at — otherwise the
 * headline number describes a subset of the training data.
 */

import { useMemo } from "react";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import {
  EMPTY_PROVENANCE,
  fetchGroundTruthProvenance,
  mergeProvenance,
  type GroundTruthProvenance,
} from "@/features/finetune/groundTruthProvenance";
import type { AdaptCropsResponse } from "@/shared/types/finetune";

export function useGroundTruthProvenance(crops: AdaptCropsResponse | null) {
  // Stable key so the query does not refire on every render of an equal list.
  const segmentationIds = useMemo(() => {
    const ids = new Set<string>();
    for (const crop of crops?.crops ?? []) {
      if (crop.segmentation_id) ids.add(crop.segmentation_id);
    }
    return [...ids].sort();
  }, [crops]);
  const key = segmentationIds.join(",");

  const { data, error, loading } = useApiQuery<GroundTruthProvenance | null>(
    async () => {
      if (segmentationIds.length === 0) return EMPTY_PROVENANCE;
      const parts = await Promise.all(
        segmentationIds.map((id) => fetchGroundTruthProvenance(id))
      );
      return mergeProvenance(parts);
    },
    [key]
  );

  return {
    provenance: data ?? null,
    provenanceLoading: loading,
    provenanceError: error ?? null,
  };
}
