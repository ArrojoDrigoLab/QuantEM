/**
 * Whether an adapted model is what the next run will actually use.
 *
 * `POST /api/adapters/<id>/apply/` stamps `applied_at`, and
 * `apply_active_adapter` reads it before every run. The rules it applies are
 * not obvious from anywhere in the UI, and until now the labeling screen -- the
 * one screen where the run is started -- said nothing at all: a segmentation
 * with a fine-tuned head applied at threshold 0.45 read "Objects shown: 100
 * from QuantEM" and the model picker offered "QuantEM / OmniEM / Manual /
 * None". The wizard's Apply page and the manifest both record it; the screen
 * where the work happens did not.
 *
 * The backend's rules, mirrored exactly here:
 *
 *  - the adapter must be applied to *this* segmentation (`applied_at` set);
 *  - `adapter.base_model` must equal the source model the run will use. A
 *    threshold calibrated on `quantem:mito` describes that model's probability
 *    distribution and is meaningless for `omniem:mito`, so the run silently
 *    falls back to the published model at its published threshold.
 *
 * That second rule is why this returns a value even when the adapter will *not*
 * be used: "you have an adapter and it is not going to be applied" is the more
 * surprising of the two states and the one that silently changes results.
 */

import type {
  AdaptedModelEntry,
  ModelCatalogue,
} from "@/shared/types/finetune";

export interface AppliedAdapterState {
  adapter: AdaptedModelEntry;
  /** True when this run would go through the adapter. */
  active: boolean;
  /** The source model the run will use, i.e. what the picker is set to. */
  selectedSourceModel: string | null;
  /** Threshold the released pack ships with, for the before/after. */
  publishedThreshold: number | null;
  /** True when the adapter carries a fine-tuned head, not just a threshold. */
  trainedHead: boolean;
}

/** The adapter applied to a segmentation, and whether this run will use it. */
export function appliedAdapterState(
  catalogue: ModelCatalogue | null | undefined,
  segmentationId: string | null | undefined,
  selectedSourceModel: string | null | undefined,
  selectedAdapterId?: string | null
): AppliedAdapterState | null {
  if (!catalogue || !segmentationId) return null;

  const selectedAdapter = selectedAdapterId
    ? catalogue.adapted.find(
        (entry) => entry.id === `adapted:${selectedAdapterId}` || entry.id === selectedAdapterId
      ) ?? null
    : null;
  const applied = catalogue.adapted
    .filter(
      (entry) =>
        Boolean(entry.applied_at) &&
        (entry.segmentation_id === segmentationId ||
          entry.applied_segmentation_ids?.includes(segmentationId) ||
          entry.segmentation_ids?.includes(segmentationId))
    )
    // `active_adapter_for` orders by `-applied_at`, so the most recently
    // applied one wins when more than one has ever been applied.
    .sort((left, right) =>
      String(right.applied_at ?? "").localeCompare(String(left.applied_at ?? ""))
    );
  const adapter = selectedAdapter ?? applied[0];
  if (!adapter) return null;

  const basePack = catalogue.packs.find((pack) => pack.id === adapter.base);
  return {
    adapter,
    active: Boolean(selectedSourceModel) && adapter.base === selectedSourceModel,
    selectedSourceModel: selectedSourceModel ?? null,
    publishedThreshold: basePack?.default_threshold ?? null,
    trainedHead: adapter.mode === "head",
  };
}

/** `"0.45"`. Thresholds are always shown to two places, published or not. */
export function formatThreshold(value: number | null | undefined): string | null {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(2)
    : null;
}
