/**
 * Reading a completion preview into the sentences a dialog needs.
 *
 * `GET /api/segmentations/<id>/complete` answers "what would marking this done
 * destroy?" — a count, a breakdown by source model, and whether the discard
 * could be undone. Everything here is presentation on top of that; the numbers
 * are never derived from the segmentation payload, because `POST` compares the
 * acknowledged count against a fresh read and refuses a stale one.
 */

import type {
  SegmentationCompletionPreview,
  SourceModelOption,
} from "@/shared/types";

export interface DiscardedBySourceModel {
  value: string;
  label: string;
  count: number;
}

/** Per-model breakdown of the doomed objects, largest contributor first. */
export function discardBySourceModel(
  preview: SegmentationCompletionPreview | null,
  sourceModelOptions: SourceModelOption[] = []
): DiscardedBySourceModel[] {
  if (!preview) return [];
  return Object.entries(preview.discard_by_source_model ?? {})
    .filter(([, count]) => count > 0)
    .map(([value, count]) => ({
      value,
      label:
        sourceModelOptions.find((option) => option.value === value)?.label ??
        value,
      count,
    }))
    .sort((left, right) => right.count - left.count);
}

/** `"32 objects"` / `"1 object"`. */
export function pluraliseObjects(count: number): string {
  return `${count} object${count === 1 ? "" : "s"}`;
}

/**
 * `label_state` for an object a person looked at and threw away.
 *
 * `quantem.analysis.loaders` calls this one `REJECTED`; the model calls it
 * `EXCLUDED`. It is the *record of a review*, not a leftover.
 */
const REJECTED_LABEL_STATE = "EXCLUDED";

export interface DiscardBreakdown {
  /** Objects a person looked at and rejected. */
  rejected: number;
  /** Objects nobody ever looked at: candidates and inferences. */
  neverReviewed: number;
}

/**
 * The doomed set split into the two things it actually contains.
 *
 * "Also delete the 32 objects nobody confirmed" is arithmetically right and
 * descriptively wrong: `discardable_queryset` is everything that is not
 * `CONFIRMED`, which lumps candidates nobody has opened together with objects
 * somebody opened and rejected. They cost the same on the count and are not the
 * same thing to lose — a rejection is work, it is the evidence that a region
 * was reviewed, and `fetchGroundTruthProvenance` feeds it to the fine-tuning
 * wizard as a negative example. Deleting it silently shrinks the training set
 * for the next adaptation.
 *
 * `neverReviewed` is derived by subtraction rather than by adding up the states
 * it knows about, so a label state added server-side lands in "nobody looked at
 * it" instead of vanishing from the sentence.
 */
export function discardBreakdown(
  preview: SegmentationCompletionPreview | null
): DiscardBreakdown {
  if (!preview) return { rejected: 0, neverReviewed: 0 };
  const byState = preview.discard_by_label_state ?? {};
  const rejected = Math.max(0, byState[REJECTED_LABEL_STATE] ?? 0);
  return {
    rejected,
    neverReviewed: Math.max(0, preview.discard_count - rejected),
  };
}
