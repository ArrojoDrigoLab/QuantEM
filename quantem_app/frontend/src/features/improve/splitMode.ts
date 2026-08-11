/**
 * What each split mode means, in one place.
 *
 * `API_CONTRACT.md` honesty rule 1 turns on these three strings: a held-out
 * Dice is only interpretable next to the mode that produced it. They live apart
 * from the components so the wording is identical in the crops step, the
 * results step, the sweep legend and the apply summary — four places a reader
 * could otherwise be told four slightly different things.
 */

import type { SplitMode } from "@/shared/types/finetune";

export const SPLIT_MODE_LABELS: Record<SplitMode, string> = {
  "image-disjoint": "image-disjoint",
  "within-image": "within-image",
  "no-heldout": "no held-out data",
};

export const SPLIT_MODE_SENTENCES: Record<SplitMode, string> = {
  "image-disjoint":
    "The held-out crops come from a different image than the ones the model was fitted on, so this score measures generalisation to a new image.",
  "within-image":
    "The held-out crops come from the same image as the fitted ones. This is a within-image score: it does not measure generalisation to a new image.",
  "no-heldout":
    "Every annotated region was used to fit, so there is no held-out score at all. Annotate a region on a second image to get one.",
};

export function splitModeTone(mode: SplitMode): "good" | "warning" {
  return mode === "image-disjoint" ? "good" : "warning";
}
