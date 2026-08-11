/**
 * Everything the Improve panel says, as pure functions over the API bodies.
 *
 * Separated from the component for one reason that matters more than testing
 * convenience: **the plain-language layer may never state a direction the
 * statistics layer contradicts** (UX_PLAN §7, I-4). Both layers read the same
 * `improvement` figure — the drawer prints it, these functions word it — and
 * `copy.test.ts` asserts the two agree in sign for every branch. If the wording
 * lived inside JSX that guarantee would be a comment.
 *
 * Vocabulary is fixed by UX_PLAN §1.0 and is not negotiable here: **include
 * level** never "threshold", **kept** never "confirmed", **removed** never
 * "excluded", **checked area** never "completed ROI", and the words
 * "fine-tune", "adapter" and "Dice" do not appear in anything this module
 * returns. Dice, split mode, the oracle and the sweep all still exist — behind
 * "About the numbers", one click away, unchanged.
 */

import type {
  AdaptCrop,
  AdaptMode,
  Adapter,
  AdapterSweep,
  SplitMode,
} from "@/shared/types/finetune";

/**
 * Below this the two include levels are the same number to a reader.
 *
 * The sweep is 19 points across the range, so a difference smaller than this is
 * not something the sweep could have resolved anyway.
 */
export const LEVEL_EPSILON = 0.005;

/**
 * Below this a held-out change is not a change worth a direction word.
 *
 * Deliberately not zero: "better" for a fourth-decimal move invites the reader
 * to act on noise, and the drawer's own three-decimal rendering would show
 * `+0.000` beside it.
 */
export const IMPROVEMENT_EPSILON = 0.001;

export function formatIncludeLevel(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : value.toFixed(2);
}

function isSweep(value: unknown): value is AdapterSweep {
  return (
    typeof value === "object" &&
    value !== null &&
    Array.isArray((value as AdapterSweep).thresholds)
  );
}

export function sweepOf(adapter: Adapter | null): AdapterSweep | null {
  if (!adapter) return null;
  return isSweep(adapter.sweep) ? adapter.sweep : null;
}

// ---------------------------------------------------------------------------
// What will be learned from, before the button is pressed
// ---------------------------------------------------------------------------

function joinNames(names: string[]): string {
  if (names.length === 0) return "";
  if (names.length === 1) return names[0];
  const head = names.slice(0, -1).join(", ");
  return `${head} and ${names[names.length - 1]}`;
}

/**
 * Which images the checked areas come from, capped so the sentence stays a
 * sentence. Four images become "A, B and 2 more".
 */
function nameList(names: string[], cap = 3): string {
  const unique = [...new Set(names.filter(Boolean))];
  if (unique.length <= cap) return joinNames(unique);
  const shown = unique.slice(0, cap);
  return `${joinNames(shown)} and ${unique.length - cap} more`;
}

export interface CheckedAreaSummary {
  total: number;
  onThisImage: number;
  elsewhere: number;
  otherImageNames: string[];
}

export function summariseCheckedAreas(
  crops: AdaptCrop[] | undefined
): CheckedAreaSummary {
  const list = crops ?? [];
  const onThisImage = list.filter((crop) => crop.is_this_image).length;
  const others = list.filter((crop) => !crop.is_this_image);
  return {
    total: list.length,
    onThisImage,
    elsewhere: others.length,
    otherImageNames: [...new Set(others.map((crop) => crop.image_name ?? ""))].filter(
      Boolean
    ),
  };
}

/**
 * UX_PLAN §1.9, and the sentence exists because the old copy was false.
 *
 * "QuantEM learns from what you mark, and only from what you mark" was wrong:
 * crops are pooled across every image with the same organelle segmented. This
 * says so, and **names the images**, which turns the app's most misunderstood
 * behaviour into something the user can check.
 */
export function checkedAreasSentence(summary: CheckedAreaSummary): string {
  const { total, onThisImage, elsewhere, otherImageNames } = summary;
  if (total === 0) {
    return "You have not marked any area as checked yet, so there is nothing for me to learn from.";
  }
  const noun = total === 1 ? "area" : "areas";
  const them = total === 1 ? "it" : "them";
  const tail = `match my cut-off to what you kept in ${them}.`;
  const others = nameList(otherImageNames);

  if (elsewhere === 0) {
    return `I'll look at the ${total} ${noun} you've marked as checked on this image, and ${tail}`;
  }
  if (onThisImage === 0) {
    return (
      `I'll look at the ${total} ${noun} you've marked as checked — none on this ` +
      `image, ${elsewhere} on ${others} — and ${tail}`
    );
  }
  return (
    `I'll look at the ${total} ${noun} you've marked as checked — ${onThisImage} on ` +
    `this image and ${elsewhere} on ${others} — and ${tail}`
  );
}

/** What the run costs, stated before it is started (I-13: a stated cost is real). */
export function costSentence(mode: AdaptMode): string {
  return mode === "head"
    ? "Minutes, sometimes tens of minutes. Nothing you've drawn or kept will change."
    : "Usually about a second. Nothing you've drawn or kept will change.";
}

// ---------------------------------------------------------------------------
// What happened, after it is pressed
// ---------------------------------------------------------------------------

/**
 * The one fact both layers are derived from.
 *
 * `improvement` is held-out at the chosen level minus held-out at the default —
 * a training-set improvement is not an improvement, and the backend already
 * computes it that way. `"unknown"` is a first-class answer: with no held-out
 * area there is nothing the change could be measured against, and saying
 * "better" then would be the self-agreement trap this whole plan exists to
 * close.
 */
export type ImprovementDirection = "better" | "same" | "worse" | "unknown";

export function improvementDirection(
  sweep: AdapterSweep | null
): ImprovementDirection {
  const value = sweep?.improvement;
  if (value === null || value === undefined || Number.isNaN(value)) return "unknown";
  if (value > IMPROVEMENT_EPSILON) return "better";
  if (value < -IMPROVEMENT_EPSILON) return "worse";
  return "same";
}

/** Whether the new include level differs from the pack's published one. */
export function levelChanged(
  chosen: number | null | undefined,
  base: number | null | undefined
): boolean {
  if (chosen === null || chosen === undefined) return false;
  if (base === null || base === undefined) return false;
  return Math.abs(chosen - base) >= LEVEL_EPSILON;
}

/**
 * How the split lets the number be read, with its sample size in the sentence.
 *
 * I-4: a rendered estimate never appears without the size of the sample behind
 * it, and this is the plain-language half of that rule. The drawer's
 * `SplitModeNote` is the other half and says the same thing in its own register.
 */
export function evidenceSentence(
  splitMode: SplitMode,
  heldoutCount: number
): string {
  const areas = heldoutCount === 1 ? "1 checked area" : `${heldoutCount} checked areas`;
  if (splitMode === "no-heldout") {
    return (
      "Every checked area was used to choose the level, so there is nothing " +
      "left over to check it against."
    );
  }
  if (splitMode === "within-image") {
    return (
      `Checked against ${areas} on this same image that I did not fit to, so ` +
      "it does not tell you how it will do on a new image."
    );
  }
  return `Checked against ${areas} on a different image that I did not fit to.`;
}

/**
 * The sentence that makes the invariant visible.
 *
 * Not a reassurance: it is a description of what the extraction code does.
 * A re-run deletes only the model's own previous guesses and then drops any new
 * guess landing on an object the user kept or removed, so the counts of kept,
 * removed and hand-drawn objects are identical before and after.
 */
export const PRESERVATION_SENTENCE =
  "Nothing you have kept, removed or drawn by hand changed, and nothing will " +
  "when the model runs again. Only my own guesses are replaced.";

export interface CalibrationReport {
  /** "Done in 0.4 seconds." Omitted when the elapsed time is not known. */
  timing: string | null;
  /** The include level, with its number, and what it was before. */
  level: string;
  /** Better, the same, worse, or not knowable — and never louder than the data. */
  verdict: string;
  direction: ImprovementDirection;
  /** Which way the model was wrong, when the level moved. */
  adjustment: string | null;
  /** Sample size and split mode, in the same sentence as the claim. */
  evidence: string;
  preservation: string;
}

/**
 * The whole plain-language result, in five short sentences and no jargon.
 *
 * `elapsedSeconds` is wall-clock from the press to the result, measured by the
 * panel, because that is the number the user experienced. The job does not
 * report one for a threshold run — it takes well under a second and never had a
 * clock on it.
 */
export function calibrationReport(
  adapter: Adapter,
  elapsedSeconds: number | null
): CalibrationReport {
  const sweep = sweepOf(adapter);
  const chosen = adapter.calibrated_threshold ?? sweep?.calibrated_threshold ?? null;
  const base = adapter.default_threshold ?? null;
  const direction = improvementDirection(sweep);
  const moved = levelChanged(chosen, base);
  const heldout = (sweep?.heldout_crop_names ?? adapter.heldout_crop_names ?? []).length;

  const level = moved
    ? `New include level ${formatIncludeLevel(chosen)}, where my default is ${formatIncludeLevel(base)}.`
    : base === null
      ? `Include level ${formatIncludeLevel(chosen)}.`
      : `My default include level ${formatIncludeLevel(base)} already matched your marks best, so it stays where it is.`;

  let verdict: string;
  if (!moved) {
    verdict = "I could not do better than the level I already use.";
  } else if (direction === "better") {
    verdict = "At this level I agree with your marks better than my default did.";
  } else if (direction === "worse") {
    verdict =
      "This level matches the areas I fitted to best, but on the area I held " +
      "back my default did better. Read the numbers before you use it.";
  } else if (direction === "same") {
    verdict =
      "This level matches the areas I fitted to best, but on the area I held " +
      "back it came out the same as my default.";
  } else {
    verdict =
      "This level matches your marks best on the areas I looked at. I had " +
      "nothing left over to check that against, so I cannot say it is better.";
  }

  let adjustment: string | null = null;
  if (moved && chosen !== null && base !== null) {
    adjustment =
      chosen > base
        ? "I was including a little too much."
        : "I was leaving things out.";
  }

  return {
    timing:
      elapsedSeconds === null
        ? null
        : `Done in ${elapsedSeconds < 10 ? elapsedSeconds.toFixed(1) : Math.round(elapsedSeconds)} seconds.`,
    level,
    verdict,
    direction,
    adjustment,
    evidence: evidenceSentence(adapter.split_mode, heldout),
    preservation: PRESERVATION_SENTENCE,
  };
}
