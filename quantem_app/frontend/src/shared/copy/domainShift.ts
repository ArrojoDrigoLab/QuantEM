/**
 * The domain-shift nudge: three labelled heuristics, and the sentence they earn.
 *
 * **What this is not.** It is not an out-of-distribution detector. There is no
 * calibrated score here and no claim of one; a real OOD detector is a research
 * project, and a number that looked like one would be trusted far past what it
 * deserves. This is three arithmetic checks on numbers the run already
 * produced, and the copy says so on screen — `label` is part of the payload,
 * not decoration, and a surface that renders `message` without it is
 * misrepresenting the thing.
 *
 * **Why it exists.** A model trained on one laboratory's stain and fixation,
 * shown another's, does not announce that. It returns nothing, or it returns
 * objects it is barely willing to call objects. The screen then shows an empty
 * canvas, and every user tested read that as their own mistake: wrong image,
 * wrong click, wrong settings. The nudge names the likeliest cause and offers
 * the two things that actually work — the other model family, or twenty
 * minutes of marking up one box and adapting.
 *
 * **The three arms**, in the order they are checked, most certain first:
 *
 *  1. **Nothing at all.** A finished run that produced zero objects. The
 *     strongest signal in the set and the only one that needs no confidences.
 *  2. **Everything at the cut-off.** Almost every object sits within a hair
 *     above the threshold — the model is not finding organelles, it is finding
 *     pixels that only just cleared the bar, which is what noise does.
 *  3. **A very low mean.** The distribution as a whole sits barely above the
 *     threshold, without being as tightly piled on it as arm 2 requires.
 *
 * Arms 2 and 3 need per-object confidences and stay silent without them, which
 * is the honest behaviour: a surface that has only the object count can fire
 * arm 1 and nothing else.
 *
 * Where the numbers come from
 * ---------------------------
 * MEASURED, not chosen. `quantem:mito` on CPU, threshold 0.5, through the
 * product's own closing and minimum-area filters, over four real EM images
 * from four different datasets in the fig4 ground-truth set and two synthetic
 * noise images:
 *
 * | image | objects | mean confidence | share within 0.05 of the cut-off |
 * |---|---:|---:|---:|
 * | `deeppi_em_skeletal_muscle te_00021` | 15 | 0.708 | 0.13 |
 * | `deepcontact_cell cell_00066` | 73 | 0.814 | 0.00 |
 * | `zenodo_mitoem2 ME2-Sperm_00857` | 118 | 0.705 | 0.03 |
 * | `orgsegnet_plant tr_00219` | 22 | 0.608 | 0.32 |
 * | uniform noise | **0** | — | — |
 * | smoothed gaussian noise | **0** | — | — |
 *
 * Two things that table decides, and one it reports:
 *
 *  - `cutOffShare` is 0.8 because the worst real image put 0.32 of its objects
 *    in the band. Anything under about 0.5 would fire on a plant sample the
 *    model handled correctly.
 *  - `lowMeanMargin` is 0.05, not the 0.10 first written: the same plant image
 *    has a mean of 0.608, which clears a 0.10 margin by 0.008 — inside the
 *    noise of the measurement, so a slightly different crop would have fired
 *    the nudge on a good result. At 0.05 the closest real case clears by more
 *    than the whole band.
 *  - **On this build, out-of-domain input produces nothing at all**, not a
 *    pile of barely-confident objects: mean probability over the whole noise
 *    image is 0.0002, and the 99th percentile is 0.002. So arm 1 is the arm
 *    that fires, and arms 2 and 3 are provision for the intermediate case — a
 *    stain the model half-recognises — which this measurement did not produce
 *    and therefore did not validate. Their constants are bounded by the real
 *    images above (they must not fire there) and are otherwise unproven.
 */

/** The numbers each arm turns on, named so they can be argued with. */
export const DOMAIN_SHIFT_THRESHOLDS = {
  /**
   * Below this many objects, the confidence arms are not evaluated: a mean
   * over three numbers is not a distribution, and firing on it produces a
   * confident-sounding sentence from noise. Conservative on purpose — a real
   * 15-object image measured 0.708 and would not have fired anyway, so the
   * floor costs nothing that has been observed.
   */
  minObjectsForDistribution: 20,
  /** How far above the cut-off still counts as "at the cut-off". */
  cutOffBand: 0.05,
  /** The share of objects inside that band that makes arm 2 fire. */
  cutOffShare: 0.8,
  /** How far above the cut-off the *mean* has to be to clear arm 3. */
  lowMeanMargin: 0.05,
} as const;

/** Which arm fired. Wire values, never shown to a user. */
export type DomainShiftReason =
  | "no_objects"
  | "confidence_at_the_cut_off"
  | "low_mean_confidence";

export interface DomainShiftEvidence {
  /** Objects the run produced. Required: arm 1 is nothing but this. */
  objectCount: number;
  /**
   * True once the run has finished. A run still in flight has produced zero
   * objects for a reason that is not domain shift.
   */
  runFinished: boolean;
  /**
   * Per-object mean foreground probability, when the surface has them. `null`
   * or absent silences arms 2 and 3 rather than guessing at them.
   */
  confidences?: readonly number[] | null;
  /** The foreground threshold the run used, if it is known. */
  threshold?: number | null;
}

export interface DomainShiftNudge {
  reason: DomainShiftReason;
  /**
   * The label, which is not optional. Rendered with the message, every time.
   */
  label: string;
  /** The sentence itself. */
  message: string;
  /** Why this fired, in numbers, for the person who wants to check it. */
  evidence: string;
}

/**
 * The one sentence, verbatim from the UX plan.
 *
 * First person on purpose: the model is speaking about itself, which is what
 * makes it an admission rather than an accusation. The user has just been shown
 * an empty screen; the app taking the blame by name is the whole point of the
 * surface.
 */
export const DOMAIN_SHIFT_MESSAGE =
  "This does not look like the images I was trained on. That usually means the stain or fixation is different. Try the other model family, or mark up one box and let me learn from it.";

/**
 * The label. Short, plain, and honest about what produced the sentence above.
 *
 * "Might be" and "a guess" are load-bearing. Without them the nudge reads as a
 * measurement, and a user who trusts it and switches families for the wrong
 * reason has been misled by us rather than by their data.
 */
export const DOMAIN_SHIFT_LABEL =
  "A guess from the numbers, not a measurement";

function mean(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function round(value: number, places = 2): string {
  return value.toFixed(places);
}

/**
 * Whether to nudge, and why, or `null` for "say nothing".
 *
 * `null` is the common answer and the default one. The nudge is worth its
 * pixels only when the screen is otherwise unexplained; on a run that found a
 * normal population of confident objects it would be noise, and noise is how a
 * warning stops being read.
 */
export function assessDomainShift(
  evidence: DomainShiftEvidence
): DomainShiftNudge | null {
  const { objectCount, runFinished, threshold } = evidence;
  if (!runFinished) return null;

  if (objectCount === 0) {
    return {
      reason: "no_objects",
      label: DOMAIN_SHIFT_LABEL,
      message: DOMAIN_SHIFT_MESSAGE,
      evidence: "The run finished and found nothing at all.",
    };
  }

  const confidences = (evidence.confidences ?? []).filter(
    (value) => Number.isFinite(value)
  );
  if (
    confidences.length < DOMAIN_SHIFT_THRESHOLDS.minObjectsForDistribution ||
    confidences.length !== evidence.confidences?.length ||
    typeof threshold !== "number" ||
    !Number.isFinite(threshold)
  ) {
    return null;
  }

  const { cutOffBand, cutOffShare, lowMeanMargin } = DOMAIN_SHIFT_THRESHOLDS;
  const nearCutOff = confidences.filter(
    (value) => value < threshold + cutOffBand
  ).length;
  const share = nearCutOff / confidences.length;
  if (share >= cutOffShare) {
    return {
      reason: "confidence_at_the_cut_off",
      label: DOMAIN_SHIFT_LABEL,
      message: DOMAIN_SHIFT_MESSAGE,
      evidence: `${Math.round(share * 100)}% of the ${
        confidences.length
      } objects sit within ${cutOffBand} of the ${round(
        threshold
      )} cut-off, so almost nothing here cleared it by a margin.`,
    };
  }

  const average = mean(confidences);
  if (average < threshold + lowMeanMargin) {
    return {
      reason: "low_mean_confidence",
      label: DOMAIN_SHIFT_LABEL,
      message: DOMAIN_SHIFT_MESSAGE,
      evidence: `Average confidence across the ${
        confidences.length
      } objects is ${round(average)}, barely above the ${round(
        threshold
      )} cut-off.`,
    };
  }

  return null;
}
