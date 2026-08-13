/**
 * Where the objects on the labeling screen came from.
 *
 * Two different claims get made in this header and they must not be conflated:
 *
 *   - **Model to run** -- the source model the next inference run will use.
 *   - **Objects shown** -- the model that produced the objects currently
 *     listed and rasterised.
 *
 * They coincide once something has been run with the selected model, and they
 * do not before that. Rendering only the first (which is what a badge reading
 * "Model: MitoNet v1_mini" did) tells the user a model produced objects it
 * never touched, and that claim then travels into a methods section.
 *
 * A third claim hides inside "no objects", and getting it wrong sends the user
 * the opposite way from the fix. **Not yet run** and **ran and found nothing**
 * both show zero, and this said the first for both: "Nothing has been run with
 * QuantEM on this image ... Run model to produce some, or choose
 * another model." A calibrated 5 nm/px image, run to completion with zero
 * candidates, got that -- so the reading was *press the button*, when the button
 * had been pressed and would produce the same nothing again, and the suggested
 * lever ("choose another model") is not the one that moves. The pixel size is:
 * the same pixels measured 0, 19, 120 and 233 objects at 5 nm, unset, 10 nm and
 * 20 nm. `run_notice` is the server's own finding about that run, and it names
 * the pixel size first; the branch that renders it exists so the user is told
 * the run happened.
 *
 * A fourth claim is the worst of them, and it was silent. `status_stage` can be
 * `FAILED` -- the runner sets it when a worker dies, and
 * `reconcile_domain_objects_for_cancelled_job` sets it when the user cancels --
 * with `status_error` carrying a specific sentence. Nothing rendered either.
 * The objects still on screen are the *previous* run's, so the header reported
 * them in the ordinary way: a user cancelled an ER re-run and read
 * "190 confirmed of 190 from QuantEM", in green, over 190 objects the failed run
 * had nothing to do with. Re-run at a corrected pixel size, have it fail, and
 * the wrongly-scaled objects still look like the finished answer. The failure
 * is checked before every other branch because it is a fact about the
 * segmentation that changes what every count below it means.
 *
 * A fifth is arithmetic rather than provenance, and it is the one a reader
 * notices first. `confirmed` is segmentation-wide; `count` is the selected
 * model's. The summary joined them with the word "of" unconditionally, so
 * confirming QuantEM's output and flipping the selector to OmniEM to compare
 * gave **"41 confirmed of 3 from OmniEM"**. The two are only a fraction of one
 * another when the selected model produced every object here, which
 * `isSoleSourceOfEveryObject` is the test for; otherwise they are separate
 * facts and the chip separates them.
 */

import type { ImageSegmentation, SourceModelOption } from "@/shared/types";

/**
 * Synthetic selection meaning "list nothing model-derived".
 *
 * Not a backend source model: the segment filters treat an unrecognised value
 * as "confirmed/manual only", which is exactly the intent.
 */
export const NONE_SOURCE_MODEL = "none";
const MANUAL_SOURCE_MODEL = "manual";

/**
 * `error` is not a louder `warning`.
 *
 * "Ran and found no objects" and "the run you launched never produced anything"
 * are different events, and the second one leaves numbers on screen that belong
 * to a different run. They needed to be distinguishable at a glance, not only
 * in the sentence.
 */
export type ProvenanceTone = "good" | "warning" | "neutral" | "error";

export interface DisplayedObjectsDescription {
  /** One short line for the badge. */
  summary: string;
  /** The full explanation, shown as a tooltip. */
  detail: string;
  tone: ProvenanceTone;
}

/** Human label for a source-model value, falling back to the raw value. */
export function resolveSourceModelLabel(
  value: string | null,
  options: SourceModelOption[]
): string {
  if (!value || value === NONE_SOURCE_MODEL) return "no model";
  return options.find((option) => option.value === value)?.label ?? value;
}

/**
 * Objects attributed to one source model.
 *
 * Reads `source_models[].count`, which the serializer builds from the raw
 * per-source rows. Do not derive this from `segment_counts_by_source_model`:
 * that map deliberately overwrites every bucket's `CONFIRMED` with the
 * segmentation-wide confirmed total, so summing a bucket double-counts.
 */
function countForSourceModel(
  value: string,
  options: SourceModelOption[]
): number | null {
  const option = options.find((candidate) => candidate.value === value);
  return typeof option?.count === "number" ? option.count : null;
}

/**
 * True when every object on this segmentation is attributed to `value`.
 *
 * This is the whole licence for the word "of". `confirmed` is a
 * segmentation-wide total and `count` is one model's, so
 * "`confirmed` confirmed **of** `count` from `label`" is a claim that the
 * confirmed objects are a subset of that model's — and it is only true when
 * that model produced everything here. It routinely is not: the serializer
 * always lists a `manual` bucket alongside the models, and a second model that
 * has been run has a bucket of its own. Flipping the selector to compare then
 * produced arithmetic nonsense — reported from the running app as
 * **"41 confirmed of 3 from OmniEM"** and "38 confirmed of 20 from OmniEM" —
 * with the tooltip underneath saying the right thing and the visible chip
 * saying a false one.
 *
 * Every option must report a numeric `count`: an unknown bucket could hold
 * anything, and "I do not know" is not evidence of a subset.
 */
function isSoleSourceOfEveryObject(
  value: string,
  count: number,
  options: SourceModelOption[]
): boolean {
  if (count <= 0 || options.length === 0) return false;
  let attributed = 0;
  for (const option of options) {
    if (typeof option.count !== "number") return false;
    attributed += option.count;
  }
  return attributed === count && options.some((option) => option.value === value);
}

/** Every object on this segmentation, whatever its label state. */
function totalObjects(segmentation: ImageSegmentation): number {
  const counts = segmentation.segment_counts;
  if (!counts) return 0;
  return Object.values(counts).reduce(
    (sum, value) => sum + (typeof value === "number" ? value : 0),
    0
  );
}

/**
 * A run that stopped without producing anything, and the objects it did not
 * replace.
 *
 * `status_stage === "FAILED"` is set by exactly one place --
 * `quantem.jobs.failure_reconcile` -- for a dead worker, a cancelled job and a
 * job removed from the queue before it started; `status_error` is that module's
 * own sentence and is reproduced verbatim, because it is the only text that
 * distinguishes those three. It is checked before the count branches for the
 * same reason `run_notice` is checked before them, only more so: the counts are
 * not merely uninformative here, they are *about a different run*, and a chip
 * that reports them without qualification is how "190 confirmed of 190 from
 * QuantEM" came to sit, in green, over a re-run the user had just cancelled.
 *
 * The failure survives on the segmentation until something moves the stage
 * again, so this does not need the job row and keeps saying so after a reload.
 */
function describeFailedRun(
  segmentation: ImageSegmentation | null,
  staleNote: string
): DisplayedObjectsDescription | null {
  if (segmentation?.status_stage !== "FAILED") return null;

  const confirmed = segmentation.segment_counts?.CONFIRMED ?? 0;
  const total = totalObjects(segmentation);
  // The model is free-form text server-side and "" is its default, so an empty
  // string has to read as "no reason given" rather than as a blank sentence.
  const reason = segmentation.status_error?.trim() || null;

  const listed =
    total === 0
      ? "Nothing is listed here"
      : `The ${total} object${total === 1 ? "" : "s"} listed here — ${confirmed} of them confirmed — ${
          total === 1 ? "was" : "were"
        } already on this segmentation before that run started`;

  return {
    summary:
      total === 0
        ? "Last run failed · no objects"
        : `Last run failed · ${confirmed} confirmed predate it`,
    detail: [
      "The last run on this segmentation failed and saved no objects.",
      reason ?? "The server recorded no reason for it.",
      `${listed}. Nothing on screen is that run's output.`,
    ].join(" ") + staleNote,
    tone: "error",
  };
}

export function describeDisplayedObjects({
  segmentation,
  sourceModelOptions,
  activeSourceModel,
  displayedSourceModel,
}: {
  segmentation: ImageSegmentation | null;
  sourceModelOptions: SourceModelOption[];
  activeSourceModel: string;
  displayedSourceModel?: string | null;
}): DisplayedObjectsDescription {
  const confirmed = segmentation?.segment_counts?.CONFIRMED ?? 0;

  // The overlay raster is built per source model and can lag a selector change
  // by one poll. Saying which model the pixels on screen belong to is the whole
  // point of this badge, so a mismatch is reported rather than smoothed over.
  const staleOverlay =
    displayedSourceModel != null &&
    displayedSourceModel !== "" &&
    displayedSourceModel !== activeSourceModel &&
    activeSourceModel !== NONE_SOURCE_MODEL;
  const staleNote = staleOverlay
    ? ` The overlay still shows output from ${resolveSourceModelLabel(
        displayedSourceModel,
        sourceModelOptions
      )}; it will catch up on the next refresh.`
    : "";

  // Before everything, including the None selection: a failed run is a fact
  // about the segmentation, and the objects the other branches would describe
  // are not the ones that run produced.
  const failed = describeFailedRun(segmentation, staleNote);
  if (failed) return failed;

  if (activeSourceModel === NONE_SOURCE_MODEL) {
    return {
      summary: `Objects shown: manual only (${confirmed} confirmed)`,
      detail:
        "No model output is listed. Only objects you confirmed or drew by hand are shown.",
      tone: "neutral",
    };
  }

  const label = resolveSourceModelLabel(activeSourceModel, sourceModelOptions);
  const count = countForSourceModel(activeSourceModel, sourceModelOptions);

  // Before either count branch, because this is a fact about the segmentation
  // rather than about the selected model, and because it is the one state where
  // both counts (`count === 0`, and `count === null` with nothing confirmed)
  // would otherwise describe a finished run as one that never happened.
  const runNotice = segmentation?.run_notice ?? null;
  const noticeMatchesDisplayedModel = Boolean(
    runNotice?.source_model &&
      activeSourceModel !== NONE_SOURCE_MODEL &&
      activeSourceModel !== MANUAL_SOURCE_MODEL &&
      runNotice.source_model === activeSourceModel &&
      (!displayedSourceModel || displayedSourceModel === activeSourceModel)
  );
  if (runNotice && noticeMatchesDisplayedModel) {
    return {
      // The server's own short line (`run_notice.summary`), not one composed
      // here. There are two kinds of empty run and they need different chips:
      // "Ran and found no objects" is right for a run over an unlabelled
      // image and false over a proofread segmentation holding twelve
      // confirmed objects, which is exactly where the `no_new_objects` kind
      // fires. The fallback covers an older backend that predates `summary`;
      // such a backend only emits the empty-run kind, so the old wording is
      // the right one there. ("Ran and found none", not "none yet": the old
      // chip read "No objects from QuantEM yet" over a completed run, which
      // says *press the button*; the run has already been pressed.)
      summary: runNotice.summary || "Ran and found no objects",
      detail: [runNotice.message, ...runNotice.next_steps].join(" ") + staleNote,
      tone: "warning",
    };
  }

  if (count === null) {
    return {
      summary: `${confirmed} confirmed · ${label}`,
      detail: `${confirmed} object${
        confirmed === 1 ? " is" : "s are"
      } confirmed, which is what the analysis measures. The objects listed are the ones attributed to ${label}; this build did not report a per-model object count.${staleNote}`,
      tone: staleOverlay ? "warning" : "neutral",
    };
  }

  // Reached only when `run_notice` is absent, which for this segmentation means
  // other models have produced objects here and this one has not -- i.e. it
  // genuinely has not been run. The server owns that distinction (the notice is
  // derived from the stage *and* the object count), so this branch does not
  // re-derive it from `status_stage` and risk disagreeing.
  if (count === 0) {
    return {
      summary: `No objects from ${label} yet`,
      detail: `No successful empty result from ${label} is displayed on this image. Run model to produce one, or choose another model.${staleNote}`,
      tone: "warning",
    };
  }

  // `confirmed` is segmentation-wide and `count` is this model's tally. They
  // can only be read as a fraction of one another when this model produced
  // every object here; otherwise they are two separate facts and the chip says
  // so with a separator rather than the word "of".
  const soleSource = isSoleSourceOfEveryObject(
    activeSourceModel,
    count,
    sourceModelOptions
  );

  // Exactly `count - confirmed` when this model produced everything, and a
  // floor otherwise: some of the confirmed objects may belong to another
  // bucket, which can only make this model's unconfirmed tally larger. Skipped
  // entirely when `confirmed >= count`, where the subtraction is meaningless.
  const unconfirmed = count > confirmed ? count - confirmed : null;
  const remainder =
    unconfirmed === null
      ? ""
      : soleSource
        ? ` The other ${unconfirmed} attributed to ${label} ${
            unconfirmed === 1
              ? "is an unconfirmed candidate"
              : "are unconfirmed candidates"
          } and are not measured.`
        : ` At least ${unconfirmed} attributed to ${label} ${
            unconfirmed === 1
              ? "is an unconfirmed candidate"
              : "are unconfirmed candidates"
          } and are not measured.`;

  const provenanceNote = soleSource
    ? ""
    : ` The ${count} listed here ${
        count === 1 ? "is the object" : "are the objects"
      } attributed to ${label}; the confirmed total counts everything confirmed on this segmentation, whatever produced it, so the two are not a fraction of one another.`;

  return {
    // Confirmed leads, because it is the number the analysis uses and the
    // number that belongs in a figure legend. The attributed count used to be
    // the whole chip -- "Objects shown: 42 from QuantEM", in green, after the
    // user had confirmed 28 and rejected 14 -- with the 28 reachable only by
    // hovering for a tooltip.
    summary: soleSource
      ? `${confirmed} confirmed of ${count} from ${label}`
      : `${confirmed} confirmed · ${count} from ${label}`,
    detail: `${confirmed} object${
      confirmed === 1 ? " is" : "s are"
    } confirmed on this segmentation; that is the number the analysis measures and the one to quote.${provenanceNote}${remainder}${staleNote}`,
    // Nothing confirmed yet is not a good state: the analysis would measure
    // nothing at all, however many candidates are on screen.
    tone: staleOverlay || confirmed === 0 ? "warning" : "good",
  };
}
