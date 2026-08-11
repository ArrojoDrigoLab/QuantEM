/**
 * The include-level dial.
 *
 * Moving it re-derives the objects from the confidence map the run already
 * saved. No model runs, so it costs seconds rather than the tens of minutes a
 * full re-run costs, and the objects that come out are exactly the objects a
 * re-run at that level would have produced.
 *
 * **It is not a live scrub, and that is deliberate.** The slider is local state
 * until the user presses the button. One request is one job, one job rewrites
 * every candidate on the image and renumbers the result; firing one per pixel
 * of travel would queue a hundred of them for one gesture, each invalidating
 * the last, and the queue is one slot wide. The value under the thumb updates
 * as you drag so the gesture still feels connected to something.
 *
 * Three states, and the middle one is the one that usually gets skipped:
 *
 * * **can move** -- slider, what it costs, what it preserves, one button;
 * * **cannot move** -- the control is disabled and the server's own sentence
 *   says why. There are two such sentences and they are different futures: a
 *   map that was never saved is fixed for good by running once, and a saved
 *   result from an older build will go on being refused until it is replaced.
 *   Neither is replaced with a generic "try running again";
 * * **working** -- the job is queued or running, and the button cannot be
 *   pressed again.
 */

import { useCallback, useEffect, useMemo, useState } from "react";

import "./IncludeLevelDial.css";
import { useJobProgress } from "@/shared/hooks/useJobProgress";
import { extractApiErrorMessage } from "@/utils/apiErrors";

import { getIncludeLevel, setIncludeLevel } from "./api";
import type { IncludeLevelState } from "./api";
import {
  DIAL_BLOCKED_ACTION,
  DIAL_COST,
  DIAL_EXPLANATION,
  DIAL_FAILED_FALLBACK,
  DIAL_PRESERVATION,
  DIAL_TITLE,
  DIAL_WORKING,
  LEVEL_NOT_SET,
  dialDone,
  formatIncludeLevel,
  levelsDiffer,
} from "./copy";

/**
 * Hundredths. The stored map is 8-bit, so 255 levels is all the resolution
 * there is; a hundredth is finer than any of them is worth showing.
 */
const STEP = 0.01;

export interface IncludeLevelDialProps {
  segmentationId: string;
  /** Which model's stored result to re-read. Omit for the organelle default. */
  sourceModel?: string | null;
  /**
   * Called once, after a re-extract has finished, so the screen can reload the
   * objects and the overlay. The dial does not know how to draw anything.
   */
  onReextracted?: () => void;
}

export function IncludeLevelDial({
  segmentationId,
  sourceModel = null,
  onReextracted,
}: IncludeLevelDialProps) {
  const [state, setState] = useState<IncludeLevelState | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [draft, setDraft] = useState<number | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const load = useCallback(async () => {
    try {
      const next = await getIncludeLevel(segmentationId, sourceModel);
      setState(next);
      setLoadError(null);
      return next;
    } catch (error) {
      setLoadError(
        extractApiErrorMessage(error, "The include level could not be read.")
      );
      return null;
    }
  }, [segmentationId, sourceModel]);

  useEffect(() => {
    setDraft(null);
    setJobId(null);
    setSubmitError(null);
    void load();
  }, [load]);

  const { job } = useJobProgress(jobId);
  const jobStatus = job?.status;

  useEffect(() => {
    if (!jobStatus) return;
    if (jobStatus === "PENDING" || jobStatus === "RUNNING") return;
    // Terminal. Re-read before telling the screen anything: on success the
    // level and the count have both changed, and on failure the dial has to go
    // back to showing where it actually is rather than where it was asked to be.
    setJobId(null);
    if (jobStatus === "FAILED") {
      setSubmitError(job?.message?.trim() || DIAL_FAILED_FALLBACK);
    }
    void load().then((next) => {
      if (next) setDraft(null);
      if (jobStatus === "SUCCESS") onReextracted?.();
    });
  }, [jobStatus, job?.message, load, onReextracted]);

  /**
   * Where the thumb sits. The draft while the user is holding it, then the
   * recorded level, then the model's own -- never an invented 0.5, which would
   * put the thumb somewhere the objects were not found.
   */
  const position = useMemo(() => {
    if (draft !== null) return draft;
    if (state?.include_level !== null && state?.include_level !== undefined) {
      return state.include_level;
    }
    if (
      state?.default_include_level !== null &&
      state?.default_include_level !== undefined
    ) {
      return state.default_include_level;
    }
    return 0.5;
  }, [draft, state]);

  const working = submitting || jobId !== null;
  const settled = state?.include_level ?? state?.default_include_level ?? null;
  // `draft === null` is "the user has not touched it", which is not a move.
  // Asking `levelsDiffer` alone answered *true* there -- a null draft against a
  // set level reads as a change -- so the button was live before anything had
  // been dragged, offering to re-extract at the level the objects were already
  // found at.
  const moved = draft !== null && levelsDiffer(draft, settled);

  const apply = useCallback(async () => {
    if (draft === null || working) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const queued = await setIncludeLevel(segmentationId, draft, sourceModel);
      setJobId(queued.job_id);
    } catch (error) {
      setSubmitError(extractApiErrorMessage(error, DIAL_FAILED_FALLBACK));
    } finally {
      setSubmitting(false);
    }
  }, [draft, segmentationId, sourceModel, working]);

  if (loadError) {
    return (
      <div className="include-level-dial" data-testid="include-level-dial">
        <h4 className="include-level-title">{DIAL_TITLE}</h4>
        <p className="include-level-problem" role="alert">
          {loadError}
        </p>
      </div>
    );
  }

  if (!state) {
    return (
      <div className="include-level-dial" data-testid="include-level-dial">
        <h4 className="include-level-title">{DIAL_TITLE}</h4>
        <p className="include-level-note">Checking the saved result…</p>
      </div>
    );
  }

  const blocked = !state.can_move;

  return (
    <div className="include-level-dial" data-testid="include-level-dial">
      <h4 className="include-level-title">{DIAL_TITLE}</h4>
      <p className="include-level-note">{DIAL_EXPLANATION}</p>

      {blocked ? (
        // The server's own sentence, never a generic one: the two reasons the
        // dial can be blocked have different futures, and only this text says
        // which of them the user is in.
        <p className="include-level-blocked" data-testid="include-level-blocked">
          {state.detail} {DIAL_BLOCKED_ACTION}
        </p>
      ) : null}

      <label className="include-level-slider" htmlFor="include-level-input">
        <span className="include-level-slider-label">
          Level
          {/* No `htmlFor` here. An `<output for=...>` is announced as a label
              for the input it names, so this readout became a second label
              beside the real one and the slider was introduced as "Level 0.50"
              -- the value read twice, once as its name. It is a derived
              display; `aria-live` is what makes it useful while dragging. */}
          <output
            className="include-level-value"
            data-testid="include-level-value"
            aria-live="polite"
          >
            {formatIncludeLevel(position)}
          </output>
        </span>
        <input
          id="include-level-input"
          type="range"
          min={state.minimum}
          max={state.maximum}
          step={STEP}
          value={position}
          disabled={blocked || working}
          aria-describedby="include-level-cost"
          onChange={(event) => setDraft(Number(event.target.value))}
        />
      </label>

      <p className="include-level-status" data-testid="include-level-status">
        {state.include_level === null
          ? LEVEL_NOT_SET
          : dialDone(state.object_count)}
      </p>

      <p className="include-level-note" id="include-level-cost">
        {DIAL_COST}
      </p>
      <p className="include-level-preserved">{DIAL_PRESERVATION}</p>

      {submitError ? (
        <p className="include-level-problem" role="alert">
          {submitError}
        </p>
      ) : null}

      <button
        type="button"
        className="include-level-apply"
        data-testid="include-level-apply"
        disabled={blocked || working || !moved}
        onClick={() => {
          void apply();
        }}
      >
        {working
          ? DIAL_WORKING
          : `Find objects at ${formatIncludeLevel(position)}`}
      </button>
    </div>
  );
}
