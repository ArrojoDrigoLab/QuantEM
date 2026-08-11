/**
 * A fine-tune run's progress, as one of the shared progress rows.
 *
 * The same row model the Tasks drawer and the labeling banner draw, so a run
 * that is visible in two places says the same thing in both. What is different
 * is what a unit *is*: a segmentation pass walks tiles, and this walks training
 * steps — and under cross-validation it walks rounds of them, which is why the
 * row carries a round clause the tile rows have no use for.
 *
 * Two things this must not do, both of which the owner asked for by name:
 *
 * * **never render a missing estimate as zero.** `eta_seconds` is null until
 *   the server has watched a round, or a tenth of the steps, actually finish.
 *   "0 seconds left" from a standing start is a lie the bar then spends four
 *   minutes contradicting; "estimating time left" is the truth.
 * * **divide by the server's number.** `percent` is computed server-side
 *   precisely so the bar and the words beside it cannot disagree, which they
 *   would the moment one of them tried to fold rounds into steps itself.
 */

import {
  formatTimeLeft,
  formatUnits,
  joinClauses,
} from "@/shared/progress/progressCopy";
import type { ProgressRow } from "@/shared/progress/progressRows";
import type { FineTuneProgress } from "@/shared/types/finetune";

/** What each fine-tuning phase is called on screen. */
export const FINE_TUNE_STAGE_PHRASES: Record<string, string> = {
  preparing: "preparing the training data",
  training: "training",
  evaluating: "checking the result",
  saving: "saving the model",
};

/** Shown in place of a time estimate the server cannot honestly give yet. */
export const ESTIMATING_TIME_LEFT = "estimating time left";

function stagePhrase(stage: string): string | null {
  return FINE_TUNE_STAGE_PHRASES[stage] ?? null;
}

function roundsClause(progress: FineTuneProgress): string | null {
  // One round is one training-plus-evaluation pass, so a run with one of them
  // has no rounds to speak of -- saying "round 1 of 1" would invent a
  // dimension the run does not have.
  if (!progress.total_rounds || progress.total_rounds <= 1) return null;
  return `round ${progress.round} of ${progress.total_rounds}`;
}

function stepsClause(progress: FineTuneProgress): string | null {
  if (!progress.total_steps) return null;
  return formatUnits(progress.step, progress.total_steps, "step");
}

function etaClause(progress: FineTuneProgress): string | null {
  if (progress.eta_seconds === null || progress.eta_seconds === undefined) {
    return ESTIMATING_TIME_LEFT;
  }
  return formatTimeLeft(progress.eta_seconds);
}

/**
 * One row for one fine-tune run.
 *
 * `name` is the fine-tune's own name, because that is what the user typed and
 * what the model will be called afterwards.
 */
export function fineTuneProgressRow(
  progress: FineTuneProgress,
  name: string
): ProgressRow {
  const status = progress.status;
  let percent: number | null = null;
  let showPercentText = false;
  let glyph = status === "RUNNING" ? "●" : "○";
  let tone: "normal" | "warning" = "normal";
  let detail: string;

  if (status === "FAILED") {
    glyph = "■";
    tone = "warning";
    percent = progress.percent;
    detail = joinClauses([
      stepsClause(progress) ? `stopped at ${stepsClause(progress)}` : null,
      "this one did not finish",
    ]);
  } else if (status === "SUCCESS") {
    percent = 100;
    detail = joinClauses([stepsClause(progress), "finished"]);
  } else if (status === "PENDING") {
    detail = joinClauses([
      "waiting to start",
      progress.total_steps
        ? formatUnits(0, progress.total_steps, "step")
        : null,
    ]);
  } else {
    percent = progress.percent;
    showPercentText = progress.percent !== null;
    detail = joinClauses([
      stagePhrase(progress.stage),
      roundsClause(progress),
      stepsClause(progress),
      etaClause(progress),
    ]);
  }

  if (!detail) detail = "starting";

  return {
    key: `finetune:${name}`,
    kind: "organelle",
    glyph,
    name,
    percent,
    showPercentText,
    detail,
    tone,
    ariaLabel: `${name}: ${
      percent !== null && showPercentText ? `${Math.round(percent)}%, ` : ""
    }${detail}`,
  };
}

/** The list form, for `RunProgressList`. Empty before the first poll lands. */
export function fineTuneProgressRows(
  progress: FineTuneProgress | null,
  name: string
): ProgressRow[] {
  return progress ? [fineTuneProgressRow(progress, name)] : [];
}
