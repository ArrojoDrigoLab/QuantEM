/**
 * The three ways to spend the annotations, and the help that explains them.
 *
 * On the wire there are two fields — `mode` and `cv_benchmark` — and the owner
 * described them as a two-way radio with a checkbox hanging off the second
 * option. On screen they are three radios, because the pair admits a state that
 * means nothing: "use all" with cross-validation ticked. Three mutually
 * exclusive choices cannot express it, and {@link modeChoicePayload} is the
 * only place the mapping back to the two fields lives.
 */

import type { FineTuneMode } from "@/shared/types/finetune";

export type FineTuneModeChoice = "use_all" | "holdout_1" | "holdout_1_cv";

/**
 * The tile count at or below which the server defaults to using everything.
 *
 * The owner's numbers: *use all* at three tiles or fewer, *hold out one* above
 * four. Four itself was unstated; the round-3 contract resolves it as hold-out,
 * so the boundary is "more than three". This constant exists to *describe* the
 * server's rule in the help text, never to re-decide it — `default_mode` comes
 * down in the preview response and is honoured as sent.
 */
export const USE_ALL_TILE_CEILING = 3;

export interface FineTuneModeOption {
  value: FineTuneModeChoice;
  label: string;
  /** One line under the label. The long form is in {@link TRAINING_MODE_HELP}. */
  summary: string;
}

export const FINE_TUNE_MODE_OPTIONS: FineTuneModeOption[] = [
  {
    value: "use_all",
    label: "Use all",
    summary:
      "Train on every annotated area. Nothing is held back, so there is no score to report.",
  },
  {
    value: "holdout_1",
    label: "Hold out one",
    summary:
      "Keep one annotation back and score the trained model on it. One honest number.",
  },
  {
    value: "holdout_1_cv",
    label: "Hold out one, with cross-validation benchmarking",
    summary:
      "Repeat the hold-out with each annotation held back in turn, and report the average and the per-image results. Slower, by roughly the number of annotations.",
  },
];

/** The two wire fields for a choice. */
export function modeChoicePayload(choice: FineTuneModeChoice): {
  mode: FineTuneMode;
  cv_benchmark: boolean;
} {
  if (choice === "use_all") return { mode: "use_all", cv_benchmark: false };
  return { mode: "holdout_1", cv_benchmark: choice === "holdout_1_cv" };
}

/** The choice the server's `default_mode` stands for. */
export function modeChoiceFromDefault(mode: FineTuneMode): FineTuneModeChoice {
  return mode === "use_all" ? "use_all" : "holdout_1";
}

export const TRAINING_MODE_HELP_TITLE = "How the annotations are used";

/**
 * What the "?" says. Paragraphs rather than one block so the panel can space
 * them, and plain sentences rather than a table so a screen reader gets it in
 * the order it is written.
 */
export const TRAINING_MODE_HELP: string[] = [
  "An annotated area larger than one training tile is cut into tiles, so the number of tiles is usually larger than the number of annotations. The choice below is about what happens to those tiles.",
  "Each training round runs for 20 steps per tile, with a minimum of 300 steps and a maximum of 600. Up to 15 training tiles therefore uses 300 steps; 16 tiles uses 320; and 30 or more uses 600.",
  "Use all — every tile is trained on. This gives the model the most to learn from, and leaves nothing to measure it against, so the run reports no score.",
  "Hold out one — one annotation is kept out of training and used to score the result. The score is honest but it rests on a single held-out annotation, so treat it as a rough reading.",
  "Hold out one, with cross-validation benchmarking — the hold-out is repeated with each annotation held back in turn. You get an average score and a result for each image, which is the only way to see that one image is dragging the average. It takes roughly as many training rounds as there are annotations.",
  `Which one is preselected is decided by how many tiles the selection comes to: use all at ${USE_ALL_TILE_CEILING} tiles or fewer, hold out one above that. Below that many tiles there is not enough to both train on and measure with. Change it freely — the preselection is a starting point, not a restriction.`,
];
