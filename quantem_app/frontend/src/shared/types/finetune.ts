/**
 * Model catalogue and guided fine-tuning, per `API_CONTRACT.md`.
 *
 * The one invariant that shapes this file: a Dice never travels alone. Every
 * structure that carries a held-out score also carries the `split_mode` that
 * says what it measured, and the crop names it was fit on. The types make it
 * awkward to render one without the other.
 */

export type SplitMode = "image-disjoint" | "within-image" | "no-heldout";

export type AdapterStatus = "PENDING" | "RUNNING" | "SUCCESS" | "FAILED";

/** `threshold_only` is CPU-cheap and always offered; `head` needs torch. */
export type AdaptMode = "threshold_only" | "head";

export type ModelFamily = "quantem" | "omniem";

export type OrganelleKey = "mito" | "er" | "nucleus" | "ld";

/**
 * How a pack's encoder would be built, when it can be built at all.
 *
 * `"exported"` is the shipping path (a self-describing `encoder_ts.pt`, no
 * architecture package needed); `"timm"` and `"dinov3"` rebuild it eagerly from
 * the named package. Null when the pack cannot run.
 */
export type EncoderTier = "exported" | "timm" | "dinov3" | string;

/**
 * The install job already working on a pack, reported by `GET /api/models/`.
 *
 * Why the catalogue says this at all: installer-requested downloads run at
 * first launch, and while all four were RUNNING every Models card still said
 * "not installed" with a live Download button — clicking it queued a real
 * duplicate gigabyte download. `job_id` is the row to poll through the jobs
 * API; the byte counts are a snapshot for rendering before the first poll
 * answers (both null while the job is still queued).
 */
export interface ModelPackActiveInstall {
  job_id: number | string;
  status: "QUEUED" | "RUNNING";
  progress_current_bytes: number | null;
  progress_total_bytes: number | null;
}

export interface ModelPack {
  id: string;
  family: ModelFamily;
  organelle: OrganelleKey;
  title: string;
  installed: boolean;
  download_bytes: number;
  /**
   * The QUEUED/RUNNING install job for this pack, if one exists. Null when no
   * install is underway; absent on an older backend. A card must render this
   * as an installing state instead of a Download button — the POST now
   * answers 409 while it is set.
   */
  active_install?: ModelPackActiveInstall | null;
  /** Nanometres per pixel the pack runs at; null = run at native resolution. */
  canonical_nm: number | null;
  tile_size: number;
  default_threshold: number;
  decoder: string;
  neck: string;
  adapt: string;
  licence: string;
  notes: string;
  /**
   * Whether `engine.load_model` would actually succeed on this machine.
   *
   * Not the same question as `installed`: installing only verifies files, and
   * whether those files can be turned into a module is decided separately. The
   * four QuantEM packs sit on a DINOv3 ViT-B whose architecture code is not
   * redistributed, so they install perfectly and then raise seconds into a run.
   * `probe_runnable` answers this up front precisely so the picker can grey the
   * pack out — never offer a pack for selection without consulting it.
   *
   * Optional because an older backend omits it; treat `undefined` as "unknown",
   * which is a different thing from `false`.
   */
  runnable?: boolean;
  /** Why `runnable` is false, as a sentence to show the user. Null when it is true. */
  reason?: string | null;
  encoder_tier?: EncoderTier | null;
}

export interface AdaptedModelEntry {
  id: string;
  base: string;
  name: string;
  created_at: string;
  calibrated_threshold: number | null;
  heldout_dice: number | null;
  /** Never render `heldout_dice` without this. */
  split_mode: SplitMode;
  mode?: AdaptMode;
  segmentation_id?: string | null;
  applied_at?: string | null;
}

export interface DeviceInfo {
  kind: string;
  name: string;
  cuda: boolean;
  mps: boolean;
}

export interface ModelCatalogue {
  packs: ModelPack[];
  adapted: AdaptedModelEntry[];
  device: DeviceInfo | null;
}

/** One completed-ROI crop the adaptation can learn from or be scored on. */
export interface AdaptCrop {
  id: string;
  name: string;
  /** Asset id. Crops sharing one image can never form an image-disjoint split. */
  image_key: string;
  segmentation_id?: string;
  width: number;
  height: number;
  n_objects: number;
  annotated_px: number;
  foreground_px?: number;
  pixel_size_nm?: number | null;
  /** False when no probability map covers this crop: threshold-only cannot use it. */
  has_probability?: boolean;
  /**
   * The image this checked area is on, by the name the user gave it.
   *
   * `image_key` is the asset uuid and `name` is derived from it (`4f3a91c2_0`),
   * so neither can be shown to a reader. Crops are pooled across every image
   * with the same organelle segmented, so naming them is the difference between
   * "I'll learn from 3 areas" and a sentence the user can check.
   */
  image_name?: string;
  /** True when this checked area is on the image currently open. */
  is_this_image?: boolean;
}

/**
 * Whether the chosen pack can cut a training window out of what has been checked.
 *
 * Pack-specific, because the rule is the pack's own tile size against its
 * canonical resolution — so the crops endpoint only answers it when told which
 * pack (`?base_model=`). `ok: false` is a certainty, not a guess: below this
 * size head training produces zero training steps.
 */
export interface HeadSizePreflight {
  base_model: string;
  ok: boolean;
  /** Span of the largest checked area, in nanometres. Null when unmeasurable. */
  largest_nm: number | null;
  /** Span head training needs, in nanometres, for that same area. */
  required_nm: number | null;
  largest_px: number;
  required_px: number;
  /** How many checked areas were measured. */
  n_areas: number;
  /** The sentence to show. Null when the geometry is fine. */
  reason: string | null;
}

export interface AdaptCropsResponse {
  crops: AdaptCrop[];
  split_mode: SplitMode;
  n_images: number;
  ready: boolean;
  /** Hard stops. A completed ROI is the one that cannot be worked around. */
  blockers: string[];
  warnings: string[];
  /** True when at least one checked area has a stored probability map. */
  has_probability?: boolean;
  /**
   * Reasons that stop **one** rung rather than everything.
   *
   * `blockers` is what stops the whole page; this is what greys a single
   * button. Matching my cut-off to your marks needs a stored probability map to
   * sweep; head training computes its own but needs a physically larger checked
   * area. Reporting them apart is what keeps one rung reachable when the other
   * is not — and it is what lets the panel refuse *before* anything is queued.
   */
  mode_blockers?: Partial<Record<AdaptMode, string[]>>;
  /** The size verdict behind `mode_blockers.head`; null when no pack was named. */
  head_size?: HeadSizePreflight | null;
  /** Known before the run so the UI can badge the fitted crops up front. */
  train_crop_names: string[];
  heldout_crop_names: string[];
  /** Rungs this machine can actually offer; `head` is absent without torch. */
  modes: AdaptMode[];
}

export interface AdaptStartPayload {
  base_model: string;
  mode: AdaptMode;
  steps?: number;
  lr?: number;
  seed?: number;
  name?: string;
  /**
   * Use the result the moment it exists, rather than fitting it and leaving the
   * user to find a separate Apply.
   *
   * Opt-in per request: a run started to *look* at the numbers must not
   * silently become the model. It stamps the adapter and nothing else — no
   * object is written, moved or deleted by applying.
   */
  apply_and_rerun?: boolean;
}

export interface AdaptStartResponse {
  job_id: string;
  adapter_id: string;
}

/**
 * The threshold sweep. `train_dice` is the curve; everything else is a point on
 * it or a score at a point on it.
 */
export interface AdapterSweep {
  thresholds: number[];
  train_dice: Array<number | null>;
  calibrated_threshold: number;
  train_dice_at_calibrated: number | null;
  train_dice_at_default: number | null;
  heldout_dice_at_calibrated: number | null;
  heldout_dice_at_default: number | null;
  /**
   * Best achievable with a per-crop threshold chosen using the answers.
   * A ceiling, never a target — render it as such (honesty rule 3).
   */
  heldout_oracle: number | null;
  improvement: number | null;
  per_crop: Record<string, number | null>;
  train_crop_names: string[];
  heldout_crop_names: string[];
}

/**
 * What running the model again would do, and what it would not.
 *
 * Returned by the apply endpoint because the two are one decision from the
 * user's side. `preservation` is a description of the extraction code, not a
 * reassurance: a re-run deletes only the model's own previous guesses and drops
 * any new guess that lands on an object the user kept or removed.
 */
export interface RerunAdvice {
  include_level: number | null;
  previous_include_level: number | null;
  changes_objects: boolean;
  preserves_manual_work: boolean;
  summary: string;
  preservation: string;
}

export interface Adapter {
  id: string;
  base_model: string;
  name: string;
  status: AdapterStatus;
  mode: AdaptMode;
  steps: number;
  trainable_params: number | null;
  segmentation_id: string | null;
  split_mode: SplitMode;
  train_crop_names: string[];
  heldout_crop_names: string[];
  sweep: AdapterSweep | Record<string, never>;
  calibrated_threshold: number | null;
  /**
   * The pack's published cut-off, so a new one can be reported as a change.
   *
   * Without it the panel can print "0.45" and not "was 0.50", and a bare
   * number says nothing about which way the model moved. Optional because an
   * older backend omits it.
   */
  default_threshold?: number | null;
  heldout_dice: number | null;
  /** True once the saved head was reloaded onto a fresh encoder and re-scored. */
  verified_reload: boolean;
  train_seconds: number | null;
  applied_at: string | null;
  created_at: string;
  error: string;
  caveats: string[];
  /** Present only on the apply response, which is where it is computed. */
  rerun_advice?: RerunAdvice;
}

/**
 * The most recent run for a segmentation, from the server.
 *
 * This is how the panel reattaches after a reload. It replaces a `localStorage`
 * pointer that was invisible on a second machine, invisible after a cleared
 * browser store, and — because it was written once and never cleared on
 * success — the reason a second run could not be started at all.
 */
export interface AdaptLatestResponse {
  adapter: Adapter | null;
  /** The queue row behind it, when the queue still has one. */
  job_id: string | null;
}

// ---------------------------------------------------------------------------
// Fine-tuning over a scope of images (owner ruling R13, round-3 contract §4).
//
// The types above describe the older single-segmentation "Improve" flow, which
// still ships and still uses them. These describe the flow that trains one
// organelle across a *selection* of datasets and images: the scope tree the
// dialog renders, the preview it totals, the run it starts, and the run's
// progress, result and opt-in application.
//
// Nothing here is derived client-side that the server also computes. The count,
// the default mode and the eligibility verdict all arrive decided, precisely so
// the dialog and the endpoint that refuses a bad request cannot drift apart.
// ---------------------------------------------------------------------------

/** An experiment named just enough to put on screen. */
export interface FineTuneExperimentRef {
  id: string;
  name: string;
}

/**
 * One image in the scope tree, with what it contributes.
 *
 * `annotation_count` is `confirmed_areas + done_rois`: the plain number of
 * annotated regions, **not** the number of training tiles one is cut into. The
 * owner's example is a dataset where two images have three annotations each and
 * a third has one, and the dialog shows 7.
 */
export interface FineTuneScopeImage {
  id: string;
  name: string;
  confirmed_areas: number;
  done_rois: number;
  annotation_count: number;
}

export interface FineTuneScopeDataset {
  id: string;
  name: string;
  /** Every image in the dataset, annotated or not. */
  image_count: number;
  annotated_image_count: number;
  /** The dataset's own total. Authoritative when the whole dataset is picked. */
  annotation_count: number;
  images: FineTuneScopeImage[];
}

export interface FineTuneScopeExperiment {
  id: string;
  name: string;
  datasets: FineTuneScopeDataset[];
  /** In this experiment, in no dataset. */
  ungrouped_images: FineTuneScopeImage[];
}

/**
 * `GET /api/finetune/scope/` — the whole tree in one call.
 *
 * Every active image appears under an experiment; ``ungrouped_images`` means
 * only that it has no dataset inside that experiment.
 */
export interface FineTuneScopeResponse {
  experiments: FineTuneScopeExperiment[];
}

/** What a selection is on the wire: datasets expand, then union with assets. */
export interface FineTuneScopeSelectionPayload {
  segmentation_type: string;
  asset_ids: string[];
  dataset_ids: string[];
  /** The pack whose tile size, install state, and training mode are previewed. */
  base_model?: string;
}

export interface FineTunePreviewImage {
  asset_id: string;
  name: string;
  confirmed_areas: number;
  done_rois: number;
  tiles: number;
}

/**
 * How the training data is split.
 *
 * Two values on the wire; three choices on screen, because hold-out with
 * cross-validation benchmarking is a different decision from hold-out without
 * it and pairing a radio with a checkbox made it possible to tick the checkbox
 * under "use all", where it means nothing.
 */
export type FineTuneMode = "use_all" | "holdout_1";

/**
 * `POST /api/finetune/preview/` — everything the dialog needs before it can
 * offer the button.
 *
 * `eligible` and `blockers` are the same verdict `POST /runs/` enforces with a
 * 400, sent ahead of time so a user never reaches one.
 */
export interface FineTunePreviewResponse {
  experiment: FineTuneExperimentRef | null;
  /** The exact pack used for install checks, tile counts, and the mode default. */
  base_model: string;
  asset_count: number;
  annotation_count: number;
  confirmed_areas: number;
  done_rois: number;
  tile_count: number;
  per_image: FineTunePreviewImage[];
  /** Chosen by the server from `tile_count`; the dialog honours it, never re-derives it. */
  default_mode: FineTuneMode;
  eligible: boolean;
  blockers: string[];
}

export interface FineTuneRunPayload {
  name: string;
  /** Non-null replaces that fine-tune in place; null is a new one, and a name collision is a 409. */
  overwrite_adapter_id: string | null;
  segmentation_type: string;
  base_model: string;
  asset_ids: string[];
  dataset_ids: string[];
  mode: FineTuneMode;
  cv_benchmark: boolean;
}

export interface FineTuneRunResponse {
  adapter_id: string;
  job_id: string;
}

export type FineTuneStage = "preparing" | "training" | "evaluating" | "saving";

export type FineTuneRunStatus = "PENDING" | "RUNNING" | "SUCCESS" | "FAILED";

/**
 * `GET /api/finetune/runs/<id>/progress/`.
 *
 * `percent` is computed server-side so the bar and the words beside it divide
 * by the same thing. `eta_seconds` is null until a round, or a tenth of the
 * steps, has actually finished — an absent estimate is rendered as "estimating"
 * and never as zero.
 */
export interface FineTuneProgress {
  status: FineTuneRunStatus;
  stage: FineTuneStage | string;
  step: number;
  total_steps: number;
  round: number;
  total_rounds: number;
  percent: number | null;
  eta_seconds: number | null;
  message: string;
  error: string;
}

export interface FineTuneCvFold {
  fold: number;
  held_out_asset_id: string;
  dice: number | null;
  iou: number | null;
  n_tiles: number;
}

export interface FineTuneCvPerImage {
  asset_id: string;
  name: string;
  dice: number | null;
  iou: number | null;
  n_tiles: number;
}

/** Per-image results are required, not optional: a mean alone hides the outlier. */
export interface FineTuneCvResults {
  folds: FineTuneCvFold[];
  mean: { dice: number | null; iou: number | null };
  per_image: FineTuneCvPerImage[];
}

/** `GET /api/finetune/runs/<id>/` — the adapter row, plus what CV measured. */
export type FineTuneRunDetail = Adapter & {
  cv_results?: FineTuneCvResults | Record<string, never>;
  training_mode?: FineTuneMode;
  cv_benchmark?: boolean;
  experiment?: FineTuneExperimentRef | null;
  asset_ids?: string[];
  dataset_ids?: string[];
};

/**
 * A row in the overwrite dropdown.
 *
 * `experiment` is typed loosely because §4.7 of the contract names the field
 * without giving its shape; {@link fineTuneExperimentName} reads either form.
 */
export interface FineTuneAdapterSummary {
  id: string;
  name: string;
  base_model: string;
  status: AdapterStatus;
  created_at: string;
  experiment: FineTuneExperimentRef | string | null;
  asset_count: number;
}

export interface FineTuneApplyQueued {
  asset_id: string;
  segmentation_id: string;
  job_id: string;
}

/** `POST /api/finetune/runs/<id>/apply/` — never automatic; the user clicks. */
export interface FineTuneApplyResponse {
  batch_id: string;
  adapter_id: string;
  dataset_ids: string[];
  queued: FineTuneApplyQueued[];
}

export interface FineTuneApplyImageProgress extends FineTuneApplyQueued {
  asset_name: string;
  status: FineTuneRunStatus | "CANCELLED" | "RETRY";
  stage: string;
  progress: number;
  units_done: number | null;
  units_total: number | null;
  message: string;
  failure: string;
  adapter_id: string;
  result: Record<string, unknown> | null;
}

/** Per-image state for one opt-in Dataset/image application batch. */
export interface FineTuneApplyProgress {
  batch_id: string | null;
  adapter_id: string;
  total: number;
  complete: number;
  succeeded: number;
  failed: number;
  images: FineTuneApplyImageProgress[];
}

/** The experiment's name, whichever of the two shapes §4.7 turns out to send. */
export function fineTuneExperimentName(
  value: FineTuneExperimentRef | string | null | undefined
): string | null {
  if (!value) return null;
  return typeof value === "string" ? value : value.name;
}

/**
 * The extra fields the job result carries that the adapter row does not:
 * the *base* model's sweep, for the before/after comparison, and the
 * training telemetry.
 */
export interface AdapterJobResult {
  id?: string;
  base_model?: string;
  mode?: AdaptMode;
  split_mode?: SplitMode;
  sweep?: AdapterSweep;
  base_sweep?: AdapterSweep;
  training?: Record<string, unknown>;
  tile?: number;
  trainable_params?: number;
  train_seconds?: number;
  verified_reload?: boolean;
  reloaded_heldout_dice?: number | null;
  warnings?: string[];
  caveats?: string[];
  /**
   * Whether the run put its own result to work, and what is still outstanding.
   *
   * `rerun_pending` is the honest half: applying costs nothing and changes
   * nothing on screen, so the objects the user is looking at are still the ones
   * found at the old include level until the model is run again.
   */
  apply_and_rerun?: {
    requested: boolean;
    applied: boolean;
    applied_at: string | null;
    include_level: number | null;
    previous_include_level: number | null;
    changes_objects: boolean;
    rerun_pending: boolean;
    preserves_manual_work: boolean;
    preservation: string;
  };
  [key: string]: unknown;
}
