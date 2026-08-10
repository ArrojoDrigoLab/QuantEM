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
}

export interface AdaptCropsResponse {
  crops: AdaptCrop[];
  split_mode: SplitMode;
  n_images: number;
  ready: boolean;
  /** Hard stops. A completed ROI is the one that cannot be worked around. */
  blockers: string[];
  warnings: string[];
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
  heldout_dice: number | null;
  /** True once the saved head was reloaded onto a fresh encoder and re-scored. */
  verified_reload: boolean;
  train_seconds: number | null;
  applied_at: string | null;
  created_at: string;
  error: string;
  caveats: string[];
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
  [key: string]: unknown;
}
