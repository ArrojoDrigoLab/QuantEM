import type {
  CellStatus,
  CellStatusLabel,
  LabelState,
  BBox,
  RefinementStatus,
  SegmentStatus,
  SegmentStatusLabel,
} from "@/shared/types/common";
import type {
  ImageSegmentation,
  ProbabilityMap,
  SegmentationInstanceParams,
  StatusStage,
} from "@/shared/types/images";

export interface SegmentationRoi {
  id: string;
  segmentation: string;
  x: number;
  y: number;
  width: number;
  height: number;
  source: "AUTO" | "MANUAL" | "DEFAULT";
  seed?: number | null;
  is_active: boolean;
  is_complete: boolean;
  /**
   * Per-organelle completion of this ROI for the segmentation it was listed
   * under ("mark ROI as done" for this specific organelle). `null` when no
   * segmentation context applies.
   */
  completed_for_segmentation?: boolean | null;
  created_at: string;
  updated_at: string;
}

export interface CandidateCleanupSummary {
  deleted: number;
  updated: number;
  created: number;
}

export interface CompletedRoi {
  id: string;
  segmentation: string;
  polygon_coords: Array<[number, number]>;
  /** Interior rings (excluded holes) of the confirmed-area polygon, if any. */
  holes?: Array<Array<[number, number]>>;
  bbox: {
    x0: number;
    y0: number;
    x1: number;
    y1: number;
  } | null;
  created_at: string;
  updated_at: string;
}

export type CompletedRoiMode = "include" | "exclude";

export interface SubtractCompletedRoiResponse {
  updated: number;
  deleted: number;
  created: number;
  completed_rois: CompletedRoi[];
}

export interface SegmentObject {
  id: string;
  segmentation: string;
  label_state: LabelState;
  refined?: RefinementStatus;
  status?: SegmentStatus | CellStatus | null;
  status_label?: SegmentStatusLabel | CellStatusLabel | null;
  source_model?: string;
  scope?: "ROI" | "FULL";
  confidence_score: number | null;
  geometry_coords: Array<[number, number]>;
  geometry?:
    | {
        type: "Polygon";
        coordinates: Array<Array<[number, number]>>;
      }
    | {
        type: "MultiPolygon";
        coordinates: Array<Array<Array<[number, number]>>>;
      }
    | null;
  smoothed_geometry_coords?: Array<[number, number]>;
  created_at: string;
  updated_at: string;
}

export interface SegmentationOverlayMutationState {
  desired_revision: number;
  applied_revision: number;
  lut_revision?: number;
  bundle_version?: number;
  sync_applied: boolean;
  rebuild_mode: "sync_partial" | "async_partial" | "async_full" | "metadata_only";
  source_model?: string | null;
  /** Desired revision of the separate source-less confirmed-display bundle. */
  confirmed_display_desired_revision?: number;
}

export interface RoiCompletionResponse extends SegmentationRoi {
  candidate_cleanup: CandidateCleanupSummary;
  overlay?: SegmentationOverlayMutationState | null;
}

/**
 * ID-map overlay manifest. The overlay is an integer label raster (`labels`)
 * plus a border mask (`border`); object colour/state is delivered out-of-band
 * via `lut_url` and applied at render time, so state changes bump `lut_revision`
 * without changing `bundle_version`/`applied_revision` (the raster identity).
 */
export interface SegmentationOverlayManifest {
  status: "MISSING" | "READY" | "DIRTY" | "BUILDING" | "FAILED";
  overlay_kind?: "object_ids" | "binary_mask";
  pickable?: boolean;
  /**
   * Why the last overlay build failed, verbatim from the server.
   *
   * `build_overlay_manifest` has always put this on the payload -- empty when
   * there is nothing wrong -- and the client threw it away. A `FAILED` status
   * is terminal: nothing re-queues the build, so if this string is not
   * rendered the user is left with a spinner that will never stop. It is the
   * only place the actual cause ("[WinError 183] Cannot create a file when
   * that file already exists: ...", a full disk, a locked file) is ever said.
   *
   * Optional on the type because a `FAILED` manifest from an older server, or
   * a build that died without recording anything, legitimately has no reason
   * to give -- and the renderer must say *that* rather than nothing.
   */
  last_error?: string;
  ngff_url: string | null;
  lut_url: string;
  arrays: string[];
  label_dtype: "uint32";
  source_model?: string | null;
  display_role?: "model" | "confirmed";
  data_ready?: boolean;
  update_job?: {
    id: string;
    status: "PENDING" | "RETRY" | "RUNNING" | "CANCEL_REQUESTED";
    progress: number;
    message: string;
    progress_units_done: number | null;
    progress_units_total: number | null;
    progress_unit_label: string;
  } | null;
  bundle_version: number;
  applied_revision: number;
  desired_revision: number;
  lut_revision: number;
  chunk_size: [number, number];
  level_count: number;
  width: number;
  height: number;
  overlay?: SegmentationOverlayMutationState;
}

/** One row of the JSON LUT variant (label -> object), used for picking. */
export interface OverlayLutObject {
  label: number;
  uuid: string;
  is_cell: boolean;
  state: string;
  color: string;
}

export interface OverlayLutJson {
  lut_revision: number;
  bundle_version: number;
  max_label: number;
  overlay_kind?: "object_ids" | "binary_mask";
  pickable?: boolean;
  color?: string;
  objects: OverlayLutObject[];
}

/** Parsed binary LUT: a flat RGBA8 palette indexed by dense label. */
export interface OverlayLutBinary {
  rgba: Uint8Array;
  maxLabel: number;
  lutRevision: number;
  bundleVersion: number;
}

export interface ProbabilityMapsResponse {
  segmentation: {
    id: string;
    status_stage: string;
    status_progress: number;
    status_error: string;
  };
  probability_maps: ProbabilityMap[];
}

export interface SegmentRegionQueryPayload {
  bbox?: BBox;
  polygon_coords?: Array<[number, number]>;
  states?: LabelState[];
  statuses?: CellStatus[];
  source_model?: string | null;
  include_geometry?: boolean;
}

export interface SegmentRegionQueryResult {
  id: string;
  label_state: LabelState;
  status?: CellStatus | SegmentStatus;
  status_label?: CellStatusLabel | SegmentStatusLabel;
  source_model?: string;
  confidence_score: number | null;
  geometry_coords?: Array<[number, number]>;
  smoothed_geometry_coords?: Array<[number, number]>;
  cell_type?: number;
  cell_type_name?: string;
}

export interface PointParams {
  x: number;
  y: number;
  states?: LabelState[];
  statuses?: CellStatus[];
  source_model?: string | null;
}

export interface SegmentLabelUpdatePayload {
  label_state: LabelState;
}

export interface BatchLabelUpdatePayload {
  labels: Array<{ id: string; label_state: LabelState }>;
  source_model?: string | null;
}

export interface BatchLabelUpdateResponse {
  created?: number;
  created_ids?: string[];
  updated: number;
  updated_ids?: string[];
  deleted?: number;
  deleted_ids?: string[];
  overlays?: Record<string, SegmentationOverlayMutationState>;
}

export type UserFeedbackInputType = "point" | "polygon";
export type UserFeedbackType = "CONFIRMED" | "REJECTED";
export type UserFeedbackUtilizedStatus =
  | "QUEUED"
  | "PROCESSING"
  | "FAILED"
  | "SUCCESS";

export interface UserFeedback {
  id: string;
  segmentation: string;
  input_type: UserFeedbackInputType;
  point: { x: number; y: number } | null;
  polygon_coords: Array<[number, number]> | null;
  feedback_type: UserFeedbackType;
  utilized_status: UserFeedbackUtilizedStatus;
  created_at: string;
  updated_at: string;
}

export interface UserFeedbackCreatePayload {
  input_type?: UserFeedbackInputType;
  point: { x: number; y: number };
  feedback_type: UserFeedbackType;
}

export interface UserFeedbackListParams {
  ids?: string[];
  utilized_statuses?: UserFeedbackUtilizedStatus[];
}

export interface ConfirmBatchSegmentPayload {
  geometry_coords?: Array<[number, number]>;
  /** First ring is the exterior; subsequent rings are holes. */
  geometry_rings?: Array<Array<[number, number]>>;
  operation?: "include" | "exclude";
  sam_score?: number | null;
}

export interface ConfirmBatchRequestPayload {
  segments: ConfirmBatchSegmentPayload[];
  merge_overlaps?: boolean;
  manual_creation?: boolean;
}

/** One outline of a batch, and what became of it. */
export interface ConfirmBatchOutlineOutcome {
  /** Index of the outline in the `segments` array as it was sent. */
  index: number;
  /** How many separate areas it turned out to enclose. */
  areas: number;
  /** How many of those were large enough to store as objects. */
  kept: number;
}

/**
 * What a batch did with an outline, when that is not what the gesture looked
 * like.
 *
 * A freehand stroke that crosses itself does not enclose one area: a
 * figure-of-eight encloses two, and a stroke that crosses twice can enclose
 * four. Every one of them is real — the user drew round it. The endpoint used
 * to keep the largest and drop the rest under a plain `200 {"created": 1}`
 * (measured: two 2500 px lobes stored 2500 px), which is why this block exists:
 * one gesture producing several objects is a surprise that belongs in the
 * response rather than in a `created` count the caller has no baseline for.
 *
 * `null` whenever every outline enclosed exactly one area and that area was
 * stored, which is the ordinary case.
 *
 * The two lists answer different questions and an outline can be in both:
 *
 * - `separated` — it enclosed more than one area. Everything drawn is stored,
 *   just as several objects. Worth saying; not a loss.
 * - `dropped` — `kept < areas`. Part or all of what was drawn is **not**
 *   stored, because a polygon spanning a pixel or less in either dimension is
 *   refused. `areas === 1, kept === 0` means the whole outline went nowhere,
 *   which is the same answer `POST .../segments/` gives as a 400 with a
 *   sentence. Never report a response carrying this as a plain success.
 */
export interface ConfirmBatchOutlinesNotice {
  separated: ConfirmBatchOutlineOutcome[];
  dropped?: ConfirmBatchOutlineOutcome[];
  /** The server's sentence, fit to show a user. */
  detail: string;
}

/**
 * Objects whose outline was written but whose measurements were not.
 *
 * `null` when everything measured. Present with a `207` status: the geometry
 * edit is committed and cannot be taken back, so it is not an error — but the
 * morphometric columns those objects reach `objects.csv` with are missing, and
 * that is not a plain success either.
 */
export interface SegmentMeasurementNotice {
  measured: number;
  unmeasured_ids: string[];
  detail: string;
}

export interface ConfirmBatchResponse {
  created: number;
  updated: number;
  deleted: number;
  confirmed_ids: string[];
  overlay?: SegmentationOverlayMutationState | null;
  outlines?: ConfirmBatchOutlinesNotice | null;
  measurement?: SegmentMeasurementNotice | null;
  candidate_cleanup?: CandidateCleanupSummary;
}

export interface RemoveAreaPolygonPayload {
  geometry_coords: Array<[number, number]>;
}

export interface RemoveAreaRequestPayload {
  areas: RemoveAreaPolygonPayload[];
}

export interface RemoveAreaResponse {
  created: number;
  updated: number;
  deleted: number;
  created_ids: string[];
  updated_ids: string[];
  deleted_ids: string[];
  overlay?: SegmentationOverlayMutationState | null;
  /**
   * Objects the cut reshaped but could not re-measure.
   *
   * The endpoint has always returned this (and a `207` with it), and this type
   * left it out, so no caller could read it: an erase that reshaped an object
   * and failed to re-measure it looked identical to one that worked, while the
   * object's stored `area` and `perimeter` still described the shape before
   * the cut.
   */
  measurement?: SegmentMeasurementNotice | null;
}

/** Correction tools shipped by QuantEM (SAM prompting is not shipped). */
export type CorrectionTool =
  | "draw"
  | "erase"
  | "add"
  | "polygon"
  | "completed_roi"
  | "sam";

export interface CorrectionModeState {
  reviewPhase: "model" | "correction";
  correctionTool: CorrectionTool;
}

export interface SegmentationConfig {
  id: string;
  segmentation: string;
  supports_instance_params?: boolean;
  instance_params?: SegmentationInstanceParams | null;
  probability_maps: ProbabilityMap[];
  created_at: string;
  updated_at: string;
}

/** Mirrors `SegmentationConfigResponseSerializer` -- no MitoNet fields exist. */
export interface SegmentationConfigResponse {
  supports_instance_params: boolean;
  instance_params: SegmentationInstanceParams | null;
}

export interface ClearSegmentationManualLabelsResponse {
  deleted: number;
  overlay?: SegmentationOverlayMutationState | null;
}

export interface ConfirmSegmentResponse extends SegmentObject {
  overlay?: SegmentationOverlayMutationState;
}

export interface RerunSegmentationRoiResponse {
  job_id: string;
  roi_id: string;
}

export interface RunFullSegmentationResponse {
  job_id: string;
}

export type QuerySegmentsInRegionResponse = {
  segments: SegmentRegionQueryResult[];
};

export type CreateUserFeedbackResponse = UserFeedback & { job_id?: string };

export type ActivateSegmentationRoiResponse = SegmentationRoi;

export type MarkRoiCompleteResponse = RoiCompletionResponse;

/**
 * `GET /api/segmentations/<id>/complete` — what marking done would destroy.
 *
 * Read-only, and the only trustworthy source for the number a confirmation
 * dialog puts in front of a user: the segmentation payload's `segment_counts`
 * can be a poll behind, and the POST refuses an `acknowledged_discard_count`
 * that does not match this exactly.
 */
export interface SegmentationCompletionPreview {
  segmentation_id: string;
  status_stage: StatusStage;
  is_complete: boolean;
  /** Objects that survive: the ones a human confirmed. */
  confirmed_count: number;
  /** Objects a discard would delete. */
  discard_count: number;
  discard_by_label_state: Record<string, number>;
  discard_by_source_model: Record<string, number>;
  /**
   * Whether a discard of this size could be archived, and so undone.
   * A prediction, not a promise — the snapshot is size-capped too, and the
   * POST response says what actually happened.
   */
  restorable: boolean;
  archive_max_objects: number;
  /** Objects currently sitting in the archive from a previous completion. */
  restorable_count: number;
}

/** What a completion actually did. */
export interface SegmentationCompletionOutcome {
  discarded_unconfirmed: boolean;
  discarded_count: number;
  /** False with a non-zero count means those objects are gone for good. */
  restorable: boolean;
  archive_id: string | null;
}

/** What unlocking actually restored. */
export interface SegmentationRestoreOutcome {
  restored_count: number;
  archived_count: number;
  restorable: boolean;
}

export type MarkSegmentationCompleteResponse = ImageSegmentation & {
  completion?: SegmentationCompletionOutcome;
};

export type UnlockSegmentationResponse = ImageSegmentation & {
  restored?: SegmentationRestoreOutcome;
};

/**
 * `GET /api/segmentations/<id>/` `delete_preview` — what deleting the
 * segmentation destroys, and what survives it.
 *
 * Same contract as the completion preview: read fresh when the confirm dialog
 * opens, because DELETE refuses an `acknowledged_object_count` that no longer
 * matches — usually a run that finished while the dialog was open.
 */
export interface SegmentationDeletePreview {
  segmentation_id: string;
  segmentation_type: string;
  /** Every object on the segmentation, whatever its label state. */
  object_count: number;
  objects_by_label_state: Record<string, number>;
  probability_map_count: number;
  overlay_count: number;
  adapter_count: number;
  /** Kept, not deleted — they survive marked `segmentation_deleted`. */
  analysis_run_count: number;
  locked: boolean;
}

export type SegmentationDetailResponse = ImageSegmentation & {
  delete_preview: SegmentationDeletePreview;
};

/** `DELETE /api/segmentations/<id>/` — what was deleted, and what was kept. */
export interface DeleteSegmentationResponse {
  deleted: SegmentationDeletePreview;
  analysis_runs_kept: number;
}
