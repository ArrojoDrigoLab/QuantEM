import type { LabelState } from "@/shared/types/common";
import type { UserFeedbackUtilizedStatus } from "@/shared/types/segmentation";
export {
  CANDIDATE_BORDER_COLOR,
  CANDIDATE_FILL_COLOR,
  CONFIRMED_BORDER_COLOR,
  CONFIRMED_FILL_COLOR,
  LABELED_BORDER_COLOR,
  LABELED_FILL_COLOR,
  RASTER_BORDER_OPACITY,
  REFINED_BORDER_COLOR,
  REFINED_FILL_COLOR,
  TISSUE_INTERNAL_NAME,
} from "@/shared/constants/segmentation";

export const STATUS_POLL_MS = 3000;
export const FEEDBACK_POLL_MS = 1500;
export const ROI_JOB_TYPE = "run_segmentation_roi_task";
export const FULL_IMAGE_JOB_TYPE = "run_segmentation_full_task";
export const PROCESSING_BANNER_JOB_TYPES = new Set([
  ROI_JOB_TYPE,
  FULL_IMAGE_JOB_TYPE,
]);
export const ORGANELLE_ACTION_JOB_TYPES = new Set([
  ROI_JOB_TYPE,
  FULL_IMAGE_JOB_TYPE,
]);
export const RIGHT_BBOX_DRAG_THRESHOLD_PX = 8;
/** Fixed brush diameter (image px) for the one-click "add object" correction tool. */
export const ADD_BRUSH_DIAMETER = 20;
export const DEFAULT_CANDIDATE_OVERLAY_OPACITY = 0.18;
export const OVERLAY_REFRESH_IDLE_DELAY_MS = 5000;
export const OVERLAY_REFRESH_MAX_DELAY_MS = 30000;
export const HOVER_QUERY_DEBOUNCE_MS = 250;
export const FEEDBACK_PENDING_STATUSES: UserFeedbackUtilizedStatus[] = [
  "QUEUED",
  "PROCESSING",
];
export const POINT_FEEDBACK_SEGMENTATION_TYPES = new Set([
  "quantem_internal_mito",
  "quantem_internal_mito_deepcontact_cell",
  "quantem_internal_mito_deepcontact_sem",
  "quantem_internal_mito_deepcontact_tem",
  "quantem_internal_er",
  "quantem_internal_er_deepcontact_cell",
  "quantem_internal_er_deepcontact_sem",
  "quantem_internal_er_deepcontact_tem",
  "quantem_internal_nucleus",
  "quantem_internal_ld",
]);
export const LABELING_LEFT_PANEL_STATES = new Set<LabelState>([
  "CONFIRMED",
  "CANDIDATE",
]);
