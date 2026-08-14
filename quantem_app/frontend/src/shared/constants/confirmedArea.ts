/** Shared copy for the two annotation-completeness mechanisms. */

export const CONFIRMED_AREA_LABEL = "Confirmed area";

export const CONFIRMED_AREA_HOW_TO =
  'Switch to Review, choose Correct, then "Confirmed area", and draw round the ' +
  "region you finished.";

/** Completion state for one fixed ROI window and one segmentation. */
export const ROI_REVIEWED_LABEL = "Done";

export const ROI_REVIEWED_EXPLANATION =
  "Done marks this ROI as completely labeled for the current segmentation.";

export const ROI_REVIEWED_TOOLTIP =
  `${ROI_REVIEWED_EXPLANATION} Fine-tuning treats unconfirmed pixels inside it ` +
  "as background and ignores pixels outside it.";

export const TRAINING_ANNOTATION_SOURCES =
  `A ${CONFIRMED_AREA_LABEL.toLowerCase()} you drew, or an ROI marked ` +
  `"${ROI_REVIEWED_LABEL}", both count as one annotation each.`;
