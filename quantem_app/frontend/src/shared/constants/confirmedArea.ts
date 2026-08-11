/**
 * One name and one explanation for each of the two things that count as
 * ground truth.
 *
 * The confirmed area had three names: "Confirmed Area" in the toolbar,
 * "completed ROI" in the API and `blockers`, and "mark the area you have
 * finished annotating" in a tooltip -- and the only place the *reason* was
 * written down was the body of a 400 response nobody reads. That reason is the
 * whole point of the feature: it is what separates "background" from "not
 * looked at yet", and without it every Dice reported here would be a fiction.
 *
 * The UI name is "confirmed area". `completed ROI` stays as the API term, and
 * is mentioned once so the 400 body and the readiness blockers are
 * recognisable.
 */

export const CONFIRMED_AREA_LABEL = "Confirmed area";

/** The sentence that says why the feature exists. Shown next to the control. */
export const CONFIRMED_AREA_EXPLANATION =
  "Inside a confirmed area, anything you have not confirmed as an object counts " +
  "as true background. Outside it, pixels are ignored. Without one, training " +
  "and scoring cannot tell background from not-yet-annotated.";

/** Where the same concept appears under its API name. */
export const CONFIRMED_AREA_API_ALIAS_NOTE =
  "The API calls this a completed ROI.";

/** Compact form for `title` attributes, which cannot hold two paragraphs. */
export const CONFIRMED_AREA_TOOLTIP = `${CONFIRMED_AREA_EXPLANATION} ${CONFIRMED_AREA_API_ALIAS_NOTE}`;

/** How to make one, from a screen that is not the toolbar. */
export const CONFIRMED_AREA_HOW_TO =
  'Switch to Review, choose Correct, then "Confirmed area", and draw round the ' +
  "region you finished.";

/**
 * The other way to say "the labels in here are finished".
 *
 * The ROI list has a per-organelle tick box labelled **Reviewed**, writing
 * `RoiSegmentationStatus.is_complete`. It used to be bookkeeping and nothing
 * more: the copy here said so at length, and said that only a `CompletedROI`
 * polygon was "what Adapt a model needs". Both halves of that were true and
 * both are now false -- a reviewed ROI is read as training data, on exactly the
 * terms its own docstring always described (dense labels inside, ignore
 * outside), which is the same treatment a confirmed-area polygon gets.
 *
 * What is left of the distinction is shape and intent, not standing: a
 * confirmed area is a polygon you draw round whatever you finished, a reviewed
 * ROI is a whole window you tick once you have been through it. The label stays
 * "Reviewed" because that is the act it records; it no longer claims that
 * ticking it counts for nothing.
 */
export const ROI_REVIEWED_LABEL = "Reviewed";

export const ROI_REVIEWED_EXPLANATION =
  'Ticking "Reviewed" records that you have been through this ROI window, and ' +
  "counts it as finished labelling: inside it, anything you have not confirmed " +
  "as an object is treated as background, and outside it pixels are ignored. " +
  `A ${CONFIRMED_AREA_LABEL.toLowerCase()} says the same thing about a shape ` +
  "you draw yourself. Fine-tuning trains on both.";

export const ROI_REVIEWED_TOOLTIP = `${ROI_REVIEWED_EXPLANATION} ${CONFIRMED_AREA_HOW_TO}`;

/**
 * What the Fine-Tune dialog counts, in one phrase.
 *
 * The two sources are one number on screen -- the owner asked for "confirmed
 * areas and/or done ROIs ... as one total" -- so they need one name, and it has
 * to be the plain-language one. "Annotation" is that name everywhere the count
 * is shown.
 */
export const TRAINING_ANNOTATION_SOURCES =
  `A ${CONFIRMED_AREA_LABEL.toLowerCase()} you drew, or an ROI you ticked ` +
  `"${ROI_REVIEWED_LABEL}", both count as one annotation each.`;
