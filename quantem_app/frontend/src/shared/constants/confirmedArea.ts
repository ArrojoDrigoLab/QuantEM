/**
 * One name and one explanation for the confirmed-area concept.
 *
 * It had three: "Confirmed Area" in the toolbar, "completed ROI" in the API and
 * `blockers`, and "mark the area you have finished annotating" in a tooltip --
 * and the only place the *reason* was written down was the body of a 400
 * response nobody reads. That reason is the whole point of the feature: it is
 * what separates "background" from "not looked at yet", and without it every
 * Dice the fine-tuning wizard reports would be a fiction.
 *
 * The UI name is "confirmed area". `completed ROI` stays as the API term, and
 * is mentioned once so the 400 body and the wizard's blockers are recognisable.
 */

export const CONFIRMED_AREA_LABEL = "Confirmed area";

/** The sentence that says why the feature exists. Shown next to the control. */
export const CONFIRMED_AREA_EXPLANATION =
  "Inside a confirmed area, anything you have not confirmed as an object counts " +
  "as true background. Outside it, pixels are ignored. Without one, training " +
  "and scoring cannot tell background from not-yet-annotated.";

/** Where the same concept appears under its API name. */
export const CONFIRMED_AREA_API_ALIAS_NOTE =
  "The API and the Adapt wizard call this a completed ROI.";

/** Compact form for `title` attributes, which cannot hold two paragraphs. */
export const CONFIRMED_AREA_TOOLTIP = `${CONFIRMED_AREA_EXPLANATION} ${CONFIRMED_AREA_API_ALIAS_NOTE}`;

/** How to make one, from a screen that is not the toolbar. */
export const CONFIRMED_AREA_HOW_TO =
  'Switch to Review, choose Correct, then "Confirmed area", and draw round the ' +
  "region you finished.";

/**
 * The fourth name for "complete", and the one that was doing damage.
 *
 * The ER ROI list has a per-organelle tick box that was labelled **Done (ER)**.
 * It writes `RoiSegmentationStatus.is_complete`, which nothing outside that list
 * reads -- not `extract_crops`, not the adapter, not the analysis. The Adapt
 * wizard's precondition is a `CompletedROI` *polygon*, which only the Confirmed
 * area tool creates. So ticking Done (ER) and then opening the wizard produced
 * *"No completed ROI on this image. Mark the area you have finished annotating
 * as complete."* -- advice to do the thing you had just done, in the same words.
 *
 * The tick box is not useless: it is the user's own record of which 2048² window
 * they have been through, which is worth having when there are twenty of them.
 * It just must not use the word the wizard means.
 */
export const ROI_REVIEWED_LABEL = "Reviewed";

export const ROI_REVIEWED_EXPLANATION =
  'Ticking "Reviewed" records that you have been through this ROI window. It is ' +
  "your own bookkeeping and nothing else reads it — in particular it does not " +
  `create a ${CONFIRMED_AREA_LABEL.toLowerCase()}, which is what "Adapt a model" ` +
  "needs before it will train or score.";

export const ROI_REVIEWED_TOOLTIP = `${ROI_REVIEWED_EXPLANATION} ${CONFIRMED_AREA_HOW_TO}`;
