/**
 * Where the ground truth came from.
 *
 * A held-out Dice of 0.99 means something very different depending on how the
 * annotations were made. If 86 of 90 "annotations" are the model's own
 * candidates that the user clicked to confirm, the model is largely being
 * scored against its own output, and near-perfect agreement is close to
 * tautological. If they were drawn by hand against a blank canvas, the same
 * number is a real result.
 *
 * The backend does not compute this split: `AnnotatedCrop.n_objects` counts
 * confirmed objects inside a completed ROI and stops there. But every
 * `SegmentObject` carries `source_model`, which is `"manual"` for something the
 * user drew and the pack id for something a model proposed, so the split is
 * exactly one region query per completed ROI away.
 *
 * Counted inside the completed ROIs specifically, not across the image: the ROI
 * is what defines the training target (`_confirmed_objects_in` rasterises
 * CONFIRMED objects intersecting the ROI polygon; everything else inside it is
 * background), so objects outside would inflate the numbers with things the run
 * never saw.
 */

import { getCompletedRois } from "@/shared/api/segmentations/completedRois";
import { querySegmentsInRegion } from "@/shared/api/segmentations/annotations";

/** Source model value the backend uses for something the user drew by hand. */
export const MANUAL_SOURCE_MODEL = "manual";

export interface GroundTruthProvenance {
  /** Confirmed objects that a model proposed and the user accepted. */
  confirmedFromModel: number;
  /** Confirmed objects with no model behind them — drawn by hand. */
  drawnByHand: number;
  /** Objects explicitly rejected inside the annotated area. */
  rejected: number;
  /** `confirmedFromModel + drawnByHand`: what the run trains its positives on. */
  totalConfirmed: number;
  /** Completed ROIs the counts were gathered from. */
  regions: number;
}

export const EMPTY_PROVENANCE: GroundTruthProvenance = {
  confirmedFromModel: 0,
  drawnByHand: 0,
  rejected: 0,
  totalConfirmed: 0,
  regions: 0,
};

/**
 * Share of the confirmed positives that started life as model output, 0..1.
 *
 * Null when there is nothing confirmed, because 0/0 is not "none of it".
 */
export function modelDerivedFraction(
  provenance: GroundTruthProvenance
): number | null {
  if (provenance.totalConfirmed === 0) return null;
  return provenance.confirmedFromModel / provenance.totalConfirmed;
}

/**
 * True when the score deserves an explicit caveat next to it.
 *
 * The threshold is deliberately not "any model-derived object at all":
 * confirming model output is the intended workflow and is not by itself a
 * problem. It becomes one when almost none of the ground truth is independent
 * of the model being scored.
 */
export const MODEL_DERIVED_CAVEAT_FRACTION = 0.8;

export function needsSelfAgreementCaveat(
  provenance: GroundTruthProvenance
): boolean {
  const fraction = modelDerivedFraction(provenance);
  return fraction !== null && fraction >= MODEL_DERIVED_CAVEAT_FRACTION;
}

/** Sum the counts from several segmentations into one. */
export function mergeProvenance(
  parts: GroundTruthProvenance[]
): GroundTruthProvenance {
  return parts.reduce(
    (total, part) => ({
      confirmedFromModel: total.confirmedFromModel + part.confirmedFromModel,
      drawnByHand: total.drawnByHand + part.drawnByHand,
      rejected: total.rejected + part.rejected,
      totalConfirmed: total.totalConfirmed + part.totalConfirmed,
      regions: total.regions + part.regions,
    }),
    EMPTY_PROVENANCE
  );
}

/**
 * Count the ground truth inside one segmentation's completed ROIs.
 *
 * One request for the ROI polygons plus one region query per ROI. Geometry is
 * left out of the responses — only `source_model` and `label_state` are read.
 *
 * An object touching two overlapping ROIs would be counted twice; the backend
 * merges overlapping completed areas into one polygon on save, so in practice
 * they do not overlap.
 */
export async function fetchGroundTruthProvenance(
  segmentationId: string
): Promise<GroundTruthProvenance> {
  const rois = await getCompletedRois(segmentationId);
  if (rois.length === 0) return EMPTY_PROVENANCE;

  const perRoi = await Promise.all(
    rois.map(async (roi) => {
      const { segments } = await querySegmentsInRegion(segmentationId, {
        polygon_coords: roi.polygon_coords,
        states: ["CONFIRMED", "EXCLUDED"],
        include_geometry: false,
      });
      let confirmedFromModel = 0;
      let drawnByHand = 0;
      let rejected = 0;
      for (const segment of segments) {
        if (segment.label_state === "EXCLUDED") {
          rejected += 1;
          continue;
        }
        if (segment.source_model === MANUAL_SOURCE_MODEL) {
          drawnByHand += 1;
        } else {
          confirmedFromModel += 1;
        }
      }
      return {
        confirmedFromModel,
        drawnByHand,
        rejected,
        totalConfirmed: confirmedFromModel + drawnByHand,
        regions: 1,
      };
    })
  );

  return mergeProvenance(perRoi);
}
