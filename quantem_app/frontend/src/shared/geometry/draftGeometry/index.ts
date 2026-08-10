/**
 * Dependency-free freehand drawing geometry: draft polygons with closure
 * detection, invertible splice contracts, lasso cut and include/exclude.
 *
 * UI-agnostic on purpose — it only knows `Point`/`BBox` — so any screen that
 * needs polygon creation and closing can build on it.
 */
export type {
  DisplayGeometry,
  DraftCutOverlay,
  DraftPolygon,
  DraftSegment,
  DraftEditorMode,
  DraftRepairSession,
  SpliceAnchor,
  SpliceContract,
  SpliceSection,
} from "@/shared/geometry/draftGeometry/types";
export type {
  CutDraftResult,
} from "@/shared/geometry/draftGeometry/internalTypes";
export {
  appendDraftSection,
  buildAutomaticPrefillPolygons,
  buildClosedPolygonRing,
  buildConfirmPolygons,
  canCloseDraftPolygon,
  closeDraftPolygon,
  countCompletedSections,
  deleteLastDraftSegment,
  flattenDraftSegments,
  getActiveDraftPolygon,
  getClosedDraftPolygons,
  hasDraftPolygonCloseCandidate,
  resolveDraftPolygonCloseabilityAsync,
  resolveDraftPolygonCloseability,
} from "@/shared/geometry/draftGeometry/polygons";
export {
  appendRepairSection,
  closeActiveRepairSession,
  countRepairSegments,
  deleteLastRepairSegment,
  getActiveRepairSession,
  hasRepairSessionCloseCandidate,
  projectPointToPolyline,
  resolveRepairSessionReplacementPath,
  resolveRepairStartAnchor,
  resolveRepairSessionCloseability,
  type PolylineProjection,
} from "@/shared/geometry/draftGeometry/repair";
export {
  cutDraftPolygonsWithLasso,
  extractClosedRingPathWithinLasso,
} from "@/shared/geometry/draftGeometry/cut";
export {
  applySpliceContract,
  applySpliceContractsToBaseRing,
  invertSpliceContract,
  type SpliceResult,
} from "@/shared/geometry/draftGeometry/splice";
export {
  buildPatchBBoxFromPoints,
  patchBBoxIntersectsImage,
  patchBBoxWithinLimit,
  pointInsidePatch,
  validateAutomaticPatchSelection,
} from "@/shared/geometry/draftGeometry/patch";
export {
  dedupeConsecutive,
  pointsEqual,
} from "@/shared/geometry/draftGeometry/shared";
