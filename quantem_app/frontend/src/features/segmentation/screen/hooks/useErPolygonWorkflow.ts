import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  appendDraftSection,
  buildConfirmPolygons,
  closeDraftPolygon,
  dedupeConsecutive,
  getActiveDraftPolygon,
  hasDraftPolygonCloseCandidate,
  pointsEqual,
  resolveDraftPolygonCloseabilityAsync,
  type DraftPolygon,
  type DraftSegment,
} from "@/shared/geometry/draftGeometry";
import type { ImageSegmentation } from "@/shared/types/images";
import type { ConfirmBatchResponse } from "@/shared/types/segmentation";
import type { Point } from "@/utils/geometry";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import { mutationNoticeMessage } from "@/features/segmentation/screen/utils/mutationNotices";

interface SubmitConfirmedGeometriesOptions {
  geometries: Array<Array<[number, number]>>;
  samScores?: Array<number | null | undefined>;
  mergeOverlaps?: boolean;
  manualCreation?: boolean;
}

interface UseErPolygonWorkflowArgs {
  currentSegmentation: ImageSegmentation | null;
  active: boolean;
  isPointInsideImageBounds: (point: Point) => boolean;
  registerAnnotationActivity: () => void;
  showErrorToast: (message: string) => void;
  /** Something that worked but did not do what the gesture looked like. */
  showNoticeToast: (message: string) => void;
  submitConfirmedGeometriesOptimistically: (
    options: SubmitConfirmedGeometriesOptions
  ) => Promise<ConfirmBatchResponse | null>;
}

/**
 * ER Correct-mode polygon tool.
 *
 * Reuses the same vertex-click + `R`-to-close interaction as the Confirmed-Area
 * tool ({@link useCompletedRoiWorkflow} + the `draftGeometry` utils): a click
 * starts a section, a second click ends it, sections auto-connect, and closing
 * the polygon joins the first/last ends. The difference is the commit target --
 * instead of saving a confirmed-area mask, the closed ring is committed as a
 * filled, CONFIRMED manual ER object via the same union path Draw's "Confirm
 * Drawn Area" uses (`submitConfirmedGeometriesOptimistically` with
 * `mergeOverlaps: true`), so it fuses into overlapping confirmed objects.
 */
export function useErPolygonWorkflow({
  currentSegmentation,
  active,
  isPointInsideImageBounds,
  registerAnnotationActivity,
  showErrorToast,
  showNoticeToast,
  submitConfirmedGeometriesOptimistically,
}: UseErPolygonWorkflowArgs) {
  const idCounterRef = useRef(0);
  const [polygons, setPolygons] = useState<DraftPolygon[]>([]);
  const [draftPoints, setDraftPoints] = useState<Point[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  // Refs mirror the latest drawing state so pointer handlers stay stable and
  // never read a stale closure mid-gesture.
  const polygonsRef = useRef<DraftPolygon[]>([]);
  const draftPointsRef = useRef<Point[]>([]);
  const isDrawingRef = useRef(false);

  const nextId = useCallback((prefix: string) => {
    idCounterRef.current += 1;
    return `${prefix}-${idCounterRef.current}`;
  }, []);

  const clearDraft = useCallback(() => {
    polygonsRef.current = [];
    draftPointsRef.current = [];
    isDrawingRef.current = false;
    setPolygons([]);
    setDraftPoints([]);
    setIsDrawing(false);
  }, []);

  useEffect(() => {
    polygonsRef.current = polygons;
  }, [polygons]);

  // Clear any in-progress draft when the tool deactivates or the image changes.
  useEffect(() => {
    if (!active) {
      clearDraft();
    }
  }, [active, clearDraft]);

  useEffect(() => {
    clearDraft();
  }, [clearDraft, currentSegmentation?.id]);

  const closeCandidatePolygons = useMemo(() => {
    if (!isDrawing || draftPoints.length < 2) {
      return polygons;
    }
    return appendDraftSection(polygons, {
      id: "er-polygon-preview-section",
      kind: "section",
      points: draftPoints,
      endedOutsidePatch: false,
    });
  }, [isDrawing, draftPoints, polygons]);

  const hasDraft = polygons.length > 0 || draftPoints.length > 0;
  const canClosePolygon =
    active &&
    !submitting &&
    hasDraftPolygonCloseCandidate(getActiveDraftPolygon(closeCandidatePolygons));

  const startSection = useCallback((point: Point) => {
    const activeDraftPolygon = getActiveDraftPolygon(polygonsRef.current);
    const previousSegment =
      activeDraftPolygon?.segments[activeDraftPolygon.segments.length - 1] ?? null;
    const previousEnd = previousSegment
      ? previousSegment.points[previousSegment.points.length - 1]
      : null;
    const initialPoints = previousEnd ? dedupeConsecutive([previousEnd, point]) : [point];
    draftPointsRef.current = initialPoints;
    setDraftPoints(initialPoints);
    isDrawingRef.current = true;
    setIsDrawing(true);
  }, []);

  const finalizeSection = useCallback(
    (point: Point) => {
      const nextDraft = dedupeConsecutive([...draftPointsRef.current, point]);
      draftPointsRef.current = [];
      setDraftPoints([]);
      isDrawingRef.current = false;
      setIsDrawing(false);
      if (nextDraft.length < 2) {
        return;
      }
      const section: DraftSegment = {
        id: nextId("er-polygon-section"),
        kind: "section",
        points: nextDraft,
        endedOutsidePatch: false,
      };
      setPolygons((current) => {
        const next = appendDraftSection(current, section);
        polygonsRef.current = next;
        return next;
      });
    },
    [nextId]
  );

  const handlePolygonClick = useCallback(
    (point: Point) => {
      if (!active || submitting || !isPointInsideImageBounds(point)) {
        return;
      }
      registerAnnotationActivity();
      if (isDrawingRef.current) {
        finalizeSection(point);
      } else {
        startSection(point);
      }
    },
    [
      active,
      finalizeSection,
      isPointInsideImageBounds,
      registerAnnotationActivity,
      startSection,
      submitting,
    ]
  );

  const handlePolygonMouseMove = useCallback(
    (point: Point) => {
      if (!active || !isDrawingRef.current) {
        return;
      }
      if (!isPointInsideImageBounds(point)) {
        return;
      }
      registerAnnotationActivity();
      setDraftPoints((current) => {
        if (current.length > 0 && pointsEqual(current[current.length - 1], point)) {
          return current;
        }
        const next = [...current, point];
        draftPointsRef.current = next;
        return next;
      });
    },
    [active, isPointInsideImageBounds, registerAnnotationActivity]
  );

  const handleClosePolygon = useCallback(async () => {
    if (!active || submitting || !currentSegmentation) {
      return;
    }

    const candidatePolygons =
      isDrawingRef.current && draftPointsRef.current.length >= 2
        ? appendDraftSection(polygonsRef.current, {
            id: nextId("er-polygon-section"),
            kind: "section",
            points: draftPointsRef.current,
            endedOutsidePatch: false,
          })
        : polygonsRef.current;
    const candidatePolygon = getActiveDraftPolygon(candidatePolygons);
    const closeability = await resolveDraftPolygonCloseabilityAsync(candidatePolygon);
    if (closeability.kind === "error") {
      showErrorToast(closeability.message);
      return;
    }
    if (closeability.kind !== "ok") {
      return;
    }

    const closedPolygons = closeDraftPolygon(
      candidatePolygons,
      () => nextId("er-polygon-closing"),
      closeability.ring
    );
    const confirmPolygons = buildConfirmPolygons(closedPolygons);
    if (confirmPolygons.length === 0) {
      clearDraft();
      return;
    }

    registerAnnotationActivity();
    setSubmitting(true);
    try {
      // ER fuses a drawn shape into overlapping confirmed objects (union path),
      // the same commit Draw's "Confirm Drawn Area" uses.
      const response = await submitConfirmedGeometriesOptimistically({
        geometries: [confirmPolygons[0]],
        mergeOverlaps: true,
        manualCreation: true,
      });
      clearDraft();
      // Closing the polygon clears the draft whatever came back, so a ring the
      // server stored nothing for looks exactly like one it did.
      const notice = mutationNoticeMessage(response, {
        nothingStoredMessage:
          "Nothing was stored: that ring encloses no area more than a pixel across.",
      });
      if (notice) showNoticeToast(notice);
    } catch (error) {
      showErrorToast(
        extractApiErrorMessage(error, "Failed to create the polygon object.")
      );
    } finally {
      setSubmitting(false);
    }
  }, [
    active,
    clearDraft,
    currentSegmentation,
    nextId,
    registerAnnotationActivity,
    showErrorToast,
    showNoticeToast,
    submitConfirmedGeometriesOptimistically,
    submitting,
  ]);

  return {
    active,
    polygons,
    liveSectionPoints: draftPoints,
    hasDraft,
    canClosePolygon,
    submitting,
    handlePolygonClick,
    handlePolygonMouseMove,
    handleClosePolygon,
    clearDraft,
  };
}
