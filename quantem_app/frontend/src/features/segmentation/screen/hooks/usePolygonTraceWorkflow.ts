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
import type { Point } from "@/utils/geometry";
import { extractApiErrorMessage } from "@/utils/apiErrors";

interface UsePolygonTraceWorkflowArgs {
  /** Whether this trace tool is currently the active tool. */
  active: boolean;
  /** Distinct prefix so multiple traces on one screen never share draft ids. */
  idPrefix: string;
  isPointInsideImageBounds: (point: Point) => boolean;
  registerAnnotationActivity: () => void;
  showErrorToast: (message: string) => void;
  /**
   * Commit the closed polygon ring. The ring is a closed list of `[x, y]`
   * image-space coordinates. Resolve to commit + clear the draft; reject to
   * surface {@link commitErrorMessage} and keep the draft for retry.
   */
  onCommit: (ring: Array<[number, number]>) => Promise<unknown>;
  commitErrorMessage: string;
  /** Reset the draft whenever this value changes (e.g. the segmentation id). */
  resetKey?: string | null;
}

/**
 * Reusable click-to-trace polygon tool.
 *
 * Extracted from {@link useErPolygonWorkflow}: a click starts a section, a
 * second click ends it, sections auto-connect end-to-end, and closing the
 * polygon (R, or connecting the first/last ends) joins them into one filled
 * ring. The only per-use difference is `onCommit` -- what the closed ring is
 * saved as (a confirmed object, an excluded area, ...).
 */
export function usePolygonTraceWorkflow({
  active,
  idPrefix,
  isPointInsideImageBounds,
  registerAnnotationActivity,
  showErrorToast,
  onCommit,
  commitErrorMessage,
  resetKey,
}: UsePolygonTraceWorkflowArgs) {
  const idCounterRef = useRef(0);
  const [polygons, setPolygons] = useState<DraftPolygon[]>([]);
  const [draftPoints, setDraftPoints] = useState<Point[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const polygonsRef = useRef<DraftPolygon[]>([]);
  const draftPointsRef = useRef<Point[]>([]);
  const isDrawingRef = useRef(false);

  const nextId = useCallback(
    (suffix: string) => {
      idCounterRef.current += 1;
      return `${idPrefix}-${suffix}-${idCounterRef.current}`;
    },
    [idPrefix]
  );

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

  // Clear any in-progress draft when the tool deactivates or the reset key changes.
  useEffect(() => {
    if (!active) {
      clearDraft();
    }
  }, [active, clearDraft]);

  useEffect(() => {
    clearDraft();
  }, [clearDraft, resetKey]);

  const closeCandidatePolygons = useMemo(() => {
    if (!isDrawing || draftPoints.length < 2) {
      return polygons;
    }
    return appendDraftSection(polygons, {
      id: `${idPrefix}-preview-section`,
      kind: "section",
      points: draftPoints,
      endedOutsidePatch: false,
    });
  }, [idPrefix, isDrawing, draftPoints, polygons]);

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
        id: nextId("section"),
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
    if (!active || submitting) {
      return;
    }

    const candidatePolygons =
      isDrawingRef.current && draftPointsRef.current.length >= 2
        ? appendDraftSection(polygonsRef.current, {
            id: nextId("section"),
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
      () => nextId("closing"),
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
      await onCommit(confirmPolygons[0]);
      clearDraft();
    } catch (error) {
      showErrorToast(extractApiErrorMessage(error, commitErrorMessage));
    } finally {
      setSubmitting(false);
    }
  }, [
    active,
    clearDraft,
    commitErrorMessage,
    nextId,
    onCommit,
    registerAnnotationActivity,
    showErrorToast,
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
