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
import {
  createCompletedRoi,
  getCompletedRois,
  subtractCompletedRoi,
} from "@/shared/api/segmentations/completedRois";
import { querySegmentsInRegion } from "@/shared/api/segmentations/annotations";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import type { CompletedRoi, CompletedRoiMode } from "@/shared/types/segmentation";
import type { ImageSegmentation } from "@/shared/types/images";
import type { Point } from "@/utils/geometry";
import { extractApiErrorMessage } from "@/utils/apiErrors";

/**
 * How many unconfirmed candidates sit inside the polygon about to be saved.
 *
 * Marking an area complete declares that everything in it which is *not* a
 * confirmed object is true background, and the adapter's training target is
 * built on exactly that rule (`_confirmed_objects_in` rasterises CONFIRMED
 * only; every other pixel inside the ROI is a zero). So each candidate left
 * unconfirmed inside the area becomes a negative example. The dialog stated
 * the rule and never counted them -- 88 candidates were about to be trained
 * against as background with nothing on screen saying so.
 *
 * `status` is explicit because "we could not count them" and "there are none"
 * must not render the same way.
 */
export interface PendingBackgroundCount {
  status: "idle" | "loading" | "ready" | "error";
  count: number | null;
}

/**
 * True when the count is worth flagging rather than merely reporting.
 *
 * "Could not count" and "counted zero" both stay neutral: only a known,
 * non-zero number is something the user should stop and look at.
 */
export function isBackgroundCountWarning(
  pending: PendingBackgroundCount
): boolean {
  return pending.status === "ready" && (pending.count ?? 0) > 0;
}

interface UseCompletedRoiWorkflowArgs {
  currentSegmentation: ImageSegmentation | null;
  active: boolean;
  isPointInsideImageBounds: (point: Point) => boolean;
  registerAnnotationActivity: () => void;
  showErrorToast: (message: string) => void;
}

/**
 * Confirmed-area drawing workflow.
 *
 * The segment interaction is driven by `shared/geometry/draftGeometry`: a click
 * starts a section, a second click
 * ends it, each new section auto-connects to the previous section's end, and
 * `Close Polygon` (R) joins the first/last ends into a closed shape. Pointer
 * handlers stay stable and read the live drawing state through refs so a click
 * is never lost to a stale closure or a mid-gesture re-render.
 */
export function useCompletedRoiWorkflow({
  currentSegmentation,
  active,
  isPointInsideImageBounds,
  registerAnnotationActivity,
  showErrorToast,
}: UseCompletedRoiWorkflowArgs) {
  const idCounterRef = useRef(0);
  const [polygons, setPolygons] = useState<DraftPolygon[]>([]);
  const [draftPoints, setDraftPoints] = useState<Point[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [mode, setMode] = useState<CompletedRoiMode>("include");
  const [pendingBackground, setPendingBackground] =
    useState<PendingBackgroundCount>({ status: "idle", count: null });

  // Refs mirror the latest drawing state so the pointer handlers can be stable
  // (not recreated on every mouse move) and always read current values.
  const polygonsRef = useRef<DraftPolygon[]>([]);
  const draftPointsRef = useRef<Point[]>([]);
  const isDrawingRef = useRef(false);

  const nextId = useCallback((prefix: string) => {
    idCounterRef.current += 1;
    return `${prefix}-${idCounterRef.current}`;
  }, []);

  const query = useApiQuery(
    () =>
      active && currentSegmentation
        ? getCompletedRois(currentSegmentation.id)
        : Promise.resolve([] as CompletedRoi[]),
    [active, currentSegmentation?.id]
  );

  const clearDraft = useCallback(() => {
    polygonsRef.current = [];
    draftPointsRef.current = [];
    isDrawingRef.current = false;
    setPolygons([]);
    setDraftPoints([]);
    setIsDrawing(false);
    setSaveDialogOpen(false);
    setPendingBackground({ status: "idle", count: null });
  }, []);

  useEffect(() => {
    polygonsRef.current = polygons;
  }, [polygons]);

  useEffect(() => {
    if (active) {
      return;
    }
    clearDraft();
  }, [active, clearDraft]);

  useEffect(() => {
    clearDraft();
    setMode("include");
  }, [clearDraft, currentSegmentation?.id]);

  const confirmPolygons = useMemo(() => buildConfirmPolygons(polygons), [polygons]);
  const activePolygon = useMemo(() => getActiveDraftPolygon(polygons), [polygons]);

  const confirmPolygonsRef = useRef(confirmPolygons);
  useEffect(() => {
    confirmPolygonsRef.current = confirmPolygons;
  }, [confirmPolygons]);

  const closeCandidatePolygons = useMemo(() => {
    if (!isDrawing || draftPoints.length < 2) {
      return polygons;
    }
    return appendDraftSection(polygons, {
      id: "completed-roi-preview-section",
      kind: "section",
      points: draftPoints,
      endedOutsidePatch: false,
    });
  }, [isDrawing, draftPoints, polygons]);

  const hasDraft = polygons.length > 0 || draftPoints.length > 0;
  const canClosePolygon =
    active &&
    hasDraftPolygonCloseCandidate(getActiveDraftPolygon(closeCandidatePolygons));
  const canSave = active && !isDrawing && confirmPolygons.length === 1 && !saving;

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
        id: nextId("completed-roi-section"),
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
      if (!active || !isPointInsideImageBounds(point)) {
        return;
      }
      // One confirmed-area polygon at a time: ignore new clicks until the
      // closed shape is saved or cleared.
      if (confirmPolygonsRef.current.length > 0) {
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
    if (!active) {
      return;
    }

    const candidatePolygons =
      isDrawingRef.current && draftPointsRef.current.length >= 2
        ? appendDraftSection(polygonsRef.current, {
            id: nextId("completed-roi-section"),
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

    registerAnnotationActivity();
    const closedPolygons = closeDraftPolygon(
      candidatePolygons,
      () => nextId("completed-roi-closing"),
      closeability.ring
    );
    polygonsRef.current = closedPolygons;
    draftPointsRef.current = [];
    isDrawingRef.current = false;
    setPolygons(closedPolygons);
    setDraftPoints([]);
    setIsDrawing(false);
  }, [active, nextId, registerAnnotationActivity, showErrorToast]);

  const requestSave = useCallback(() => {
    if (!canSave) {
      return;
    }
    setSaveDialogOpen(true);

    // Only "include" turns unconfirmed candidates into background; "exclude"
    // removes area from the mask, so there is nothing to warn about.
    if (mode !== "include" || !currentSegmentation) {
      setPendingBackground({ status: "idle", count: null });
      return;
    }
    const polygon = confirmPolygonsRef.current[0];
    if (!polygon) {
      setPendingBackground({ status: "idle", count: null });
      return;
    }
    setPendingBackground({ status: "loading", count: null });
    // Not source-model filtered: every unconfirmed object inside the area
    // becomes background regardless of which model proposed it.
    querySegmentsInRegion(currentSegmentation.id, {
      polygon_coords: polygon,
      states: ["CANDIDATE", "INFERRED"],
      include_geometry: false,
    })
      .then((result) => {
        setPendingBackground({
          status: "ready",
          count: result.segments.length,
        });
      })
      .catch((error) => {
        console.error("Failed to count candidates in the confirmed area:", error);
        setPendingBackground({ status: "error", count: null });
      });
  }, [canSave, currentSegmentation, mode]);

  const cancelSave = useCallback(() => {
    setSaveDialogOpen(false);
    setPendingBackground({ status: "idle", count: null });
  }, []);

  const confirmSave = useCallback(async () => {
    if (!currentSegmentation || confirmPolygons.length !== 1 || saving) {
      return;
    }
    setSaving(true);
    registerAnnotationActivity();
    try {
      if (mode === "exclude") {
        await subtractCompletedRoi(currentSegmentation.id, {
          polygon_coords: confirmPolygons[0],
        });
      } else {
        await createCompletedRoi(currentSegmentation.id, {
          polygon_coords: confirmPolygons[0],
        });
      }
      setSaveDialogOpen(false);
      setPendingBackground({ status: "idle", count: null });
      clearDraft();
      await query.refetch();
    } catch (error) {
      showErrorToast(
        extractApiErrorMessage(
          error,
          mode === "exclude"
            ? "Failed to exclude the area from the confirmed area."
            : "Failed to save the confirmed area."
        )
      );
      console.error("Failed to update confirmed area:", error);
    } finally {
      setSaving(false);
    }
  }, [
    clearDraft,
    confirmPolygons,
    currentSegmentation,
    mode,
    query,
    registerAnnotationActivity,
    saving,
    showErrorToast,
  ]);

  return {
    active,
    mode,
    setMode,
    items: active ? query.data ?? [] : [],
    loading: active && (query.loading || query.refetching),
    polygons,
    liveSectionPoints: draftPoints,
    activePolygon,
    hasDraft,
    canClosePolygon,
    canSave,
    saving,
    saveDialogOpen,
    pendingBackground,
    clearDraft,
    requestSave,
    cancelSave,
    confirmSave,
    handlePolygonClick,
    handlePolygonMouseMove,
    handleClosePolygon,
  };
}
