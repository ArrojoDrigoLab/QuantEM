/**
 * Drag a box, get an object.
 *
 * Owns the tool's on/off state, the drag, the request, and the weights. The
 * screen supplies three callbacks -- what to do after an object lands, how to
 * show an error, and how to say a gesture is under way -- and nothing else.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { BBox } from "@/shared/types/common";
import type { Point } from "@/utils/geometry";
import type { SegmentOverlay } from "@/viewer/types";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import { getSamModelStatus, promptSamBox, startSamModelDownload } from "./api";
import { samLiveBoxOverlay, samPendingBoxOverlay } from "./overlays";
import type { SamBoxResponse, SamModelStatus } from "./types";

/**
 * Drag-versus-click threshold, in **screen** pixels.
 *
 * A fixed number on purpose. The tool this was ported from used 2% of the
 * viewport width, which sounds adaptive and is not: on a narrow viewport a
 * perfectly deliberate small box falls under the threshold and is silently
 * discarded as a click. 8 px is the value the neighbouring group-select drag
 * already uses, and it does the job at every viewport size.
 */
const DRAG_THRESHOLD_PX = 8;

/** How often to re-read the download's progress while it runs. */
const MODEL_POLL_MS = 1000;

interface UseSamBoxToolArgs {
  segmentationId: string | null;
  /** False when another tool owns the pointer, or the view is read-only. */
  available: boolean;
  onObjectCreated: (response: SamBoxResponse) => void | Promise<void>;
  onError: (message: string) => void;
  /** Defers the overlay refetch while the user is mid-gesture. */
  registerActivity?: () => void;
}

export interface SamBoxTool {
  /** The tool is selected; the pointer belongs to it. */
  isActive: boolean;
  setActive: (active: boolean) => void;
  /** A request is in flight. */
  isSubmitting: boolean;
  /** Live rubber band plus any in-flight box, ready for the transient layer. */
  overlays: SegmentOverlay[];
  handleImagePress: (imagePoint: Point, screenPoint: Point) => void;
  handleImageDrag: (imagePoint: Point, screenPoint: Point) => void;
  handleImageRelease: (imagePoint: Point, screenPoint: Point) => void;
  /** Drop a half-drawn box (Escape, or leaving the tool). */
  cancelDrag: () => void;
  model: SamModelStatus | null;
  modelReady: boolean;
  downloadModel: () => void;
  /** Timing of the last successful prompt, for the "first one is slower" hint. */
  lastTiming: SamBoxResponse["timing"] | null;
}

export function useSamBoxTool({
  segmentationId,
  available,
  onObjectCreated,
  onError,
  registerActivity,
}: UseSamBoxToolArgs): SamBoxTool {
  const [isActive, setIsActive] = useState(false);
  const [liveBox, setLiveBox] = useState<BBox | null>(null);
  const [pendingBox, setPendingBox] = useState<BBox | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [model, setModel] = useState<SamModelStatus | null>(null);
  const [lastTiming, setLastTiming] = useState<SamBoxResponse["timing"] | null>(
    null
  );

  const dragStartRef = useRef<{ image: Point; screen: Point } | null>(null);
  // The tool can be switched off, or the screen unmounted, while a request is
  // in flight; a late resolve must not write to a dead component.
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const refreshModel = useCallback(async () => {
    try {
      const status = await getSamModelStatus();
      if (mountedRef.current) setModel(status);
      return status;
    } catch {
      // A status read that fails is not worth a toast: the tool reports the
      // real problem when the user actually prompts a box.
      return null;
    }
  }, []);

  // Read the weights' state when the tool is first switched on, then keep
  // reading while a download runs so the bar moves.
  useEffect(() => {
    if (!isActive) return undefined;
    void refreshModel();
    return undefined;
  }, [isActive, refreshModel]);

  useEffect(() => {
    if (!isActive) return undefined;
    if (model?.download.status !== "RUNNING") return undefined;
    const timer = window.setInterval(() => void refreshModel(), MODEL_POLL_MS);
    return () => window.clearInterval(timer);
  }, [isActive, model?.download.status, refreshModel]);

  const cancelDrag = useCallback(() => {
    dragStartRef.current = null;
    setLiveBox(null);
  }, []);

  const setActive = useCallback(
    (next: boolean) => {
      setIsActive(next);
      if (!next) cancelDrag();
    },
    [cancelDrag]
  );

  // Leaving the tool, or losing the segmentation, must not strand a rubber band.
  useEffect(() => {
    if (!available) {
      setIsActive(false);
      cancelDrag();
    }
  }, [available, cancelDrag]);

  useEffect(() => {
    cancelDrag();
  }, [segmentationId, cancelDrag]);

  const handleImagePress = useCallback(
    (imagePoint: Point, screenPoint: Point) => {
      if (!isActive || !segmentationId) return;
      registerActivity?.();
      dragStartRef.current = { image: imagePoint, screen: screenPoint };
      setLiveBox(null);
    },
    [isActive, registerActivity, segmentationId]
  );

  const boxFrom = useCallback(
    (imagePoint: Point, screenPoint: Point): BBox | null => {
      const start = dragStartRef.current;
      if (!start) return null;
      const travelled = Math.hypot(
        screenPoint.x - start.screen.x,
        screenPoint.y - start.screen.y
      );
      if (travelled < DRAG_THRESHOLD_PX) return null;
      const bbox: BBox = {
        x0: Math.min(start.image.x, imagePoint.x),
        y0: Math.min(start.image.y, imagePoint.y),
        x1: Math.max(start.image.x, imagePoint.x),
        y1: Math.max(start.image.y, imagePoint.y),
      };
      if (bbox.x1 <= bbox.x0 || bbox.y1 <= bbox.y0) return null;
      return bbox;
    },
    []
  );

  const handleImageDrag = useCallback(
    (imagePoint: Point, screenPoint: Point) => {
      if (!isActive || !dragStartRef.current) return;
      registerActivity?.();
      setLiveBox(boxFrom(imagePoint, screenPoint));
    },
    [boxFrom, isActive, registerActivity]
  );

  const handleImageRelease = useCallback(
    (imagePoint: Point, screenPoint: Point) => {
      if (!isActive || !segmentationId) return;
      const bbox = boxFrom(imagePoint, screenPoint);
      dragStartRef.current = null;
      setLiveBox(null);
      if (!bbox) return;

      // Shown immediately: the request is short but not free, and the user has
      // to see that the drag registered.
      setPendingBox(bbox);
      setIsSubmitting(true);
      registerActivity?.();

      void (async () => {
        try {
          const response = await promptSamBox(segmentationId, bbox);
          if (!mountedRef.current) return;
          setLastTiming(response.timing);
          await onObjectCreated(response);
        } catch (error) {
          if (!mountedRef.current) return;
          onError(
            extractApiErrorMessage(error, "Could not segment that box.")
          );
          // A 409 is almost always "the weights are not here yet"; re-reading
          // the status turns the toolbar into the place that says so.
          void refreshModel();
        } finally {
          if (mountedRef.current) {
            setIsSubmitting(false);
            setPendingBox(null);
          }
        }
      })();
    },
    [
      boxFrom,
      isActive,
      onError,
      onObjectCreated,
      refreshModel,
      registerActivity,
      segmentationId,
    ]
  );

  const downloadModel = useCallback(() => {
    void (async () => {
      try {
        const status = await startSamModelDownload();
        if (mountedRef.current) setModel(status);
      } catch (error) {
        onError(
          extractApiErrorMessage(
            error,
            "Could not start the download. Check this computer's internet connection."
          )
        );
      }
    })();
  }, [onError]);

  const overlays: SegmentOverlay[] = [];
  const live = samLiveBoxOverlay(liveBox);
  if (live) overlays.push(live);
  const pending = samPendingBoxOverlay(pendingBox);
  if (pending) overlays.push(pending);

  return {
    isActive: isActive && available,
    setActive,
    isSubmitting,
    overlays,
    handleImagePress,
    handleImageDrag,
    handleImageRelease,
    cancelDrag,
    model,
    modelReady: Boolean(model?.installed || model?.stub_mode),
    downloadModel,
    lastTiming,
  };
}
