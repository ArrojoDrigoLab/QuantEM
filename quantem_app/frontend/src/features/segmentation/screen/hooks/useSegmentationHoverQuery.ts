import { useCallback, useEffect, useRef } from "react";
import { getSegmentsAtPoint } from "@/shared/api/segmentations/annotations";
import { useHoverSelection } from "@/hooks/useHoverSelection";
import { HOVER_QUERY_DEBOUNCE_MS } from "@/features/segmentation/screen/utils/constants";
import type { Point } from "@/utils/geometry";
import type { CellStatus, LabelState } from "@/shared/types/common";
import type { ImageSegmentation } from "@/shared/types/images";
import type { SegmentObject } from "@/shared/types/segmentation";

interface UseSegmentationHoverQueryArgs {
  currentSegmentation: ImageSegmentation | null;
  activeSourceModel: string | null;
}

export type HoverSegmentQuery =
  | LabelState[]
  | {
      states?: LabelState[];
      statuses?: CellStatus[];
    };

export function useSegmentationHoverQuery({
  currentSegmentation,
  activeSourceModel,
}: UseSegmentationHoverQueryArgs) {
  const {
    hoverSegments,
    highlightedSegmentId,
    hoverActionMode,
    hoverPoint,
    setHoverActionMode,
    findSegmentsAtPoint,
    cycleHoverIndex,
    clearHover,
  } = useHoverSelection();

  const hoverQueryRequestRef = useRef(0);
  const hoverQueryDebounceRef = useRef<number | null>(null);
  const hoverQueryAbortControllerRef = useRef<AbortController | null>(null);

  const clearScheduledHoverQuery = useCallback(() => {
    if (hoverQueryDebounceRef.current !== null) {
      window.clearTimeout(hoverQueryDebounceRef.current);
      hoverQueryDebounceRef.current = null;
    }
  }, []);

  const abortInFlightHoverQuery = useCallback(() => {
    if (hoverQueryAbortControllerRef.current !== null) {
      hoverQueryAbortControllerRef.current.abort();
      hoverQueryAbortControllerRef.current = null;
    }
  }, []);

  const cancelPendingHoverQuery = useCallback(() => {
    clearScheduledHoverQuery();
    abortInFlightHoverQuery();
    hoverQueryRequestRef.current += 1;
  }, [abortInFlightHoverQuery, clearScheduledHoverQuery]);

  const clearHoverInteraction = useCallback(() => {
    cancelPendingHoverQuery();
    clearHover();
  }, [cancelPendingHoverQuery, clearHover]);

  useEffect(() => {
    return () => {
      cancelPendingHoverQuery();
    };
  }, [cancelPendingHoverQuery]);

  const scheduleHoverSegmentQuery = useCallback(
    (
      point: Point,
      queryOrStates: HoverSegmentQuery,
      resolveSegments: (segments: SegmentObject[]) => SegmentObject[],
      errorMessage: string
    ) => {
      if (!currentSegmentation) return;

      clearScheduledHoverQuery();
      const requestId = hoverQueryRequestRef.current + 1;
      hoverQueryRequestRef.current = requestId;
      hoverQueryDebounceRef.current = window.setTimeout(() => {
        hoverQueryDebounceRef.current = null;
        abortInFlightHoverQuery();
        const controller = new AbortController();
        hoverQueryAbortControllerRef.current = controller;
        const query = Array.isArray(queryOrStates)
          ? { states: queryOrStates }
          : queryOrStates;
        void getSegmentsAtPoint(
          currentSegmentation.id,
          {
            x: point.x,
            y: point.y,
            ...(query.states ? { states: query.states } : {}),
            ...(query.statuses ? { statuses: query.statuses } : {}),
            ...(activeSourceModel ? { source_model: activeSourceModel } : {}),
          },
          {
            signal: controller.signal,
            geometryMode: "hover",
          }
        )
          .then((result) => {
            if (requestId !== hoverQueryRequestRef.current) return;
            if (hoverQueryAbortControllerRef.current === controller) {
              hoverQueryAbortControllerRef.current = null;
            }
            findSegmentsAtPoint(point, resolveSegments(result));
          })
          .catch((error) => {
            if (hoverQueryAbortControllerRef.current === controller) {
              hoverQueryAbortControllerRef.current = null;
            }
            if (controller.signal.aborted) {
              return;
            }
            if (requestId !== hoverQueryRequestRef.current) return;
            console.error(errorMessage, error);
            clearHoverInteraction();
          });
      }, HOVER_QUERY_DEBOUNCE_MS);
    },
    [
      clearHoverInteraction,
      clearScheduledHoverQuery,
      currentSegmentation,
      activeSourceModel,
      findSegmentsAtPoint,
      abortInFlightHoverQuery,
    ]
  );

  return {
    hoverSegments,
    highlightedSegmentId,
    hoverActionMode,
    hoverPoint,
    setHoverActionMode,
    cycleHoverIndex,
    clearHoverInteraction,
    scheduleHoverSegmentQuery,
  };
}
