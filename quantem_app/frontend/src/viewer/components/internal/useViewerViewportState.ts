import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ViewerFitBounds, ViewportState } from "@/viewer/types";
import {
  buildMetrics,
  computeDeckViewState,
  defaultViewport,
  fitBoundsViewport,
  nearlyEqualViewport,
} from "@/viewer/components/internal/viewerMath";

export function useViewerViewportState(config: {
  viewportState?: ViewportState;
  initialViewport?: ViewportState;
  onViewportChange?: (viewport: ViewportState) => void;
  fitBounds?: ViewerFitBounds | null;
  fitBoundsKey?: string | null;
  fitBoundsPaddingRatio: number;
  containerSize: { width: number; height: number };
  resolvedImageWidth: number;
  resolvedImageHeight: number;
}) {
  const {
    viewportState,
    initialViewport,
    onViewportChange,
    fitBounds,
    fitBoundsKey,
    fitBoundsPaddingRatio,
    containerSize,
    resolvedImageWidth,
    resolvedImageHeight,
  } = config;
  const lastFitKeyRef = useRef<string | null>(null);
  const didApplyInitialViewportRef = useRef(false);
  const [localViewport, setLocalViewport] = useState<ViewportState | null>(null);

  useEffect(() => {
    if (containerSize.width <= 0 || containerSize.height <= 0) return;
    const nextDefault = defaultViewport(
      resolvedImageWidth,
      resolvedImageHeight,
      containerSize.width,
      containerSize.height
    );
    setLocalViewport((prev) => {
      if (!prev) return nextDefault;
      if (
        Math.abs(prev.containerWidth - containerSize.width) > 0.5 ||
        Math.abs(prev.containerHeight - containerSize.height) > 0.5
      ) {
        return {
          ...prev,
          containerWidth: containerSize.width,
          containerHeight: containerSize.height,
        };
      }
      return prev;
    });
  }, [containerSize.height, containerSize.width, resolvedImageHeight, resolvedImageWidth]);

  useEffect(() => {
    if (!viewportState) return;
    setLocalViewport((prev) => {
      const localContainerWidth =
        prev?.containerWidth ??
        (containerSize.width > 0 ? containerSize.width : viewportState.containerWidth);
      const localContainerHeight =
        prev?.containerHeight ??
        (containerSize.height > 0 ? containerSize.height : viewportState.containerHeight);
      const nextViewport: ViewportState = {
        ...viewportState,
        containerWidth: localContainerWidth,
        containerHeight: localContainerHeight,
      };
      if (nearlyEqualViewport(prev, nextViewport)) return prev;
      return nextViewport;
    });
  }, [containerSize.height, containerSize.width, viewportState]);

  useEffect(() => {
    if (didApplyInitialViewportRef.current || !initialViewport) return;
    didApplyInitialViewportRef.current = true;
    setLocalViewport(initialViewport);
  }, [initialViewport]);

  const emitViewportChange = useCallback(
    (nextViewport: ViewportState) => {
      onViewportChange?.(nextViewport);
    },
    [onViewportChange]
  );

  const setViewport = useCallback(
    (nextViewport: ViewportState, emit = true) => {
      setLocalViewport(nextViewport);
      if (emit) {
        emitViewportChange(nextViewport);
      }
    },
    [emitViewportChange]
  );

  useEffect(() => {
    if (!fitBounds || !localViewport) return;
    if (fitBoundsKey && lastFitKeyRef.current === fitBoundsKey) return;

    const nextViewport: ViewportState = fitBoundsViewport({
      fitBounds,
      fitBoundsPaddingRatio,
      containerWidth: localViewport.containerWidth,
      containerHeight: localViewport.containerHeight,
      imageWidth: resolvedImageWidth,
      imageHeight: resolvedImageHeight,
    });
    setViewport(nextViewport, true);
    lastFitKeyRef.current = fitBoundsKey ?? "default";
  }, [
    fitBounds,
    fitBoundsKey,
    fitBoundsPaddingRatio,
    localViewport,
    resolvedImageHeight,
    resolvedImageWidth,
    setViewport,
  ]);

  /**
   * What to draw before the container has been measured.
   *
   * The fallback used to be `{ target: [0, 0, 0], zoom: 0 }`, which is not a
   * neutral placeholder: it puts the image's *top-left corner* at the centre of
   * the canvas and draws it at one world unit per pixel, so three quadrants are
   * empty black and the image is at whatever fixed scale that happens to be.
   * Any frame rendered before the ResizeObserver reports -- and any state where
   * it never does -- looks exactly like a viewer that failed to open the image.
   * Fitting the image to whatever size is known instead is right on the first
   * frame and degrades to a centred square when nothing is known at all.
   */
  const effectiveViewport = useMemo(
    () =>
      localViewport ??
      defaultViewport(
        resolvedImageWidth,
        resolvedImageHeight,
        containerSize.width,
        containerSize.height
      ),
    [
      containerSize.height,
      containerSize.width,
      localViewport,
      resolvedImageHeight,
      resolvedImageWidth,
    ]
  );

  const metrics = useMemo(
    () => buildMetrics(effectiveViewport, resolvedImageWidth, resolvedImageHeight),
    [effectiveViewport, resolvedImageHeight, resolvedImageWidth]
  );

  const deckViewState = useMemo(
    () => computeDeckViewState(effectiveViewport, resolvedImageWidth),
    [effectiveViewport, resolvedImageWidth]
  );

  return { localViewport: effectiveViewport, setViewport, metrics, deckViewState };
}
