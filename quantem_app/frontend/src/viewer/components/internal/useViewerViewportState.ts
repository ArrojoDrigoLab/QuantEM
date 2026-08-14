import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ViewerFitBounds, ViewportState } from "@/viewer/types";
import {
  buildMetrics,
  clampViewportToImage,
  computeDeckViewState,
  defaultViewport,
  fitBoundsViewport,
  nearlyEqualViewport,
  oneToOneZoom,
} from "@/viewer/components/internal/viewerMath";

const VIEWER_ZOOM_STEP = 1.25;
const MIN_VIEWER_ZOOM = 0.05;
const MAX_VIEWER_ZOOM = 200;

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
  /**
   * The view this image opened at, for Reset.
   *
   * Captured once, from the first viewport the screen actually settled on --
   * which is not always the fit view: arriving from a deep link to an ROI opens
   * fitted to that ROI, and "back to where I started" has to mean that, not the
   * whole image.
   */
  const openingViewportRef = useRef<ViewportState | null>(null);
  useEffect(() => {
    if (openingViewportRef.current != null || !localViewport) return;
    openingViewportRef.current = localViewport;
  }, [localViewport]);

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
      // Every writer goes through here -- pan, wheel, double-click, the two
      // panels' viewport sync -- so the clamp only has to be true in one place
      // for the image to be impossible to lose.
      const clamped = clampViewportToImage(
        nextViewport,
        resolvedImageWidth,
        resolvedImageHeight
      );
      setLocalViewport(clamped);
      if (emit) {
        emitViewportChange(clamped);
      }
    },
    [emitViewportChange, resolvedImageHeight, resolvedImageWidth]
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

  /** Show the whole image, as large as it fits: the view an image opens at. */
  const fitImage = useCallback(() => {
    setViewport(
      defaultViewport(
        resolvedImageWidth,
        resolvedImageHeight,
        effectiveViewport.containerWidth,
        effectiveViewport.containerHeight
      ),
      true
    );
  }, [effectiveViewport, resolvedImageHeight, resolvedImageWidth, setViewport]);

  /** One image pixel per screen pixel, keeping the current centre. */
  const zoomOneToOne = useCallback(() => {
    setViewport(
      {
        ...effectiveViewport,
        zoom: oneToOneZoom(resolvedImageWidth, effectiveViewport.containerWidth),
      },
      true
    );
  }, [effectiveViewport, resolvedImageWidth, setViewport]);

  /** Zoom around the current centre in small, predictable button-sized steps. */
  const zoomBy = useCallback(
    (factor: number) => {
      const nextZoom = Math.max(
        MIN_VIEWER_ZOOM,
        Math.min(MAX_VIEWER_ZOOM, effectiveViewport.zoom * factor)
      );
      if (nextZoom === effectiveViewport.zoom) return;
      setViewport({ ...effectiveViewport, zoom: nextZoom }, true);
    },
    [effectiveViewport, setViewport]
  );

  const zoomIn = useCallback(() => zoomBy(VIEWER_ZOOM_STEP), [zoomBy]);
  const zoomOut = useCallback(() => zoomBy(1 / VIEWER_ZOOM_STEP), [zoomBy]);

  /** Back to the view this image opened at. */
  const resetView = useCallback(() => {
    const opening = openingViewportRef.current;
    setViewport(
      opening
        ? {
            ...opening,
            containerWidth: effectiveViewport.containerWidth,
            containerHeight: effectiveViewport.containerHeight,
          }
        : defaultViewport(
            resolvedImageWidth,
            resolvedImageHeight,
            effectiveViewport.containerWidth,
            effectiveViewport.containerHeight
          ),
      true
    );
  }, [effectiveViewport, resolvedImageHeight, resolvedImageWidth, setViewport]);

  return {
    localViewport: effectiveViewport,
    setViewport,
    metrics,
    deckViewState,
    fitImage,
    zoomOneToOne,
    zoomIn,
    zoomOut,
    resetView,
  };
}
