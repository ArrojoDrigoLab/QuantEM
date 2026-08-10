import type { Point } from "@/utils/geometry";
import type { ViewerFitBounds, ViewportState } from "@/viewer/types";

export interface ViewMetrics {
  imageWidth: number;
  imageHeight: number;
  containerWidth: number;
  containerHeight: number;
  visibleWidth: number;
  visibleHeight: number;
  minX: number;
  minY: number;
}

export function nearlyEqualViewport(a: ViewportState | null, b: ViewportState | null, eps = 1e-3) {
  if (!a || !b) return a === b;
  return (
    Math.abs(a.centerX - b.centerX) < eps &&
    Math.abs(a.centerY - b.centerY) < eps &&
    Math.abs(a.zoom - b.zoom) < eps &&
    Math.abs(a.containerWidth - b.containerWidth) < 1 &&
    Math.abs(a.containerHeight - b.containerHeight) < 1
  );
}

/**
 * The viewport an image opens at: the whole image, centred, as large as it fits.
 *
 * `zoom` is defined against the image *width* only -- `buildMetrics` reads
 * `visibleWidth = imageWidth / zoom` and then derives `visibleHeight` from the
 * container's aspect ratio. So `zoom: 1` means "fit the width" and says nothing
 * about the height, which is what this used to return unconditionally. On any
 * container wider than the image's own aspect ratio that crops the top and
 * bottom off the image on open, and on a container much narrower it leaves the
 * image as a strip in an otherwise empty canvas. Either way the first thing a
 * user has to do on every image is zoom by hand before they can see what they
 * came to proofread.
 *
 * Fitting the height as well needs `visibleHeight >= imageHeight`, i.e.
 * `imageWidth / (zoom * containerAspect) >= imageHeight`. Taking the smaller of
 * the two constraints fits whichever dimension is binding and letterboxes the
 * other.
 */
export function defaultViewport(
  imageWidth: number,
  imageHeight: number,
  containerWidth: number,
  containerHeight: number
): ViewportState {
  const safeContainerWidth = Math.max(containerWidth, 1);
  const safeContainerHeight = Math.max(containerHeight, 1);
  const containerAspect = safeContainerWidth / safeContainerHeight;
  const heightLimitedZoom =
    imageWidth / Math.max(imageHeight * containerAspect, 1e-6);
  return {
    centerX: 0.5,
    // Image-height units are `imageWidth`-relative here; see `buildMetrics`,
    // which multiplies both centres by `imageWidth`.
    centerY: imageHeight / imageWidth / 2,
    zoom: Math.min(1, heightLimitedZoom),
    containerWidth: safeContainerWidth,
    containerHeight: safeContainerHeight,
  };
}

export function fitBoundsViewport(config: {
  fitBounds: ViewerFitBounds;
  fitBoundsPaddingRatio: number;
  containerWidth: number;
  containerHeight: number;
  imageWidth: number;
  imageHeight: number;
}): ViewportState {
  const {
    fitBounds,
    fitBoundsPaddingRatio,
    containerWidth,
    containerHeight,
    imageWidth,
    imageHeight,
  } = config;
  const safeContainerWidth = Math.max(containerWidth, 1);
  const safeContainerHeight = Math.max(containerHeight, 1);
  const containerAspect = safeContainerWidth / safeContainerHeight;
  const paddedWidth = Math.min(
    Math.max(fitBounds.width * (1 + fitBoundsPaddingRatio), 1),
    imageWidth
  );
  const paddedHeight = Math.min(
    Math.max(fitBounds.height * (1 + fitBoundsPaddingRatio), 1),
    imageHeight
  );
  const paddedX = Math.min(
    Math.max(0, fitBounds.x - (paddedWidth - fitBounds.width) / 2),
    Math.max(0, imageWidth - paddedWidth)
  );
  const paddedY = Math.min(
    Math.max(0, fitBounds.y - (paddedHeight - fitBounds.height) / 2),
    Math.max(0, imageHeight - paddedHeight)
  );

  return {
    centerX: (paddedX + paddedWidth / 2) / imageWidth,
    centerY: (paddedY + paddedHeight / 2) / imageWidth,
    zoom: Math.min(
      imageWidth / paddedWidth,
      imageWidth / Math.max(paddedHeight * containerAspect, 1e-6)
    ),
    containerWidth: safeContainerWidth,
    containerHeight: safeContainerHeight,
  };
}

export function buildMetrics(
  viewport: ViewportState,
  imageWidth: number,
  imageHeight: number
): ViewMetrics {
  const containerWidth = Math.max(viewport.containerWidth, 1);
  const containerHeight = Math.max(viewport.containerHeight, 1);
  const containerAspect = containerWidth / containerHeight;
  const zoom = Math.max(viewport.zoom, 1e-6);
  const visibleWidth = imageWidth / zoom;
  const visibleHeight = visibleWidth / containerAspect;
  const centerPxX = viewport.centerX * imageWidth;
  const centerPxY = viewport.centerY * imageWidth;
  return {
    imageWidth,
    imageHeight,
    containerWidth,
    containerHeight,
    visibleWidth,
    visibleHeight,
    minX: centerPxX - visibleWidth / 2,
    minY: centerPxY - visibleHeight / 2,
  };
}

export function screenToImagePoint(metrics: ViewMetrics, screenPoint: Point): Point {
  return {
    x: metrics.minX + (screenPoint.x / metrics.containerWidth) * metrics.visibleWidth,
    y: metrics.minY + (screenPoint.y / metrics.containerHeight) * metrics.visibleHeight,
  };
}

export function computeDeckViewState(
  viewport: ViewportState,
  imageWidth: number
): { target: [number, number, number]; zoom: number } {
  const scale = Math.max((viewport.zoom * viewport.containerWidth) / imageWidth, 1e-6);
  const deckZoom = Math.log2(scale);
  return {
    target: [viewport.centerX * imageWidth, viewport.centerY * imageWidth, 0],
    zoom: deckZoom,
  };
}
