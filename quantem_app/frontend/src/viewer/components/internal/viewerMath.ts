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

/**
 * Keep the image on the canvas.
 *
 * Nothing bounded the pan: a drag wrote `centerX`/`centerY` straight through,
 * so a few flicks of the wrist left a black canvas with the image somewhere off
 * to one side and no indication of which way to drag back. Recovering meant
 * reloading the screen.
 *
 * The rule is the smallest one that cannot fail: the point at the centre of the
 * canvas must stay inside the image. So the middle pixel of the viewport is
 * always over image data at every zoom, and the image can never be pushed
 * entirely out of view. Both centres are stored in `imageWidth`-relative units
 * (see `buildMetrics`), which is why the vertical bound is
 * `imageHeight / imageWidth` and not 1.
 *
 * Returns the input object unchanged when it is already in bounds, so the
 * common case does not churn a `useState` reference.
 */
export function clampViewportToImage(
  viewport: ViewportState,
  imageWidth: number,
  imageHeight: number
): ViewportState {
  if (!(imageWidth > 0) || !(imageHeight > 0)) return viewport;
  const maxCenterY = imageHeight / imageWidth;
  const centerX = Math.min(Math.max(viewport.centerX, 0), 1);
  const centerY = Math.min(Math.max(viewport.centerY, 0), maxCenterY);
  if (centerX === viewport.centerX && centerY === viewport.centerY) return viewport;
  return { ...viewport, centerX, centerY };
}

/**
 * The zoom at which one image pixel covers exactly one CSS pixel.
 *
 * `buildMetrics` reads `visibleWidth = imageWidth / zoom` across
 * `containerWidth` screen pixels, so 1:1 is `zoom = imageWidth / containerWidth`.
 */
export function oneToOneZoom(imageWidth: number, containerWidth: number): number {
  return imageWidth / Math.max(containerWidth, 1);
}

export interface ScaleBarPlan {
  /** Width of the drawn bar, in CSS pixels. */
  lengthPx: number;
  /** The physical length the bar stands for, already unit-formatted. */
  label: string;
  /** The same length in nanometres, for tests and for the accessible name. */
  lengthNm: number;
}

const SCALE_BAR_STEPS = [1, 2, 5];

function formatPhysicalLength(nm: number): string {
  if (nm >= 1e6) {
    return `${trimNumber(nm / 1e6)} mm`;
  }
  if (nm >= 1e3) {
    return `${trimNumber(nm / 1e3)} µm`;
  }
  return `${trimNumber(nm)} nm`;
}

function trimNumber(value: number): string {
  const rounded = Math.round(value * 1000) / 1000;
  return String(rounded);
}

/**
 * Pick a round physical length that fits inside `maxLengthPx` and say how wide
 * it is on screen.
 *
 * A scale bar is only worth drawing if it is honest, so this returns `null`
 * when the image has no pixel size rather than inventing one. The length is
 * always 1, 2 or 5 times a power of ten so the number under the bar is one a
 * reader can hold in their head.
 */
export function scaleBarPlan(
  metrics: ViewMetrics,
  pixelSizeNm: number | null | undefined,
  maxLengthPx: number
): ScaleBarPlan | null {
  if (pixelSizeNm == null || !Number.isFinite(pixelSizeNm) || pixelSizeNm <= 0) return null;
  if (!(maxLengthPx > 0)) return null;
  const nmPerScreenPixel = (metrics.visibleWidth / Math.max(metrics.containerWidth, 1)) * pixelSizeNm;
  if (!Number.isFinite(nmPerScreenPixel) || nmPerScreenPixel <= 0) return null;

  const maxLengthNm = nmPerScreenPixel * maxLengthPx;
  const exponent = Math.floor(Math.log10(maxLengthNm));
  let lengthNm = 0;
  for (let power = exponent + 1; power >= exponent - 1; power -= 1) {
    for (let index = SCALE_BAR_STEPS.length - 1; index >= 0; index -= 1) {
      const candidate = SCALE_BAR_STEPS[index] * 10 ** power;
      if (candidate <= maxLengthNm) {
        lengthNm = candidate;
        break;
      }
    }
    if (lengthNm > 0) break;
  }
  if (lengthNm <= 0) return null;

  return {
    lengthPx: lengthNm / nmPerScreenPixel,
    label: formatPhysicalLength(lengthNm),
    lengthNm,
  };
}
