/**
 * Utilities for viewport calculations and transformations.
 */

import type { ViewportState } from "@/viewer/types";

export interface ViewportBbox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

/**
 * Calculate viewport bbox in image coordinates from the shared Viv viewport state.
 *
 * The app stores centerX/centerY normalized to image-width units:
 * - x spans [0, 1] across the image width
 * - y spans [0, imageHeight / imageWidth] using the same unit scale
 * - zoom=1 means the full image width fits in the viewport
 */
export function calculateViewportBbox(
  viewport: ViewportState | null,
  imageWidth: number,
  imageHeight: number
): ViewportBbox | undefined {
  if (!viewport) return undefined;
  if (imageWidth <= 0 || imageHeight <= 0) return undefined;

  // Shared viewport coordinates: x in [0, 1], y in [0, imageHeight/imageWidth]
  // Convert to image pixel coordinates: multiply both by imageWidth
  const clampedCenterX = Math.min(Math.max(viewport.centerX, 0), 1);
  const centerPixelX = clampedCenterX * imageWidth;
  // centerY uses the same scale as centerX (imageWidth = 1.0 in viewport units),
  // so multiply by imageWidth, NOT imageHeight. Do not clamp to [0,1] since
  // centerY can exceed 0.5 for non-square images (range is 0 to imageHeight/imageWidth).
  const centerPixelY = viewport.centerY * imageWidth;

  // Visible area in image pixels
  // At zoom=1, visible width in viewport coords = 1 (i.e., full image width)
  const viewportWidthPixels = imageWidth / viewport.zoom;

  // Visible height depends on the viewer container's aspect ratio, not the image aspect ratio
  const containerAspect =
    viewport.containerWidth && viewport.containerHeight
      ? viewport.containerWidth / viewport.containerHeight
      : 1;
  const viewportHeightPixels = viewportWidthPixels / containerAspect;

  // Add padding (20% of viewport size) to ensure we get segments near edges
  const paddingX = viewportWidthPixels * 0.2;
  const paddingY = viewportHeightPixels * 0.2;

  const rawMinX = centerPixelX - viewportWidthPixels / 2 - paddingX;
  const rawMinY = centerPixelY - viewportHeightPixels / 2 - paddingY;
  const rawMaxX = centerPixelX + viewportWidthPixels / 2 + paddingX;
  const rawMaxY = centerPixelY + viewportHeightPixels / 2 + paddingY;

  const x_min = Math.max(0, Math.min(rawMinX, rawMaxX));
  const y_min = Math.max(0, Math.min(rawMinY, rawMaxY));
  const x_max = Math.min(imageWidth, Math.max(rawMinX, rawMaxX));
  const y_max = Math.min(imageHeight, Math.max(rawMinY, rawMaxY));

  if (x_min >= x_max || y_min >= y_max) {
    return undefined;
  }

  return { x_min, y_min, x_max, y_max };
}
