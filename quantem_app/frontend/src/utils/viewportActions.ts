import type { ViewportState } from "@/viewer/types";
import type { ViewportAction, ViewportActionResolver } from "@/viewer/viewportSync/viewportSyncStore";

function clamp(value: number, min = 0, max = 1) {
  return Math.min(Math.max(value, min), max);
}

function normalizePoint(x: number, y: number, imageWidth: number, imageHeight: number) {
  return {
    centerX: clamp(x / imageWidth),
    centerY: clamp(y / imageHeight),
  };
}

export function createViewportActionResolver(
  imageWidth: number,
  imageHeight: number
): ViewportActionResolver {
  return (action: ViewportAction, currentViewport: ViewportState | null) => {
    if (action.type === "setViewport") {
      return action.viewport;
    }

    // Carry forward container dimensions from the current viewport when available,
    // otherwise use 0 (calculateViewportBbox falls back to aspect ratio 1).
    const cw = currentViewport?.containerWidth ?? 0;
    const ch = currentViewport?.containerHeight ?? 0;

    if (action.type === "fitToBounds") {
      const padding = action.padding ?? 0;
      const paddedWidth = Math.max(action.width * (1 + padding), 1);
      const paddedHeight = Math.max(action.height * (1 + padding), 1);
      const paddedX = action.x - (paddedWidth - action.width) / 2;
      const paddedY = action.y - (paddedHeight - action.height) / 2;
      const center = normalizePoint(
        paddedX + paddedWidth / 2,
        paddedY + paddedHeight / 2,
        imageWidth,
        imageHeight
      );
      const zoom = Math.min(imageWidth / paddedWidth, imageHeight / paddedHeight);
      return { centerX: center.centerX, centerY: center.centerY, zoom, containerWidth: cw, containerHeight: ch };
    }

    if (action.type === "centerOnPoint") {
      const center = normalizePoint(action.x, action.y, imageWidth, imageHeight);
      const zoom =
        action.zoom ??
        (action.keepZoom ? currentViewport?.zoom : null) ??
        currentViewport?.zoom ??
        1;
      return { centerX: center.centerX, centerY: center.centerY, zoom, containerWidth: cw, containerHeight: ch };
    }

    if (action.type === "setZoom") {
      const centerX = action.centerX ?? currentViewport?.centerX ?? 0.5;
      const centerY = action.centerY ?? currentViewport?.centerY ?? 0.5;
      return { centerX, centerY, zoom: action.zoom, containerWidth: cw, containerHeight: ch };
    }

    if (action.type === "panTo") {
      const zoom =
        action.zoom ??
        (action.keepZoom ? currentViewport?.zoom : null) ??
        currentViewport?.zoom ??
        1;
      return { centerX: action.centerX, centerY: action.centerY, zoom, containerWidth: cw, containerHeight: ch };
    }

    return null;
  };
}

