import type { Point } from "@/utils/geometry";

export interface RoiRectangle {
  x: number;
  y: number;
  width: number;
  height: number;
}

export type RoiEditHandle =
  | "north-west"
  | "north"
  | "north-east"
  | "east"
  | "south-east"
  | "south"
  | "south-west"
  | "west"
  | "move";

export const MIN_EDITED_ROI_SIZE = 32;

/**
 * A source-pixel hit target sized relative to the ROI. The pending ROI is
 * normally shown with useful image context, so 3.5% gives an approximately
 * 12-16 screen-pixel target without making small ROIs difficult to resize.
 */
export function roiEditHandleRadius(bounds: RoiRectangle): number {
  return Math.max(8, Math.min(32, Math.min(bounds.width, bounds.height) * 0.035));
}

export function roiEditHandleCenters(
  bounds: RoiRectangle
): Array<{ handle: Exclude<RoiEditHandle, "move">; point: Point }> {
  const left = bounds.x;
  const centerX = bounds.x + bounds.width / 2;
  const right = bounds.x + bounds.width;
  const top = bounds.y;
  const centerY = bounds.y + bounds.height / 2;
  const bottom = bounds.y + bounds.height;

  return [
    { handle: "north-west", point: { x: left, y: top } },
    { handle: "north", point: { x: centerX, y: top } },
    { handle: "north-east", point: { x: right, y: top } },
    { handle: "east", point: { x: right, y: centerY } },
    { handle: "south-east", point: { x: right, y: bottom } },
    { handle: "south", point: { x: centerX, y: bottom } },
    { handle: "south-west", point: { x: left, y: bottom } },
    { handle: "west", point: { x: left, y: centerY } },
  ];
}

/** Return the edge/corner (or interior move target) under a pointer. */
export function resolveRoiEditHandle(
  bounds: RoiRectangle,
  point: Point
): RoiEditHandle | null {
  const radius = roiEditHandleRadius(bounds);
  const left = bounds.x;
  const right = bounds.x + bounds.width;
  const top = bounds.y;
  const bottom = bounds.y + bounds.height;
  const nearLeft = Math.abs(point.x - left) <= radius;
  const nearRight = Math.abs(point.x - right) <= radius;
  const nearTop = Math.abs(point.y - top) <= radius;
  const nearBottom = Math.abs(point.y - bottom) <= radius;

  if (nearLeft && nearTop) return "north-west";
  if (nearRight && nearTop) return "north-east";
  if (nearRight && nearBottom) return "south-east";
  if (nearLeft && nearBottom) return "south-west";
  if (nearTop && point.x >= left && point.x <= right) return "north";
  if (nearRight && point.y >= top && point.y <= bottom) return "east";
  if (nearBottom && point.x >= left && point.x <= right) return "south";
  if (nearLeft && point.y >= top && point.y <= bottom) return "west";

  if (
    point.x >= left &&
    point.x <= right &&
    point.y >= top &&
    point.y <= bottom
  ) {
    return "move";
  }
  return null;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(value, maximum));
}

/** Resize one rectangular edge/corner, or move the rectangle by a drag delta. */
export function updateRoiForDrag(
  start: RoiRectangle,
  handle: RoiEditHandle,
  dragStart: Point,
  point: Point,
  image: { width: number; height: number }
): RoiRectangle {
  if (handle === "move") {
    const x = clamp(
      Math.round(start.x + point.x - dragStart.x),
      0,
      Math.max(0, image.width - start.width)
    );
    const y = clamp(
      Math.round(start.y + point.y - dragStart.y),
      0,
      Math.max(0, image.height - start.height)
    );
    return { ...start, x, y };
  }

  const startRight = start.x + start.width;
  const startBottom = start.y + start.height;
  const minimumWidth = Math.max(1, Math.min(MIN_EDITED_ROI_SIZE, start.width));
  const minimumHeight = Math.max(1, Math.min(MIN_EDITED_ROI_SIZE, start.height));
  const movesLeft = handle.includes("west");
  const movesRight = handle.includes("east");
  const movesTop = handle.includes("north");
  const movesBottom = handle.includes("south");

  const left = movesLeft
    ? clamp(Math.round(point.x), 0, startRight - minimumWidth)
    : start.x;
  const right = movesRight
    ? clamp(Math.round(point.x), start.x + minimumWidth, image.width)
    : startRight;
  const top = movesTop
    ? clamp(Math.round(point.y), 0, startBottom - minimumHeight)
    : start.y;
  const bottom = movesBottom
    ? clamp(Math.round(point.y), start.y + minimumHeight, image.height)
    : startBottom;

  return {
    x: left,
    y: top,
    width: right - left,
    height: bottom - top,
  };
}
