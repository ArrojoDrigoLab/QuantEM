/**
 * Utilities for converting brush strokes into connected binary polygons.
 */

import type { Point } from "@/utils/geometry";

export interface BrushStrokeInput {
  points: Point[];
  size: number;
}

export interface BrushPolygonRings {
  exterior: Point[];
  holes: Point[][];
}

interface StrokeBounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

interface RasterBounds {
  xMin: number;
  yMin: number;
  width: number;
  height: number;
}

interface LabeledComponent {
  id: number;
  pixels: number[];
}

interface Edge {
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  used: boolean;
}

const MIN_BRUSH_RADIUS = 1;
const RASTER_PADDING = 2;
const MAX_RASTER_PIXELS = 8_000_000;

function getStrokeRadius(stroke: BrushStrokeInput): number {
  return Math.max(stroke.size / 2, MIN_BRUSH_RADIUS);
}

function computeStrokeBounds(stroke: BrushStrokeInput): StrokeBounds | null {
  if (stroke.points.length === 0) {
    return null;
  }
  const radius = getStrokeRadius(stroke);
  const xs = stroke.points.map((point) => point.x);
  const ys = stroke.points.map((point) => point.y);
  return {
    minX: Math.min(...xs) - radius,
    minY: Math.min(...ys) - radius,
    maxX: Math.max(...xs) + radius,
    maxY: Math.max(...ys) + radius,
  };
}

function boundsOverlap(left: StrokeBounds, right: StrokeBounds): boolean {
  return !(
    left.maxX < right.minX ||
    right.maxX < left.minX ||
    left.maxY < right.minY ||
    right.maxY < left.minY
  );
}

function polygonArea(points: Point[]): number {
  if (points.length < 4) {
    return 0;
  }
  let area = 0;
  for (let index = 0; index < points.length - 1; index += 1) {
    const current = points[index];
    const next = points[index + 1];
    area += current.x * next.y - next.x * current.y;
  }
  return Math.abs(area) / 2;
}

function pointsEqual(left: Point, right: Point): boolean {
  return left.x === right.x && left.y === right.y;
}

function dedupeConsecutive(points: Point[]): Point[] {
  if (points.length === 0) {
    return [];
  }
  const deduped: Point[] = [];
  for (const point of points) {
    const last = deduped[deduped.length - 1];
    if (!last || !pointsEqual(last, point)) {
      deduped.push(point);
    }
  }
  if (
    deduped.length > 1 &&
    pointsEqual(deduped[0], deduped[deduped.length - 1])
  ) {
    deduped.pop();
  }
  return deduped;
}

function closePolygon(points: Point[]): Point[] {
  if (points.length === 0) {
    return [];
  }
  const closed = [...points];
  if (!pointsEqual(closed[0], closed[closed.length - 1])) {
    closed.push({ x: closed[0].x, y: closed[0].y });
  }
  return closed;
}

function isCollinear(prev: Point, current: Point, next: Point): boolean {
  const cross =
    (current.x - prev.x) * (next.y - current.y) -
    (current.y - prev.y) * (next.x - current.x);
  return Math.abs(cross) < 1e-9;
}

function simplifyClosedPolygon(points: Point[]): Point[] {
  const closed = closePolygon(points);
  if (closed.length < 4) {
    return [];
  }

  let ring = dedupeConsecutive(closed.slice(0, -1));
  if (ring.length < 3) {
    return [];
  }

  let changed = true;
  while (changed && ring.length >= 3) {
    changed = false;
    const reduced: Point[] = [];

    for (let index = 0; index < ring.length; index += 1) {
      const prev = ring[(index - 1 + ring.length) % ring.length];
      const current = ring[index];
      const next = ring[(index + 1) % ring.length];
      if (isCollinear(prev, current, next)) {
        changed = true;
        continue;
      }
      reduced.push(current);
    }

    ring = dedupeConsecutive(reduced);
  }

  if (ring.length < 3) {
    return [];
  }

  return [...ring, { x: ring[0].x, y: ring[0].y }];
}

function comparePoints(left: Point, right: Point): number {
  if (left.x !== right.x) {
    return left.x - right.x;
  }
  return left.y - right.y;
}

function convexHull(points: Point[]): Point[] {
  if (points.length <= 1) {
    return points.slice();
  }

  const sorted = points.slice().sort(comparePoints);
  const unique: Point[] = [];
  for (const point of sorted) {
    const last = unique[unique.length - 1];
    if (!last || !pointsEqual(last, point)) {
      unique.push(point);
    }
  }

  if (unique.length <= 2) {
    return unique;
  }

  const cross = (origin: Point, a: Point, b: Point) =>
    (a.x - origin.x) * (b.y - origin.y) -
    (a.y - origin.y) * (b.x - origin.x);

  const lower: Point[] = [];
  for (const point of unique) {
    while (
      lower.length >= 2 &&
      cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0
    ) {
      lower.pop();
    }
    lower.push(point);
  }

  const upper: Point[] = [];
  for (let index = unique.length - 1; index >= 0; index -= 1) {
    const point = unique[index];
    while (
      upper.length >= 2 &&
      cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0
    ) {
      upper.pop();
    }
    upper.push(point);
  }

  lower.pop();
  upper.pop();
  return [...lower, ...upper];
}

function approximateStrokePolygon(stroke: BrushStrokeInput): Point[] | null {
  if (stroke.points.length === 0) {
    return null;
  }

  const radius = getStrokeRadius(stroke);
  const directions = 16;
  const supportPoints: Point[] = [];

  for (const point of stroke.points) {
    for (let index = 0; index < directions; index += 1) {
      const angle = (index / directions) * Math.PI * 2;
      supportPoints.push({
        x: point.x + Math.cos(angle) * radius,
        y: point.y + Math.sin(angle) * radius,
      });
    }
  }

  const hull = convexHull(supportPoints);
  if (hull.length < 3) {
    return null;
  }
  return [...hull, { x: hull[0].x, y: hull[0].y }];
}

function approximateStrokePieces(stroke: BrushStrokeInput): BrushPolygonRings[] {
  if (stroke.points.length <= 1) {
    const polygon = approximateStrokePolygon(stroke);
    return polygon ? [{ exterior: polygon, holes: [] }] : [];
  }
  // The large-raster fallback must not take a convex hull of an entire bent or
  // closed stroke: doing that to a large brush-drawn ring silently fills its
  // hole. Capsules for each consecutive segment remain bounded in memory and
  // are unioned by the confirmation endpoint, reconstructing the same path
  // topology without allocating the full raster client-side.
  return stroke.points.slice(0, -1).flatMap((point, index) => {
    const polygon = approximateStrokePolygon({
      size: stroke.size,
      points: [point, stroke.points[index + 1]],
    });
    return polygon ? [{ exterior: polygon, holes: [] }] : [];
  });
}

class DisjointSet {
  private readonly parent: number[];

  constructor(size: number) {
    this.parent = Array.from({ length: size }, (_, index) => index);
  }

  find(index: number): number {
    if (this.parent[index] !== index) {
      this.parent[index] = this.find(this.parent[index]);
    }
    return this.parent[index];
  }

  union(left: number, right: number): void {
    const leftRoot = this.find(left);
    const rightRoot = this.find(right);
    if (leftRoot !== rightRoot) {
      this.parent[rightRoot] = leftRoot;
    }
  }
}

function buildStrokeClusters(strokes: BrushStrokeInput[]): BrushStrokeInput[][] {
  if (strokes.length === 0) {
    return [];
  }

  const bounds = strokes.map(computeStrokeBounds);
  const disjointSet = new DisjointSet(strokes.length);

  for (let left = 0; left < strokes.length; left += 1) {
    const leftBounds = bounds[left];
    if (!leftBounds) continue;
    for (let right = left + 1; right < strokes.length; right += 1) {
      const rightBounds = bounds[right];
      if (!rightBounds) continue;
      if (boundsOverlap(leftBounds, rightBounds)) {
        disjointSet.union(left, right);
      }
    }
  }

  const grouped = new Map<number, BrushStrokeInput[]>();
  for (let index = 0; index < strokes.length; index += 1) {
    const root = disjointSet.find(index);
    const group = grouped.get(root) ?? [];
    group.push(strokes[index]);
    grouped.set(root, group);
  }
  return Array.from(grouped.values());
}

function makeRasterBounds(strokes: BrushStrokeInput[]): RasterBounds | null {
  const strokeBounds = strokes
    .map(computeStrokeBounds)
    .filter((bounds): bounds is StrokeBounds => bounds !== null);
  if (strokeBounds.length === 0) {
    return null;
  }

  const minX = Math.min(...strokeBounds.map((bounds) => bounds.minX));
  const minY = Math.min(...strokeBounds.map((bounds) => bounds.minY));
  const maxX = Math.max(...strokeBounds.map((bounds) => bounds.maxX));
  const maxY = Math.max(...strokeBounds.map((bounds) => bounds.maxY));

  const xMin = Math.floor(minX) - RASTER_PADDING;
  const yMin = Math.floor(minY) - RASTER_PADDING;
  const xMax = Math.ceil(maxX) + RASTER_PADDING;
  const yMax = Math.ceil(maxY) + RASTER_PADDING;
  const width = Math.max(1, xMax - xMin);
  const height = Math.max(1, yMax - yMin);

  return { xMin, yMin, width, height };
}

function stampDisk(
  mask: Uint8Array,
  width: number,
  height: number,
  centerX: number,
  centerY: number,
  radius: number
): void {
  const radiusSq = radius * radius;
  const minY = Math.floor(centerY - radius);
  const maxY = Math.ceil(centerY + radius);

  for (let y = minY; y <= maxY; y += 1) {
    if (y < 0 || y >= height) {
      continue;
    }
    const dy = y + 0.5 - centerY;
    const dySq = dy * dy;
    if (dySq > radiusSq) {
      continue;
    }
    const xExtent = Math.sqrt(radiusSq - dySq);
    const startX = Math.ceil(centerX - xExtent - 0.5);
    const endX = Math.floor(centerX + xExtent - 0.5);
    if (endX < 0 || startX >= width) {
      continue;
    }
    const clampedStartX = Math.max(startX, 0);
    const clampedEndX = Math.min(endX, width - 1);
    const rowOffset = y * width;
    for (let x = clampedStartX; x <= clampedEndX; x += 1) {
      mask[rowOffset + x] = 1;
    }
  }
}

function rasterizeStroke(
  mask: Uint8Array,
  width: number,
  height: number,
  xMin: number,
  yMin: number,
  stroke: BrushStrokeInput
): void {
  if (stroke.points.length === 0) {
    return;
  }

  const radius = getStrokeRadius(stroke);
  const points = stroke.points;

  if (points.length === 1) {
    const point = points[0];
    stampDisk(mask, width, height, point.x - xMin, point.y - yMin, radius);
    return;
  }

  const spacing = Math.max(radius * 0.5, 0.75);

  for (let index = 0; index < points.length - 1; index += 1) {
    const start = points[index];
    const end = points[index + 1];
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const length = Math.hypot(dx, dy);
    const steps = Math.max(1, Math.ceil(length / spacing));

    for (let step = 0; step <= steps; step += 1) {
      const t = step / steps;
      const sampleX = start.x + dx * t;
      const sampleY = start.y + dy * t;
      stampDisk(mask, width, height, sampleX - xMin, sampleY - yMin, radius);
    }
  }
}

function labelConnectedComponents(
  mask: Uint8Array,
  width: number,
  height: number
): { labels: Int32Array; components: LabeledComponent[] } {
  const labels = new Int32Array(mask.length);
  const components: LabeledComponent[] = [];
  let componentId = 0;

  for (let index = 0; index < mask.length; index += 1) {
    if (mask[index] === 0 || labels[index] !== 0) {
      continue;
    }

    componentId += 1;
    const queue: number[] = [index];
    const componentPixels: number[] = [];
    labels[index] = componentId;

    for (let queueIndex = 0; queueIndex < queue.length; queueIndex += 1) {
      const current = queue[queueIndex];
      componentPixels.push(current);

      const x = current % width;
      const y = Math.floor(current / width);

      const neighbors = [
        x > 0 ? current - 1 : -1,
        x + 1 < width ? current + 1 : -1,
        y > 0 ? current - width : -1,
        y + 1 < height ? current + width : -1,
      ];

      for (const neighbor of neighbors) {
        if (neighbor < 0) continue;
        if (mask[neighbor] === 0 || labels[neighbor] !== 0) continue;
        labels[neighbor] = componentId;
        queue.push(neighbor);
      }
    }

    components.push({ id: componentId, pixels: componentPixels });
  }

  return { labels, components };
}

function makeVertexKey(x: number, y: number): string {
  return `${x},${y}`;
}

function extractComponentRings(
  component: LabeledComponent,
  labels: Int32Array,
  width: number,
  height: number,
  originX: number,
  originY: number
): BrushPolygonRings | null {
  const edges: Edge[] = [];

  for (const index of component.pixels) {
    const x = index % width;
    const y = Math.floor(index / width);

    const topInside = y > 0 && labels[index - width] === component.id;
    const rightInside = x + 1 < width && labels[index + 1] === component.id;
    const bottomInside =
      y + 1 < height && labels[index + width] === component.id;
    const leftInside = x > 0 && labels[index - 1] === component.id;

    if (!topInside) {
      edges.push({
        startX: x,
        startY: y,
        endX: x + 1,
        endY: y,
        used: false,
      });
    }
    if (!rightInside) {
      edges.push({
        startX: x + 1,
        startY: y,
        endX: x + 1,
        endY: y + 1,
        used: false,
      });
    }
    if (!bottomInside) {
      edges.push({
        startX: x + 1,
        startY: y + 1,
        endX: x,
        endY: y + 1,
        used: false,
      });
    }
    if (!leftInside) {
      edges.push({
        startX: x,
        startY: y + 1,
        endX: x,
        endY: y,
        used: false,
      });
    }
  }

  if (edges.length === 0) {
    return null;
  }

  const edgesByStart = new Map<string, number[]>();
  edges.forEach((edge, index) => {
    const key = makeVertexKey(edge.startX, edge.startY);
    const bucket = edgesByStart.get(key) ?? [];
    bucket.push(index);
    edgesByStart.set(key, bucket);
  });

  const loops: Array<{ points: Point[]; area: number }> = [];
  for (let startIndex = 0; startIndex < edges.length; startIndex += 1) {
    if (edges[startIndex].used) {
      continue;
    }

    const firstEdge = edges[startIndex];
    const firstPoint = {
      x: firstEdge.startX + originX,
      y: firstEdge.startY + originY,
    };
    const firstKey = makeVertexKey(firstEdge.startX, firstEdge.startY);
    const loop: Point[] = [];
    let currentIndex = startIndex;
    let closed = false;

    for (let guard = 0; guard <= edges.length + 1; guard += 1) {
      const currentEdge = edges[currentIndex];
      if (currentEdge.used) {
        break;
      }
      currentEdge.used = true;
      loop.push({
        x: currentEdge.startX + originX,
        y: currentEdge.startY + originY,
      });

      const nextKey = makeVertexKey(currentEdge.endX, currentEdge.endY);
      if (nextKey === firstKey) {
        loop.push({ x: firstPoint.x, y: firstPoint.y });
        closed = true;
        break;
      }

      const candidates = edgesByStart.get(nextKey);
      if (!candidates) {
        break;
      }
      const nextIndex = candidates.find(
        (candidateIndex) => !edges[candidateIndex].used
      );
      if (nextIndex === undefined) {
        break;
      }
      currentIndex = nextIndex;
    }

    if (!closed || loop.length < 4) {
      continue;
    }

    const simplified = simplifyClosedPolygon(loop);
    const area = polygonArea(simplified);
    if (simplified.length >= 4 && area > 0) {
      loops.push({ points: simplified, area });
    }
  }

  if (loops.length === 0) {
    return null;
  }

  loops.sort((left, right) => right.area - left.area);
  return {
    exterior: loops[0].points,
    // Every remaining boundary loop belongs to background enclosed by this
    // connected foreground component. Keeping these rings is what prevents a
    // brush-drawn donut from silently becoming a filled disk on confirmation.
    holes: loops.slice(1).map((loop) => loop.points),
  };
}

function extractPolygonsForCluster(strokes: BrushStrokeInput[]): BrushPolygonRings[] {
  const bounds = makeRasterBounds(strokes);
  if (!bounds) {
    return [];
  }

  if (bounds.width * bounds.height > MAX_RASTER_PIXELS) {
    return strokes.flatMap(approximateStrokePieces);
  }

  const mask = new Uint8Array(bounds.width * bounds.height);
  for (const stroke of strokes) {
    rasterizeStroke(
      mask,
      bounds.width,
      bounds.height,
      bounds.xMin,
      bounds.yMin,
      stroke
    );
  }

  const { labels, components } = labelConnectedComponents(
    mask,
    bounds.width,
    bounds.height
  );

  const polygons: BrushPolygonRings[] = [];
  for (const component of components) {
    const polygon = extractComponentRings(
      component,
      labels,
      bounds.width,
      bounds.height,
      bounds.xMin,
      bounds.yMin
    );
    if (
      polygon &&
      polygon.exterior.length >= 4 &&
      polygonArea(polygon.exterior) > 0
    ) {
      polygons.push(polygon);
    }
  }

  return polygons;
}

export function brushStrokesToConnectedPolygons(
  strokes: BrushStrokeInput[]
): Point[][] {
  return brushStrokesToConnectedPolygonRings(strokes).map(
    (polygon) => polygon.exterior
  );
}

export function brushStrokesToConnectedPolygonRings(
  strokes: BrushStrokeInput[]
): BrushPolygonRings[] {
  const validStrokes = strokes.filter((stroke) => stroke.points.length > 0);
  if (validStrokes.length === 0) {
    return [];
  }

  const clusters = buildStrokeClusters(validStrokes);
  return clusters.flatMap((cluster) => extractPolygonsForCluster(cluster));
}
