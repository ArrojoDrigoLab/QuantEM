/**
 * Geometry utility functions for polygon operations.
 */

export interface Point {
  x: number;
  y: number;
}

export interface RectBounds {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/**
 * Ramer-Douglas-Peucker polygon simplification for closed rings.
 * @param points Array of points to simplify
 * @param tolerance Maximum distance a point can be from a line segment in px
 * @returns Simplified closed polygon
 */
export function simplifyPolygon(
  points: Point[],
  tolerance: number = 1.0
): Point[] {
  if (points.length <= 2) {
    return points.slice();
  }

  const isClosed = pointsEqual(points[0], points[points.length - 1]);
  const ring = (isClosed ? points.slice(0, -1) : points.slice()).filter(
    isFinitePoint
  );
  if (ring.length < 3) {
    return points.slice();
  }

  const epsilon = Math.max(0, tolerance);
  if (epsilon === 0) {
    return [...ring, ring[0]];
  }

  const [anchorA, anchorB] = findRingAnchors(ring);
  const forwardPath = collectRingPath(ring, anchorA, anchorB);
  const reversePath = collectRingPath(ring, anchorB, anchorA);

  const simplifiedForward = simplifyPolyline(forwardPath, epsilon);
  const simplifiedReverse = simplifyPolyline(reversePath, epsilon);
  const merged = dedupeConsecutivePoints([
    ...simplifiedForward.slice(0, -1),
    ...simplifiedReverse.slice(0, -1),
  ]);

  const closed = merged.length >= 3 ? merged : ring;
  return [...closed, closed[0]];
}

function simplifyPolyline(points: Point[], tolerance: number): Point[] {
  if (points.length <= 2) {
    return points.slice();
  }

  let maxDist = 0;
  let maxIndex = -1;
  const first = points[0];
  const last = points[points.length - 1];

  for (let i = 1; i < points.length - 1; i += 1) {
    const dist = pointToLineDistance(points[i], first, last);
    if (dist > maxDist) {
      maxDist = dist;
      maxIndex = i;
    }
  }

  if (maxIndex >= 0 && maxDist > tolerance) {
    const left = simplifyPolyline(points.slice(0, maxIndex + 1), tolerance);
    const right = simplifyPolyline(points.slice(maxIndex), tolerance);
    return [...left.slice(0, -1), ...right];
  }

  return [first, last];
}

function findRingAnchors(points: Point[]): [number, number] {
  let minX = 0;
  let maxX = 0;
  let minY = 0;
  let maxY = 0;

  for (let i = 1; i < points.length; i += 1) {
    if (points[i].x < points[minX].x) minX = i;
    if (points[i].x > points[maxX].x) maxX = i;
    if (points[i].y < points[minY].y) minY = i;
    if (points[i].y > points[maxY].y) maxY = i;
  }

  if (minX !== maxX) {
    return [minX, maxX];
  }
  if (minY !== maxY) {
    return [minY, maxY];
  }

  return [0, Math.floor(points.length / 2)];
}

function collectRingPath(points: Point[], start: number, end: number): Point[] {
  const path: Point[] = [];
  let index = start;
  const maxSteps = points.length + 1;
  let steps = 0;

  path.push(points[index]);
  while (index !== end && steps < maxSteps) {
    index = (index + 1) % points.length;
    path.push(points[index]);
    steps += 1;
  }

  return path;
}

function dedupeConsecutivePoints(points: Point[]): Point[] {
  if (points.length <= 1) {
    return points.slice();
  }

  const deduped: Point[] = [points[0]];
  for (let i = 1; i < points.length; i += 1) {
    if (!pointsEqual(points[i], deduped[deduped.length - 1])) {
      deduped.push(points[i]);
    }
  }
  return deduped;
}

function pointsEqual(left: Point, right: Point): boolean {
  return left.x === right.x && left.y === right.y;
}

function isFinitePoint(point: Point): boolean {
  return Number.isFinite(point.x) && Number.isFinite(point.y);
}

/**
 * Calculate the perpendicular distance from a point to a line segment.
 */
function pointToLineDistance(
  point: Point,
  lineStart: Point,
  lineEnd: Point
): number {
  const A = point.x - lineStart.x;
  const B = point.y - lineStart.y;
  const C = lineEnd.x - lineStart.x;
  const D = lineEnd.y - lineStart.y;

  const dot = A * C + B * D;
  const lenSq = C * C + D * D;
  let param = -1;

  if (lenSq !== 0) {
    param = dot / lenSq;
  }

  let xx: number, yy: number;

  if (param < 0) {
    xx = lineStart.x;
    yy = lineStart.y;
  } else if (param > 1) {
    xx = lineEnd.x;
    yy = lineEnd.y;
  } else {
    xx = lineStart.x + param * C;
    yy = lineStart.y + param * D;
  }

  const dx = point.x - xx;
  const dy = point.y - yy;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Check if a point is inside a polygon using ray casting algorithm.
 */
export function pointInPolygon(
  point: Point,
  polygon: Point[]
): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;

    const intersect =
      yi > point.y !== yj > point.y &&
      point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function normalizeRect(rect: RectBounds): RectBounds {
  return {
    x0: Math.min(rect.x0, rect.x1),
    y0: Math.min(rect.y0, rect.y1),
    x1: Math.max(rect.x0, rect.x1),
    y1: Math.max(rect.y0, rect.y1),
  };
}

function pointInRect(point: Point, rect: RectBounds): boolean {
  return (
    point.x >= rect.x0 &&
    point.x <= rect.x1 &&
    point.y >= rect.y0 &&
    point.y <= rect.y1
  );
}

function orientation(a: Point, b: Point, c: Point): number {
  return (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y);
}

function onSegment(a: Point, b: Point, c: Point): boolean {
  return (
    Math.min(a.x, c.x) <= b.x &&
    b.x <= Math.max(a.x, c.x) &&
    Math.min(a.y, c.y) <= b.y &&
    b.y <= Math.max(a.y, c.y)
  );
}

function segmentsIntersect(p1: Point, q1: Point, p2: Point, q2: Point): boolean {
  const o1 = orientation(p1, q1, p2);
  const o2 = orientation(p1, q1, q2);
  const o3 = orientation(p2, q2, p1);
  const o4 = orientation(p2, q2, q1);

  if (o1 * o2 < 0 && o3 * o4 < 0) return true;

  if (o1 === 0 && onSegment(p1, p2, q1)) return true;
  if (o2 === 0 && onSegment(p1, q2, q1)) return true;
  if (o3 === 0 && onSegment(p2, p1, q2)) return true;
  if (o4 === 0 && onSegment(p2, q1, q2)) return true;

  return false;
}

function polygonBounds(polygon: Point[]) {
  const xs = polygon.map((point) => point.x);
  const ys = polygon.map((point) => point.y);
  return {
    x0: Math.min(...xs),
    y0: Math.min(...ys),
    x1: Math.max(...xs),
    y1: Math.max(...ys),
  };
}

/**
 * Returns true when polygon and rectangle overlap, touch edges, or one contains the other.
 */
export function polygonIntersectsRect(
  polygon: Point[],
  rectInput: RectBounds
): boolean {
  if (!polygon.length) return false;

  const rect = normalizeRect(rectInput);
  const polyBox = polygonBounds(polygon);
  if (
    polyBox.x1 < rect.x0 ||
    polyBox.x0 > rect.x1 ||
    polyBox.y1 < rect.y0 ||
    polyBox.y0 > rect.y1
  ) {
    return false;
  }

  for (const point of polygon) {
    if (pointInRect(point, rect)) return true;
  }

  const rectPoints: Point[] = [
    { x: rect.x0, y: rect.y0 },
    { x: rect.x1, y: rect.y0 },
    { x: rect.x1, y: rect.y1 },
    { x: rect.x0, y: rect.y1 },
  ];

  for (const point of rectPoints) {
    if (pointInPolygon(point, polygon)) return true;
  }

  const polyVertices =
    polygon.length >= 2 &&
    polygon[0].x === polygon[polygon.length - 1].x &&
    polygon[0].y === polygon[polygon.length - 1].y
      ? polygon.slice(0, -1)
      : polygon;
  if (polyVertices.length < 2) return false;

  const rectEdges: Array<[Point, Point]> = [
    [rectPoints[0], rectPoints[1]],
    [rectPoints[1], rectPoints[2]],
    [rectPoints[2], rectPoints[3]],
    [rectPoints[3], rectPoints[0]],
  ];

  for (let index = 0; index < polyVertices.length; index += 1) {
    const a = polyVertices[index];
    const b = polyVertices[(index + 1) % polyVertices.length];
    for (const [r0, r1] of rectEdges) {
      if (segmentsIntersect(a, b, r0, r1)) return true;
    }
  }

  return false;
}







