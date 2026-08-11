export interface ProbabilityOverlay {
  probData: Uint8ClampedArray; // RGBA from the decoded grayscale prob PNG; R holds prob*255
  width: number;
  height: number;
  bounds: [number, number, number, number]; // [x, y, width, height] in image px
  color: [number, number, number];
  sourceModel: string;
}

export type ErProbOverlay = ProbabilityOverlay;

/** Decode the grayscale probability PNG (data URL) into raw pixel data. */
export async function decodeProbImage(
  dataUrl: string
): Promise<{ data: Uint8ClampedArray; width: number; height: number }> {
  const img = new Image();
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error("Failed to decode probability image"));
    img.src = dataUrl;
  });
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2D canvas context unavailable");
  ctx.drawImage(img, 0, 0);
  const id = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return { data: id.data, width: canvas.width, height: canvas.height };
}

/**
 * Colorize the probability map into an RGBA canvas at the given threshold: the
 * model's color where prob >= threshold (alpha ramped by confidence), transparent
 * below. Cheap enough to recompute live on every slider tick — no model re-run.
 */
export function colorizeProb(
  overlay: ProbabilityOverlay,
  threshold: number,
  opacity: number,
  mask?: {
    polygons?: Array<{
      polygon_coords: Array<[number, number]>;
      holes?: Array<Array<[number, number]>>;
    }>;
    rectangles?: Array<{ x: number; y: number; width: number; height: number }>;
  }
): HTMLCanvasElement {
  const { probData, width, height, color } = overlay;
  const out = new Uint8ClampedArray(width * height * 4);
  const t = threshold * 255;
  const denom = Math.max(1e-6, 255 - t);
  for (let i = 0; i < width * height; i++) {
    const p = probData[i * 4]; // R channel holds the probability
    if (p >= t) {
      const j = i * 4;
      out[j] = color[0];
      out[j + 1] = color[1];
      out[j + 2] = color[2];
      const conf = Math.min(1, (p - t) / denom);
      out[j + 3] = Math.round((0.4 + 0.5 * conf) * 255 * opacity);
    }
  }
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2D canvas context unavailable");
  ctx.putImageData(new ImageData(out, width, height), 0, 0);

  const [boundsX, boundsY, boundsWidth, boundsHeight] = overlay.bounds;
  const canvasX = (x: number) => ((x - boundsX) * width) / boundsWidth;
  const canvasY = (y: number) => ((y - boundsY) * height) / boundsHeight;
  ctx.save();
  ctx.globalCompositeOperation = "destination-out";
  for (const polygon of mask?.polygons ?? []) {
    if (polygon.polygon_coords.length < 3) continue;
    ctx.beginPath();
    const traceRing = (ring: Array<[number, number]>) => {
      if (!ring.length) return;
      ctx.moveTo(canvasX(ring[0][0]), canvasY(ring[0][1]));
      for (const [x, y] of ring.slice(1)) ctx.lineTo(canvasX(x), canvasY(y));
      ctx.closePath();
    };
    traceRing(polygon.polygon_coords);
    for (const hole of polygon.holes ?? []) traceRing(hole);
    ctx.fill("evenodd");
  }
  for (const rectangle of mask?.rectangles ?? []) {
    ctx.fillRect(
      canvasX(rectangle.x),
      canvasY(rectangle.y),
      (rectangle.width * width) / boundsWidth,
      (rectangle.height * height) / boundsHeight
    );
  }
  ctx.restore();
  return canvas;
}
