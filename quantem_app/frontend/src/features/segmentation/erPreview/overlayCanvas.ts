import { resolveApiUrl } from "@/shared/api/core/http";

export interface ProbabilityOverlay {
  probData: Uint8ClampedArray; // RGBA from the decoded grayscale prob PNG; R holds prob*255
  width: number;
  height: number;
  bounds: [number, number, number, number]; // [x, y, width, height] in image px
  color: [number, number, number];
  sourceModel: string;
}

type ProbabilityMask = {
  polygons?: Array<{ polygon_coords: Array<[number, number]>; holes?: Array<Array<[number, number]>> }>;
  rectangles?: Array<{ x: number; y: number; width: number; height: number }>;
};

/** Prepare once per mask edit. Slider motion never copies or recolours pixels. */
export function prepareProbabilityImage(overlay: ProbabilityOverlay, mask?: ProbabilityMask): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = overlay.width;
  canvas.height = overlay.height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("2D canvas context unavailable");
  context.putImageData(new ImageData(overlay.probData.slice(), overlay.width, overlay.height), 0, 0);
  eraseMask(canvas, overlay, mask);
  return canvas;
}

/** Decode the grayscale probability PNG (data URL) into raw pixel data. */
export async function decodeProbImage(
  dataUrl: string
): Promise<{ data: Uint8ClampedArray; width: number; height: number }> {
  const img = new Image();
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error("Failed to decode probability image"));
    // Saved-map endpoints are returned as backend-relative paths. In development
    // the UI and API use separate ports, so letting the browser resolve `/api/...`
    // against the page origin loads Vite's HTML fallback instead of the PNG.
    // Opt into CORS before loading from the API so the decoded image can be read
    // back from a canvas without the browser marking it as origin-tainted.
    // Transient ER previews are data URLs and must remain untouched.
    if (dataUrl.startsWith("/")) img.crossOrigin = "anonymous";
    img.src = dataUrl.startsWith("/") ? resolveApiUrl(dataUrl) : dataUrl;
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

function eraseMask(canvas: HTMLCanvasElement, overlay: ProbabilityOverlay, mask?: ProbabilityMask): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2D canvas context unavailable");
  const { width, height } = overlay;
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
}
