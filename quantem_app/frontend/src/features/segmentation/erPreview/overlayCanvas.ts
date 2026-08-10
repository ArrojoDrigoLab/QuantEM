export interface ErProbOverlay {
  probData: Uint8ClampedArray; // RGBA from the decoded grayscale prob PNG; R holds prob*255
  width: number;
  height: number;
  bounds: [number, number, number, number]; // [x, y, width, height] in image px
  color: [number, number, number];
  sourceModel: string;
}

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
  overlay: ErProbOverlay,
  threshold: number,
  opacity: number
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
  return canvas;
}
