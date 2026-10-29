import { parseHexColor } from "@/viewer/components/internal/vivUtils";

export interface TintedLutResult {
  /** Flat RGBA8 palette indexed by dense label (length = (maxLabel + 1) * 4). */
  rgba: Uint8Array;
  /** Highest label seen across the supplied objects. */
  maxLabel: number;
}

/**
 * Builds a flat RGBA8 colour LUT (indexed by dense label) that tints every
 * object whose `state` is in `visibleStates` with `color` at full alpha, and
 * leaves all other labels fully transparent (alpha 0). Used for the read-only
 * viewer where each segmentation shows its confirmed objects in a single
 * user-chosen colour.
 */
export function buildTintedLut(
  objects: { label: number; state: string; hidden_by_completed_roi?: boolean }[],
  color: string,
  visibleStates: Set<string>,
  options: { overlay_kind?: string; max_label?: number } = {}
): TintedLutResult {
  let maxLabel = Math.max(0, options.max_label ?? 0, options.overlay_kind === "binary_mask" ? 1 : 0);
  for (const object of objects) {
    if (object.label > maxLabel) maxLabel = object.label;
  }

  const rgba = new Uint8Array((maxLabel + 1) * 4);
  const [r, g, b] = parseHexColor(color);
  if (options.overlay_kind === "binary_mask" && visibleStates.has("confirmed")) {
    rgba.set([r, g, b, 255], 4);
  }

  for (const object of objects) {
    if (!visibleStates.has(object.state) || object.hidden_by_completed_roi) continue;
    if (object.label <= 0 || object.label > maxLabel) continue;
    const offset = object.label * 4;
    rgba[offset] = r;
    rgba[offset + 1] = g;
    rgba[offset + 2] = b;
    rgba[offset + 3] = 255;
  }

  return { rgba, maxLabel };
}
