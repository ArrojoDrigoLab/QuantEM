/**
 * What counts as an importable file, and how the panel talks about one.
 *
 * Split out of `ImageUploadPanel.tsx` unchanged. Three checks decide whether a
 * file joins the queue — the extension (`accept` is a hint the picker may
 * honour and a drop ignores entirely), the server's size limit (unknowable to
 * the file dialog), and whether it is already queued — and they are the same
 * three whether the file arrived by picker or by drop. They are here so that
 * the rules can be changed without opening the panel, and so the panel's own
 * file is about state and layout.
 */

import type { FileDeclaredPixelSize } from "@/shared/fileDeclaredPixelSize";

/**
 * Used only until the first `/api/system/status/` response lands (or if it
 * fails). Kept minimal on purpose: it is a stopgap, not a second source of
 * truth.
 */
export const FALLBACK_UPLOAD_FORMATS = [".tif", ".tiff", ".png"];

/** One file waiting to become an import. */
export interface ChosenFile {
  /**
   * Identity for React and for the per-file state map.
   *
   * Not the filename: two folders can hold `Grid1.tif`, and a user who removes
   * one row must not remove the other's progress with it.
   */
  key: string;
  file: File;
  /** What the file's own header declares. `null` while it is still being read. */
  scale: FileDeclaredPixelSize | null;
}

/**
 * Where one file has got to.
 *
 * `failed` carries the server's own sentence. It is per file because a batch
 * fails per file: one corrupt TIFF in a plate of forty is a fact about that
 * TIFF, and collapsing it into a single form-level error would hide which
 * image the user has to look at.
 */
export type FileImportState =
  | { kind: "waiting" }
  | { kind: "uploading" }
  | { kind: "imported" }
  | { kind: "failed"; message: string };

/** What the last batch did, once it has finished. */
export interface BatchSummary {
  attempted: number;
  imported: number;
  failed: number;
}

export function normaliseExtension(value: string): string {
  const trimmed = value.trim().toLowerCase();
  if (!trimmed) return "";
  return trimmed.startsWith(".") ? trimmed : `.${trimmed}`;
}

/** `[".tif", ".tiff"]` -> `".tif, .tiff"` for prose. */
export function formatExtensionList(extensions: string[]): string {
  return extensions.join(", ");
}

/**
 * `[".tif", ".tiff", ".png"]` -> `"TIFF or PNG"`.
 *
 * The plan's copy for this line is literally "TIFF or PNG from this computer",
 * and the accepted set still has to come from `supported_upload_formats` rather
 * than a constant here -- the server validates against `UPLOAD_SUFFIXES` and
 * this field exists precisely so the picker cannot drift from it. Deriving the
 * family names satisfies both: the shipped build renders the planned sentence
 * word for word, and a build that accepts something else says so instead of
 * lying.
 */
export function formatFormatFamilies(extensions: string[]): string {
  const families: string[] = [];
  for (const extension of extensions) {
    const family = extension.replace(/^\./, "").toUpperCase();
    const merged = family === "TIF" ? "TIFF" : family;
    if (!families.includes(merged)) families.push(merged);
  }
  if (families.length === 0) return "Images";
  if (families.length === 1) return families[0];
  return `${families.slice(0, -1).join(", ")} or ${families[families.length - 1]}`;
}

export function stripKnownExtension(filename: string, extensions: string[]): string {
  const lower = filename.toLowerCase();
  const match = extensions.find((extension) => lower.endsWith(extension));
  return match ? filename.slice(0, -match.length) : filename;
}

/** "image" / "images", so no sentence has to read "1 images". */
export function plural(count: number, singular: string, pluralForm = `${singular}s`) {
  return count === 1 ? singular : pluralForm;
}

/**
 * The same file, chosen twice.
 *
 * A user who drops a folder and then drops it again should not upload every
 * image twice; the server would happily create forty more assets. Name, size
 * and modification time is what a browser can compare without reading the
 * bytes, and it is what every file manager uses for the same question.
 */
export function isSameFile(left: File, right: File): boolean {
  return (
    left.name === right.name &&
    left.size === right.size &&
    left.lastModified === right.lastModified
  );
}
