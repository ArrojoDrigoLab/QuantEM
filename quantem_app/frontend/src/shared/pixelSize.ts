/**
 * Pixel size, and where it came from.
 *
 * Nothing measurable works without a pixel size: it gates per-organelle
 * resampling and every physical-unit number the analysis suite produces, and an
 * asset with `pixel_size_nm === null` is refused by analysis outright. EM
 * exports frequently carry no resolution tag, so the value has to be editable.
 *
 * The backend stores one number and does not record its origin, but the detail
 * payload does carry what the *file itself* declared, on the FULL rendition's
 * metadata (`source_metadata.pixel_size_nm` for a 2D import,
 * `volume_metadata.voxel_size_nm` for a volume). Comparing the effective value
 * against that is how "read from file" is separated from "entered by hand" --
 * a distinction that belongs in a figure caption, so it belongs in the UI.
 */

import type { AssetDetail, AssetRendition, HomeEntry } from "@/shared/types/images";

/**
 * Where the effective pixel size came from.
 *
 * `"unknown"` is not a synonym for `"manual"`: it means a value exists and
 * nothing in the payload says whether the file declared it. Collapsing the two
 * is the bug this type exists to prevent — the library list used to resolve
 * every calibrated image to `"manual"` and assert "the image file declared no
 * pixel size" about images whose 5 nm/px came straight out of a TIFF tag.
 */
export type PixelSizeSource = "file" | "manual" | "unknown" | "unset";

export interface ResolvedPixelSize {
  /** Effective nm per pixel in plane, or null when the image is uncalibrated. */
  valueNm: number | null;
  source: PixelSizeSource;
  /** What the source file declared, when it declared anything. */
  fileDeclaredNm: number | null;
  /** True when a value exists and analysis will emit physical units. */
  calibrated: boolean;
}

/**
 * Relative tolerance for "the stored value is the file's value".
 *
 * The round-trip is float -> JSON -> float, so exact equality is not safe; but
 * the gap between a real re-measurement and a rounding artefact is orders of
 * magnitude, so a tight relative epsilon cannot mislabel a hand-typed value as
 * file-derived unless the user typed the file's own number, which is harmless.
 */
const FILE_MATCH_RELATIVE_EPSILON = 1e-6;

/** True when the effective value is, within float noise, the file's own value. */
function matchesFileDeclaration(
  valueNm: number,
  fileDeclaredNm: number | null
): boolean {
  return (
    fileDeclaredNm !== null &&
    Math.abs(valueNm - fileDeclaredNm) <=
      FILE_MATCH_RELATIVE_EPSILON * Math.max(valueNm, fileDeclaredNm)
  );
}

function asPositiveNumber(value: unknown): number | null {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

/**
 * In-plane pixel size declared by the source file, or null if it was silent.
 *
 * Reads the FULL rendition first and falls back to any rendition, because a
 * volume import rewrites the same rendition row in place and only the FULL type
 * is guaranteed to exist.
 */
export function readFileDeclaredPixelSizeNm(
  renditions: AssetRendition[] | undefined
): number | null {
  if (!renditions || renditions.length === 0) return null;
  const ordered = [
    ...renditions.filter((rendition) => rendition.type === "FULL"),
    ...renditions.filter((rendition) => rendition.type !== "FULL"),
  ];
  for (const rendition of ordered) {
    const metadata = record(rendition.metadata);
    if (!metadata) continue;

    const sourceMetadata = record(metadata.source_metadata);
    const declared = asPositiveNumber(sourceMetadata?.pixel_size_nm);
    if (declared !== null) return declared;

    // Volumes: voxel_size_nm is (z, y, x) and x is the in-plane size.
    const volumeMetadata = record(metadata.volume_metadata);
    const voxel = volumeMetadata?.voxel_size_nm;
    if (Array.isArray(voxel) && voxel.length >= 3) {
      const inPlane = asPositiveNumber(voxel[2]);
      if (inPlane !== null) return inPlane;
    }
  }
  return null;
}

/**
 * Effective pixel size plus its provenance.
 *
 * `HomeEntry` (list payload) has no renditions, so it can only ever resolve to
 * `"manual"` or `"unset"`; call sites that need the file/manual distinction must
 * pass an `AssetDetail`.
 */
export function resolvePixelSize(
  asset: Pick<AssetDetail, "pixel_size_nm"> &
    Partial<Pick<AssetDetail, "renditions">>
): ResolvedPixelSize {
  const valueNm = asPositiveNumber(asset.pixel_size_nm ?? null);
  const fileDeclaredNm = readFileDeclaredPixelSizeNm(asset.renditions);
  if (valueNm === null) {
    return { valueNm: null, source: "unset", fileDeclaredNm, calibrated: false };
  }
  return {
    valueNm,
    source: matchesFileDeclaration(valueNm, fileDeclaredNm) ? "file" : "manual",
    fileDeclaredNm,
    calibrated: true,
  };
}

/**
 * Same resolution for a library list entry.
 *
 * The list payload carries no renditions, so the file's own claim arrives as
 * the scalar `file_declared_pixel_size_nm` instead (`serialize_asset_entry`).
 * When that field is absent entirely — an older backend — the provenance is
 * `"unknown"` rather than `"manual"`, because "the file declared nothing" and
 * "this payload cannot say" are different statements and only one of them is
 * safe to print next to a number that ends up in a figure caption.
 */
export function resolveEntryPixelSize(entry: HomeEntry): ResolvedPixelSize {
  const valueNm = asPositiveNumber(entry.pixel_size_nm ?? null);
  if (valueNm === null) {
    return { valueNm: null, source: "unset", fileDeclaredNm: null, calibrated: false };
  }
  if (!("file_declared_pixel_size_nm" in entry)) {
    return { valueNm, source: "unknown", fileDeclaredNm: null, calibrated: true };
  }
  const fileDeclaredNm = asPositiveNumber(entry.file_declared_pixel_size_nm ?? null);
  return {
    valueNm,
    source: matchesFileDeclaration(valueNm, fileDeclaredNm) ? "file" : "manual",
    fileDeclaredNm,
    calibrated: true,
  };
}

/** `"4.2 nm/px"`, trimmed of trailing zeros; empty string when unset. */
export function formatPixelSizeNm(valueNm: number | null): string {
  if (valueNm === null || !Number.isFinite(valueNm) || valueNm <= 0) return "";
  const text = valueNm.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  return `${text} nm/px`;
}

/** Short provenance word for a badge. Never shown without the number. */
export function pixelSizeSourceLabel(source: PixelSizeSource): string {
  switch (source) {
    case "file":
      return "read from file";
    case "manual":
      return "entered by hand";
    case "unknown":
      return "source not recorded";
    default:
      return "not set";
  }
}

/**
 * Validate a typed pixel size against the same rule the backend enforces
 * (`parse_pixel_size_nm`): blank means "unknown", anything else must parse to a
 * number greater than zero. Returning the error text here keeps the message the
 * user sees identical whether the client or the server rejects it.
 */
export function parsePixelSizeInput(
  raw: string
): { value: number | null; error: null } | { value: null; error: string } {
  const text = raw.trim();
  if (!text) return { value: null, error: null };
  const parsed = Number(text);
  if (!Number.isFinite(parsed)) {
    return { value: null, error: "Pixel size must be a number." };
  }
  if (parsed <= 0) {
    return { value: null, error: "Pixel size must be greater than zero." };
  }
  return { value: parsed, error: null };
}
