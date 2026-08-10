/**
 * Whether the file the user just picked declares its own pixel size.
 *
 * The import form warns that a run will happen at the wrong scale, and until
 * this existed it decided that from the *typed* box alone. The helper text
 * directly above the box says "Leave blank to use the value in the file", so the
 * commonest correct workflow -- a TIFF that declares 5 nm/px, box left empty,
 * organelles ticked -- produced the full uncalibrated warning and a submit
 * button reading "Import and segment uncalibrated" over an import that came back
 * `pixel_size_nm: 5.0, file_declared_pixel_size_nm: 5.0`. A warning that fires
 * on the correct workflow is one people learn to click through, which costs the
 * warning its force everywhere it is right.
 *
 * So the file is read before the form claims anything. Only the header and the
 * first IFD are fetched -- a few hundred bytes through `Blob.slice`, never the
 * pixel data -- so this is cheap on a 40k x 40k image.
 *
 * **The precedence mirrors the server exactly**, because the point is to predict
 * what the import will store: OME `PhysicalSizeX` first, then ImageJ's `unit=`
 * applied to `XResolution`, then bare `XResolution` with `ResolutionUnit`. See
 * `quantem.assets.utils._tiff_pixel_size_nm` and
 * `quantem.assets.volume_readers._resolution_tag_nm`.
 *
 * Three answers, and the asymmetry between them is deliberate:
 *
 *   - `"declared"` suppresses the warning, so it is only returned when a
 *     positive value was actually read. A wrong `"declared"` hides the warning
 *     that matters most, which is the worst thing this file could do.
 *   - `"silent"` is returned only after the structure was understood and no
 *     scale was in it. That is what earns the unconditional wording.
 *   - `"unknown"` is everything else -- BigTIFF, a truncated read, a structure
 *     this does not model. The form then hedges rather than asserting either
 *     way.
 */

/** Formats the importer can read a pixel size out of. Mirrors `UPLOAD_SUFFIXES`. */
const TIFF_SUFFIXES = [".tif", ".tiff"];

/**
 * Formats that cannot carry one at all.
 *
 * `extract_png_metadata` returns no `pixel_size_nm` key -- Pillow's `pHYs` is
 * never consulted -- so a PNG import is uncalibrated unless the box is typed
 * in. That is a definite "silent", not an "unknown".
 */
const PIXEL_SIZE_BLIND_SUFFIXES = [".png"];

export type FileDeclaredPixelSize =
  /** The file says so, and this is the value the import will store. */
  | { state: "declared"; pixelSizeNm: number }
  /** The file was read and says nothing: this import will be uncalibrated. */
  | { state: "silent" }
  /** Not readable here. Say "if", not "will". */
  | { state: "unknown" };

const UNKNOWN: FileDeclaredPixelSize = { state: "unknown" };
const SILENT: FileDeclaredPixelSize = { state: "silent" };

/** Nanometres per unit, matching `volume_readers._UNIT_TO_NM`. */
const UNIT_TO_NM: Record<string, number> = {
  nm: 1,
  nanometer: 1,
  nanometre: 1,
  um: 1000,
  "µm": 1000,
  micron: 1000,
  micrometer: 1000,
  micrometre: 1000,
  mm: 1_000_000,
  a: 0.1,
  angstrom: 0.1,
  "å": 0.1,
  pm: 0.001,
};

/**
 * TIFF `ResolutionUnit`, in nm per unit.
 *
 * 1 means "no absolute unit" -- the resolution is a bare aspect ratio and
 * carries no physical scale, so it must not be converted at all. The server
 * returns `None` for it, and so does this.
 */
const RESOLUTION_UNIT_NM: Record<number, number> = {
  2: 25_400_000,
  3: 10_000_000,
};

/**
 * Bytes per TIFF 6.0 field type. Only needed to decide whether a value fits in
 * the entry's 4-byte field or lives at an offset; an unlisted type is treated
 * as "does not fit", which sends the read to the offset and then fails a type
 * check, never a silent misread.
 */
const TYPE_SIZES: Record<number, number> = {
  1: 1, // BYTE
  2: 1, // ASCII
  3: 2, // SHORT
  4: 4, // LONG
  5: 8, // RATIONAL
  6: 1, // SBYTE
  7: 1, // UNDEFINED
  8: 2, // SSHORT
  9: 4, // SLONG
  10: 8, // SRATIONAL
  11: 4, // FLOAT
  12: 8, // DOUBLE
};

const TAG_IMAGE_DESCRIPTION = 270;
const TAG_X_RESOLUTION = 282;
const TAG_RESOLUTION_UNIT = 296;

/**
 * Ceilings, so a hostile or corrupt file cannot make this read a whole disk.
 * An IFD with more than this many entries, or an ImageDescription larger than
 * this, is not something to guess about: the answer becomes `"unknown"`.
 */
const MAX_IFD_ENTRIES = 1024;
const MAX_DESCRIPTION_BYTES = 4 * 1024 * 1024;

interface TiffEntry {
  type: number;
  count: number;
  /** File offset of the value, or null when it sits inside the entry. */
  offset: number | null;
  /** The raw value field, used when the value fits inline. */
  inline: DataView;
  littleEndian: boolean;
}

/**
 * `length` bytes at `start`, or null when the file does not have them.
 *
 * `FileReader` rather than `Blob.arrayBuffer()`: the two are equivalent in every
 * browser this ships to, and only this one exists in jsdom -- so the tests
 * exercise the same code path production does instead of a second, untested
 * branch behind a capability check.
 */
async function readRange(
  file: Blob,
  start: number,
  length: number
): Promise<DataView | null> {
  if (!Number.isSafeInteger(start) || start < 0 || length <= 0) return null;
  if (start + length > file.size) return null;
  try {
    const buffer = await new Promise<ArrayBuffer>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as ArrayBuffer);
      reader.onerror = () => reject(reader.error ?? new Error("read failed"));
      reader.readAsArrayBuffer(file.slice(start, start + length));
    });
    if (buffer.byteLength < length) return null;
    return new DataView(buffer);
  } catch {
    return null;
  }
}

/** `value * factor`, or null when the unit is one that carries no scale. */
function toNm(value: number, unit: string | null): number | null {
  if (!Number.isFinite(value) || value <= 0) return null;
  const factor = UNIT_TO_NM[(unit ?? "nm").trim().toLowerCase()] ?? 1;
  const nm = value * factor;
  return Number.isFinite(nm) && nm > 0 ? nm : null;
}

/** `_xml_attr`: the first `name="..."` in the document, as a number if it is one. */
function xmlAttr(xml: string, name: string): string | null {
  const match = new RegExp(`${name}="([^"]+)"`).exec(xml);
  return match ? match[1] : null;
}

/** OME `PhysicalSizeX` in nm, or null. Unit defaults to µm, as OME specifies. */
function omePhysicalSizeXNm(xml: string): number | null {
  const raw = xmlAttr(xml, "PhysicalSizeX");
  if (raw === null) return null;
  const value = Number(raw);
  if (!Number.isFinite(value)) return null;
  return toNm(value, xmlAttr(xml, "PhysicalSizeXUnit") ?? "um");
}

/** ImageJ writes `key=value` lines; the server reads only `unit`. */
function imageJUnit(description: string): string | null {
  if (!description.startsWith("ImageJ=")) return null;
  for (const line of description.split("\n")) {
    const [key, ...rest] = line.split("=");
    if (key.trim() === "unit" && rest.length > 0) {
      const value = rest.join("=").trim();
      if (value) return value;
    }
  }
  return null;
}

/**
 * Classic-TIFF IFD entries by tag. 12 bytes each: tag, type, count, then a
 * 4-byte field holding either the value itself or a file offset to it.
 *
 * BigTIFF's 20-byte entries are not modelled -- `probeTiff` answers `"unknown"`
 * for that magic rather than carry a second layout that nothing here exercises.
 */
function parseEntries(
  view: DataView,
  entryCount: number,
  littleEndian: boolean
): Map<number, TiffEntry> {
  const entries = new Map<number, TiffEntry>();
  for (let index = 0; index < entryCount; index += 1) {
    const base = index * 12;
    const tag = view.getUint16(base, littleEndian);
    const type = view.getUint16(base + 2, littleEndian);
    const count = view.getUint32(base + 4, littleEndian);
    const valueField = new DataView(view.buffer, view.byteOffset + base + 8, 4);
    const size = TYPE_SIZES[type];
    entries.set(tag, {
      type,
      count,
      offset:
        size !== undefined && size * count <= 4
          ? null
          : valueField.getUint32(0, littleEndian),
      inline: valueField,
      littleEndian,
    });
  }
  return entries;
}

async function entryBytes(
  file: Blob,
  entry: TiffEntry,
  byteLength: number
): Promise<DataView | null> {
  if (entry.offset === null) {
    return byteLength <= entry.inline.byteLength ? entry.inline : null;
  }
  return readRange(file, entry.offset, byteLength);
}

async function readAscii(file: Blob, entry: TiffEntry): Promise<string | null> {
  if (entry.type !== 2 || entry.count <= 0 || entry.count > MAX_DESCRIPTION_BYTES) {
    return null;
  }
  const view = await entryBytes(file, entry, entry.count);
  if (!view) return null;
  const bytes = new Uint8Array(view.buffer, view.byteOffset, entry.count);
  // Latin-1 rather than UTF-8: TIFF ASCII is byte-oriented, and this is only
  // ever pattern-matched, never displayed.
  let text = "";
  for (const byte of bytes) {
    if (byte === 0) break;
    text += String.fromCharCode(byte);
  }
  return text;
}

/** `XResolution` as source-units-per-pixel, mirroring `_resolution_tag_nm`. */
async function readUnitSize(
  file: Blob,
  entry: TiffEntry | undefined
): Promise<number | null> {
  if (!entry || entry.type !== 5 || entry.count < 1) return null;
  const view = await entryBytes(file, entry, 8);
  if (!view) return null;
  const numerator = view.getUint32(0, entry.littleEndian);
  const denominator = view.getUint32(4, entry.littleEndian);
  if (numerator === 0 || denominator === 0) return null;
  const pixelsPerUnit = numerator / denominator;
  if (!Number.isFinite(pixelsPerUnit) || pixelsPerUnit <= 0) return null;
  return 1 / pixelsPerUnit;
}

function readResolutionUnit(entry: TiffEntry | undefined): number | null {
  if (!entry || entry.type !== 3 || entry.count < 1) return null;
  return entry.inline.getUint16(0, entry.littleEndian);
}

async function probeTiff(file: Blob): Promise<FileDeclaredPixelSize> {
  const header = await readRange(file, 0, 16);
  if (!header) return UNKNOWN;

  const order = header.getUint16(0, false);
  if (order !== 0x4949 && order !== 0x4d4d) return UNKNOWN;
  const littleEndian = order === 0x4949;

  const magic = header.getUint16(2, littleEndian);
  if (magic === 43) {
    // BigTIFF. Rare for the 2D imports this form accepts, and a second layout
    // to get wrong; hedging is cheaper than a confident wrong answer.
    return UNKNOWN;
  }
  if (magic !== 42) return UNKNOWN;

  const ifdOffset = header.getUint32(4, littleEndian);
  const countView = await readRange(file, ifdOffset, 2);
  if (!countView) return UNKNOWN;
  const entryCount = countView.getUint16(0, littleEndian);
  if (entryCount === 0) return SILENT;
  if (entryCount > MAX_IFD_ENTRIES) return UNKNOWN;

  const entriesView = await readRange(file, ifdOffset + 2, entryCount * 12);
  if (!entriesView) return UNKNOWN;
  const entries = parseEntries(entriesView, entryCount, littleEndian);

  const descriptionEntry = entries.get(TAG_IMAGE_DESCRIPTION);
  const description = descriptionEntry
    ? await readAscii(file, descriptionEntry)
    : null;

  // OME first, exactly as the server orders it, and falling through when the
  // XML carries no PhysicalSizeX.
  if (description && description.includes("<OME")) {
    const ome = omePhysicalSizeXNm(description);
    if (ome !== null) return { state: "declared", pixelSizeNm: ome };
  }

  const unitSize = await readUnitSize(file, entries.get(TAG_X_RESOLUTION));
  if (unitSize === null) return SILENT;

  const imagejUnit = description ? imageJUnit(description) : null;
  if (imagejUnit) {
    const nm = toNm(unitSize, imagejUnit);
    return nm === null ? SILENT : { state: "declared", pixelSizeNm: nm };
  }

  const resolutionUnit = readResolutionUnit(entries.get(TAG_RESOLUTION_UNIT));
  if (resolutionUnit === null) return SILENT;
  const factor = RESOLUTION_UNIT_NM[resolutionUnit];
  if (factor === undefined) return SILENT;
  const nm = unitSize * factor;
  return Number.isFinite(nm) && nm > 0
    ? { state: "declared", pixelSizeNm: nm }
    : SILENT;
}

function suffixOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot === -1 ? "" : name.slice(dot).toLowerCase();
}

/**
 * What this file will contribute to `Asset.pixel_size_nm` if the box is blank.
 *
 * Never throws and never rejects: a probe that fails is an `"unknown"`, and the
 * form is written so an `"unknown"` still says something true.
 */
export async function probeFileDeclaredPixelSize(
  file: File
): Promise<FileDeclaredPixelSize> {
  const suffix = suffixOf(file.name);
  if (PIXEL_SIZE_BLIND_SUFFIXES.includes(suffix)) return SILENT;
  if (!TIFF_SUFFIXES.includes(suffix)) return UNKNOWN;
  try {
    return await probeTiff(file);
  } catch {
    return UNKNOWN;
  }
}
