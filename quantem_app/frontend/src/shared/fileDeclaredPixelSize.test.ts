/**
 * The import form used to decide "this run will be uncalibrated" from the typed
 * box alone, while the helper text directly above that box says "Leave blank to
 * use the value in the file". So the commonest correct workflow -- a TIFF that
 * declares its scale, box left empty, organelles ticked -- produced the full
 * uncalibrated warning and a button reading "Import and segment uncalibrated"
 * over an import that came back calibrated at exactly the file's 5 nm/px.
 *
 * These build real TIFF headers byte by byte rather than mocking the reader,
 * because the whole value of the probe is that it agrees with what
 * `quantem.assets.utils._tiff_pixel_size_nm` will do to the same bytes.
 */

import { describe, expect, it } from "vitest";
import { probeFileDeclaredPixelSize } from "@/shared/fileDeclaredPixelSize";

const TAG_IMAGE_WIDTH = 256;
const TAG_IMAGE_DESCRIPTION = 270;
const TAG_X_RESOLUTION = 282;
const TAG_RESOLUTION_UNIT = 296;

const TYPE_ASCII = 2;
const TYPE_SHORT = 3;
const TYPE_LONG = 4;
const TYPE_RATIONAL = 5;

interface TagSpec {
  tag: number;
  type: number;
  /** A SHORT/LONG scalar, a [numerator, denominator] RATIONAL, or ASCII text. */
  value: number | [number, number] | string;
}

/**
 * A classic little-endian TIFF header with one IFD, laid out the way libtiff
 * writes one: header, IFD, then the values too big to sit in an entry.
 */
function makeTiff(tags: TagSpec[]): File {
  const sorted = [...tags].sort((left, right) => left.tag - right.tag);
  const headerSize = 8;
  const ifdSize = 2 + sorted.length * 12 + 4;
  const overflow: ArrayBuffer[] = [];
  let overflowOffset = headerSize + ifdSize;

  const ifd = new DataView(new ArrayBuffer(ifdSize));
  ifd.setUint16(0, sorted.length, true);

  sorted.forEach((spec, index) => {
    const base = 2 + index * 12;
    ifd.setUint16(base, spec.tag, true);
    ifd.setUint16(base + 2, spec.type, true);

    if (spec.type === TYPE_ASCII) {
      const text = `${spec.value as string}\0`;
      const buffer = new ArrayBuffer(text.length);
      const bytes = new Uint8Array(buffer);
      for (let i = 0; i < text.length; i += 1) bytes[i] = text.charCodeAt(i);
      ifd.setUint32(base + 4, text.length, true);
      ifd.setUint32(base + 8, overflowOffset, true);
      overflow.push(buffer);
      overflowOffset += text.length;
      return;
    }

    if (spec.type === TYPE_RATIONAL) {
      const [numerator, denominator] = spec.value as [number, number];
      const buffer = new ArrayBuffer(8);
      const view = new DataView(buffer);
      view.setUint32(0, numerator, true);
      view.setUint32(4, denominator, true);
      ifd.setUint32(base + 4, 1, true);
      ifd.setUint32(base + 8, overflowOffset, true);
      overflow.push(buffer);
      overflowOffset += 8;
      return;
    }

    ifd.setUint32(base + 4, 1, true);
    if (spec.type === TYPE_SHORT) {
      ifd.setUint16(base + 8, spec.value as number, true);
    } else {
      ifd.setUint32(base + 8, spec.value as number, true);
    }
  });

  const header = new DataView(new ArrayBuffer(headerSize));
  header.setUint16(0, 0x4949, true); // "II"
  header.setUint16(2, 42, true);
  header.setUint32(4, headerSize, true);

  return new File([header.buffer, ifd.buffer, ...overflow], "image.tif");
}

/** The minimum a TIFF needs to be a TIFF, with nothing about scale in it. */
const GEOMETRY_ONLY: TagSpec[] = [
  { tag: TAG_IMAGE_WIDTH, type: TYPE_LONG, value: 512 },
];

describe("probeFileDeclaredPixelSize", () => {
  it("reads a plain XResolution in centimetres", async () => {
    // 2 000 000 px/cm -> 5 nm/px, the reported case.
    const probe = await probeFileDeclaredPixelSize(
      makeTiff([
        ...GEOMETRY_ONLY,
        { tag: TAG_X_RESOLUTION, type: TYPE_RATIONAL, value: [2_000_000, 1] },
        { tag: TAG_RESOLUTION_UNIT, type: TYPE_SHORT, value: 3 },
      ])
    );

    expect(probe).toEqual({ state: "declared", pixelSizeNm: 5 });
  });

  it("reads an inch resolution as an inch resolution", async () => {
    // The trap the server had: reading the number and ignoring the unit tag.
    const probe = await probeFileDeclaredPixelSize(
      makeTiff([
        ...GEOMETRY_ONLY,
        { tag: TAG_X_RESOLUTION, type: TYPE_RATIONAL, value: [72, 1] },
        { tag: TAG_RESOLUTION_UNIT, type: TYPE_SHORT, value: 2 },
      ])
    );

    expect(probe.state).toBe("declared");
    expect(probe.state === "declared" && probe.pixelSizeNm).toBeCloseTo(
      25_400_000 / 72,
      3
    );
  });

  it("prefers the OME physical size, as the server does", async () => {
    const probe = await probeFileDeclaredPixelSize(
      makeTiff([
        ...GEOMETRY_ONLY,
        {
          tag: TAG_IMAGE_DESCRIPTION,
          type: TYPE_ASCII,
          value:
            '<?xml version="1.0"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">' +
            '<Image><Pixels PhysicalSizeX="0.008" PhysicalSizeXUnit="µm"/></Image></OME>',
        },
        // Deliberately disagrees: OME wins, so this must not be the answer.
        { tag: TAG_X_RESOLUTION, type: TYPE_RATIONAL, value: [72, 1] },
        { tag: TAG_RESOLUTION_UNIT, type: TYPE_SHORT, value: 2 },
      ])
    );

    expect(probe).toEqual({ state: "declared", pixelSizeNm: 8 });
  });

  it("falls through to the tags when the OME block carries no physical size", async () => {
    const probe = await probeFileDeclaredPixelSize(
      makeTiff([
        ...GEOMETRY_ONLY,
        {
          tag: TAG_IMAGE_DESCRIPTION,
          type: TYPE_ASCII,
          value: '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"/>',
        },
        { tag: TAG_X_RESOLUTION, type: TYPE_RATIONAL, value: [2_000_000, 1] },
        { tag: TAG_RESOLUTION_UNIT, type: TYPE_SHORT, value: 3 },
      ])
    );

    expect(probe).toEqual({ state: "declared", pixelSizeNm: 5 });
  });

  it("uses ImageJ's own unit string over the ResolutionUnit tag", async () => {
    // ImageJ writes its scale into the description and leaves ResolutionUnit at
    // 1 ("no absolute unit"), which on its own carries no scale at all.
    const probe = await probeFileDeclaredPixelSize(
      makeTiff([
        ...GEOMETRY_ONLY,
        {
          tag: TAG_IMAGE_DESCRIPTION,
          type: TYPE_ASCII,
          value: "ImageJ=1.53t\nimages=1\nunit=nm\n",
        },
        { tag: TAG_X_RESOLUTION, type: TYPE_RATIONAL, value: [1, 4] },
        { tag: TAG_RESOLUTION_UNIT, type: TYPE_SHORT, value: 1 },
      ])
    );

    expect(probe).toEqual({ state: "declared", pixelSizeNm: 4 });
  });

  it("calls a TIFF with no resolution tag silent, not unknown", async () => {
    // This is the case that has to keep the unconditional warning: the file was
    // read, understood, and has no scale in it.
    expect(await probeFileDeclaredPixelSize(makeTiff(GEOMETRY_ONLY))).toEqual({
      state: "silent",
    });
  });

  it("treats a unitless resolution as no scale", async () => {
    // ResolutionUnit 1 means the pair is a bare aspect ratio. The server
    // returns None; converting it would invent a scale out of nothing.
    expect(
      await probeFileDeclaredPixelSize(
        makeTiff([
          ...GEOMETRY_ONLY,
          { tag: TAG_X_RESOLUTION, type: TYPE_RATIONAL, value: [300, 1] },
          { tag: TAG_RESOLUTION_UNIT, type: TYPE_SHORT, value: 1 },
        ])
      )
    ).toEqual({ state: "silent" });
  });

  it("treats a zero resolution as no scale", async () => {
    expect(
      await probeFileDeclaredPixelSize(
        makeTiff([
          ...GEOMETRY_ONLY,
          { tag: TAG_X_RESOLUTION, type: TYPE_RATIONAL, value: [0, 1] },
          { tag: TAG_RESOLUTION_UNIT, type: TYPE_SHORT, value: 3 },
        ])
      )
    ).toEqual({ state: "silent" });
  });

  it("calls a PNG silent, because the importer never reads one from a PNG", async () => {
    // `extract_png_metadata` returns no pixel_size_nm at all: Pillow's pHYs is
    // never consulted. So this is a fact, not a guess.
    const png = new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], "image.png");

    expect(await probeFileDeclaredPixelSize(png)).toEqual({ state: "silent" });
  });

  describe("hedges instead of guessing", () => {
    it("on a file that is not a TIFF at all", async () => {
      const junk = new File([new Uint8Array(64)], "image.tif");

      expect(await probeFileDeclaredPixelSize(junk)).toEqual({ state: "unknown" });
    });

    it("on BigTIFF, whose entry layout this does not model", async () => {
      const header = new DataView(new ArrayBuffer(16));
      header.setUint16(0, 0x4949, true);
      header.setUint16(2, 43, true);
      header.setUint16(4, 8, true);

      expect(
        await probeFileDeclaredPixelSize(new File([header.buffer], "big.tif"))
      ).toEqual({ state: "unknown" });
    });

    it("on a truncated file whose IFD offset runs past the end", async () => {
      const header = new DataView(new ArrayBuffer(8));
      header.setUint16(0, 0x4949, true);
      header.setUint16(2, 42, true);
      header.setUint32(4, 1_000_000, true);

      expect(
        await probeFileDeclaredPixelSize(new File([header.buffer], "cut.tif"))
      ).toEqual({ state: "unknown" });
    });

    it("on a suffix the importer does not accept", async () => {
      const other = new File([new Uint8Array(64)], "image.mrc");

      expect(await probeFileDeclaredPixelSize(other)).toEqual({ state: "unknown" });
    });
  });

  it("reads a big-endian TIFF the same way", async () => {
    // "MM" files are common straight off microscope software.
    const tags = new DataView(new ArrayBuffer(8 + 2 + 24 + 4 + 8));
    tags.setUint16(0, 0x4d4d, false); // "MM"
    tags.setUint16(2, 42, false);
    tags.setUint32(4, 8, false);
    tags.setUint16(8, 2, false); // two entries
    tags.setUint16(10, TAG_X_RESOLUTION, false);
    tags.setUint16(12, TYPE_RATIONAL, false);
    tags.setUint32(14, 1, false);
    tags.setUint32(18, 8 + 2 + 24 + 4, false); // value offset
    tags.setUint16(22, TAG_RESOLUTION_UNIT, false);
    tags.setUint16(24, TYPE_SHORT, false);
    tags.setUint32(26, 1, false);
    tags.setUint16(30, 3, false); // centimetre, inline
    tags.setUint32(34, 0, false); // no next IFD
    tags.setUint32(38, 2_000_000, false);
    tags.setUint32(42, 1, false);

    expect(
      await probeFileDeclaredPixelSize(new File([tags.buffer], "be.tif"))
    ).toEqual({ state: "declared", pixelSizeNm: 5 });
  });
});
