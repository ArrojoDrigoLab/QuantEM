import { describe, expect, it } from "vitest";
import {
  formatPixelSizeNm,
  parsePixelSizeInput,
  readFileDeclaredPixelSizeNm,
  resolveEntryPixelSize,
  resolvePixelSize,
} from "@/shared/pixelSize";
import type { AssetRendition, HomeEntry } from "@/shared/types/images";

function fullRendition(metadata: Record<string, unknown>): AssetRendition {
  return { id: "rend-1", type: "FULL", metadata };
}

describe("readFileDeclaredPixelSizeNm", () => {
  it("reads the 2D import's source metadata", () => {
    expect(
      readFileDeclaredPixelSizeNm([
        fullRendition({ source_metadata: { pixel_size_nm: 5 } }),
      ])
    ).toBe(5);
  });

  it("reads the in-plane component of a volume's voxel size", () => {
    expect(
      readFileDeclaredPixelSizeNm([
        fullRendition({ volume_metadata: { voxel_size_nm: [50, 4.2, 4.2] } }),
      ])
    ).toBe(4.2);
  });

  it("returns null when the file declared nothing", () => {
    expect(
      readFileDeclaredPixelSizeNm([
        fullRendition({ source_metadata: { pixel_size_nm: null, width: 1024 } }),
      ])
    ).toBeNull();
    expect(readFileDeclaredPixelSizeNm([])).toBeNull();
    expect(readFileDeclaredPixelSizeNm(undefined)).toBeNull();
  });

  it("ignores a non-positive declared value rather than trusting it", () => {
    expect(
      readFileDeclaredPixelSizeNm([
        fullRendition({ source_metadata: { pixel_size_nm: 0 } }),
      ])
    ).toBeNull();
  });

  it("prefers the FULL rendition over any other", () => {
    expect(
      readFileDeclaredPixelSizeNm([
        { id: "p", type: "PREVIEW", metadata: { source_metadata: { pixel_size_nm: 99 } } },
        fullRendition({ source_metadata: { pixel_size_nm: 7 } }),
      ])
    ).toBe(7);
  });
});

describe("resolvePixelSize", () => {
  it("reports 'unset' for an uncalibrated image", () => {
    const resolved = resolvePixelSize({ pixel_size_nm: null });
    expect(resolved.source).toBe("unset");
    expect(resolved.calibrated).toBe(false);
    expect(resolved.valueNm).toBeNull();
  });

  it("reports 'file' when the stored value matches what the file declared", () => {
    const resolved = resolvePixelSize({
      pixel_size_nm: 5,
      renditions: [fullRendition({ source_metadata: { pixel_size_nm: 5 } })],
    });
    expect(resolved.source).toBe("file");
    expect(resolved.fileDeclaredNm).toBe(5);
  });

  it("reports 'manual' when the file was silent", () => {
    const resolved = resolvePixelSize({
      pixel_size_nm: 4.2,
      renditions: [fullRendition({ source_metadata: { pixel_size_nm: null } })],
    });
    expect(resolved.source).toBe("manual");
    expect(resolved.fileDeclaredNm).toBeNull();
  });

  it("reports 'manual' when the user overrode the file's own value", () => {
    const resolved = resolvePixelSize({
      pixel_size_nm: 4.2,
      renditions: [fullRendition({ source_metadata: { pixel_size_nm: 5 } })],
    });
    expect(resolved.source).toBe("manual");
    expect(resolved.fileDeclaredNm).toBe(5);
  });

  it("tolerates float round-tripping when matching the file value", () => {
    const resolved = resolvePixelSize({
      pixel_size_nm: 4.200000000000001,
      renditions: [fullRendition({ source_metadata: { pixel_size_nm: 4.2 } })],
    });
    expect(resolved.source).toBe("file");
  });

  it("cannot claim 'file' without renditions, so a list entry stays 'manual'", () => {
    expect(resolvePixelSize({ pixel_size_nm: 5 }).source).toBe("manual");
  });
});

describe("resolveEntryPixelSize", () => {
  function entry(overrides: Partial<HomeEntry>): HomeEntry {
    return {
      id: "asset-1",
      display_name: "Liver HFD3 ROI7",
      original_filename: "liver_HFD3_ROI7.tif",
      metadata_summary: "",
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
      preprocess_stage: "DONE",
      preprocess_progress: 100,
      can_open: true,
      ...overrides,
    } as HomeEntry;
  }

  it("reports 'file' when the list says the file declared the stored value", () => {
    // The regression: this image's 5 nm/px came out of the TIFF XResolution
    // tag, the viewer said "from file", and the library card said "Entered by
    // hand: the image file declared no pixel size".
    const resolved = resolveEntryPixelSize(
      entry({ pixel_size_nm: 5, file_declared_pixel_size_nm: 5 })
    );
    expect(resolved.source).toBe("file");
    expect(resolved.fileDeclaredNm).toBe(5);
  });

  it("reports 'manual' when the file genuinely declared nothing", () => {
    const resolved = resolveEntryPixelSize(
      entry({ pixel_size_nm: 5, file_declared_pixel_size_nm: null })
    );
    expect(resolved.source).toBe("manual");
    expect(resolved.fileDeclaredNm).toBeNull();
  });

  it("reports 'manual' when the user overrode the file's own value", () => {
    const resolved = resolveEntryPixelSize(
      entry({ pixel_size_nm: 4.2, file_declared_pixel_size_nm: 5 })
    );
    expect(resolved.source).toBe("manual");
    expect(resolved.fileDeclaredNm).toBe(5);
  });

  it("says 'unknown', not 'manual', when the backend omits the field", () => {
    // An older server that does not send file_declared_pixel_size_nm knows
    // nothing about provenance. Guessing "entered by hand" there is the exact
    // false claim this whole resolver exists to avoid.
    const resolved = resolveEntryPixelSize(entry({ pixel_size_nm: 5 }));
    expect(resolved.source).toBe("unknown");
    expect(resolved.calibrated).toBe(true);
    expect(resolved.valueNm).toBe(5);
  });

  it("reports 'unset' for an uncalibrated entry", () => {
    const resolved = resolveEntryPixelSize(
      entry({ pixel_size_nm: null, file_declared_pixel_size_nm: null })
    );
    expect(resolved.source).toBe("unset");
    expect(resolved.calibrated).toBe(false);
  });
});

describe("formatPixelSizeNm", () => {
  it("trims trailing zeros", () => {
    expect(formatPixelSizeNm(4.2)).toBe("4.2 nm/px");
    expect(formatPixelSizeNm(8)).toBe("8 nm/px");
  });

  it("is empty for an unset or invalid value", () => {
    expect(formatPixelSizeNm(null)).toBe("");
    expect(formatPixelSizeNm(0)).toBe("");
    expect(formatPixelSizeNm(Number.NaN)).toBe("");
  });
});

describe("parsePixelSizeInput", () => {
  it("treats blank as 'unknown', not as an error", () => {
    expect(parsePixelSizeInput("  ")).toEqual({ value: null, error: null });
  });

  it("rejects non-numeric and non-positive input the way the backend does", () => {
    expect(parsePixelSizeInput("abc").error).toBe("Pixel size must be a number.");
    expect(parsePixelSizeInput("0").error).toBe(
      "Pixel size must be greater than zero."
    );
    expect(parsePixelSizeInput("-3").error).toBe(
      "Pixel size must be greater than zero."
    );
  });

  it("accepts a positive decimal", () => {
    expect(parsePixelSizeInput(" 4.2 ")).toEqual({ value: 4.2, error: null });
  });
});
