import { describe, expect, it } from "vitest";
import { describeObjectsPixelSize } from "@/shared/objectsPixelSize";
import type { SegmentationObjectsPixelSize } from "@/shared/types/images";

function seg(info: SegmentationObjectsPixelSize | null | undefined) {
  return { objects_pixel_size: info };
}

describe("describeObjectsPixelSize", () => {
  /**
   * The reported gap: import uncalibrated, run inference, type 5 nm/px in,
   * proofread — the labeling header reads "5 nm/px" over objects that were
   * produced with no pixel size at all, and the analysis
   * blanks every physical unit. Nothing said so before the run was spent.
   */
  it("warns when the objects predate the calibration", () => {
    const warning = describeObjectsPixelSize(
      seg({
        produced_nm: [null],
        predates_calibration: true,
        unstamped_count: 0,
      })
    );

    expect(warning).not.toBeNull();
    expect(warning!.summary).toBe("Objects predate the pixel size");
    expect(warning!.detail).toContain(
      "produced while this image had no pixel size"
    );
    // The half users get wrong: typing the number afterwards feels like a fix.
    expect(warning!.detail).toContain("does not re-measure");
    expect(warning!.consequence).toContain("in pixels");
    expect(warning!.consequence).toContain("wrong-scale caveat");
  });

  it("says 'some' and lists the scales when runs at a real scale are mixed in", () => {
    const warning = describeObjectsPixelSize(
      seg({
        produced_nm: [5, null],
        predates_calibration: true,
        unstamped_count: 0,
      })
    );

    expect(warning!.detail).toMatch(/^Some of the objects/);
    expect(warning!.detail).toContain("5 nm/px");
    expect(warning!.detail).toContain("no pixel size");
  });

  it("counts unstamped objects separately, never as wrongly-scaled", () => {
    const warning = describeObjectsPixelSize(
      seg({
        produced_nm: [null],
        predates_calibration: true,
        unstamped_count: 3,
      })
    );

    expect(warning!.detail).toContain("3 object(s) here carry no run record");
    // "not produced by a model" is not "produced at an unknown scale":
    // telling someone their hand-drawn polygons are miscalibrated is what
    // would make them discard their own work.
    expect(warning!.detail).toContain("says nothing about their scale");
  });

  /**
   * The crying-wolf guard: every calibrated image whose objects were made at
   * its recorded scale must render nothing. The server owns the verdict, so
   * `predates_calibration: false` is the whole test — even with a null in
   * `produced_nm` (uncalibrated image, objects honestly in pixels: the badge
   * beside this already says "Pixel size not set").
   */
  it("is silent when the server says the objects do not predate it", () => {
    expect(
      describeObjectsPixelSize(
        seg({ produced_nm: [5], predates_calibration: false, unstamped_count: 0 })
      )
    ).toBeNull();
    expect(
      describeObjectsPixelSize(
        seg({ produced_nm: [null], predates_calibration: false, unstamped_count: 2 })
      )
    ).toBeNull();
  });

  it("is silent for an empty segmentation, an older backend, and no segmentation", () => {
    expect(describeObjectsPixelSize(seg(null))).toBeNull();
    expect(describeObjectsPixelSize(seg(undefined))).toBeNull();
    expect(describeObjectsPixelSize(null)).toBeNull();
    expect(describeObjectsPixelSize(undefined)).toBeNull();
  });

  it("reports a damaged stamp value unchanged rather than failing the read", () => {
    const warning = describeObjectsPixelSize(
      seg({
        produced_nm: ["garbage", null],
        predates_calibration: true,
        unstamped_count: 0,
      })
    );

    expect(warning!.detail).toContain("garbage");
  });
});
