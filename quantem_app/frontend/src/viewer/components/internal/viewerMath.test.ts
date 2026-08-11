import { describe, expect, it } from "vitest";
import {
  buildMetrics,
  clampViewportToImage,
  defaultViewport,
  fitBoundsViewport,
  oneToOneZoom,
  scaleBarPlan,
} from "@/viewer/components/internal/viewerMath";
import type { ViewportState } from "@/viewer/types";

describe("defaultViewport", () => {
  /**
   * The whole image has to be on screen when it opens. `zoom: 1` only ever
   * fitted the width, so a square image in a landscape panel opened with its
   * top and bottom cropped and the first act on every image was to zoom out.
   */
  function visible(
    imageWidth: number,
    imageHeight: number,
    containerWidth: number,
    containerHeight: number
  ) {
    const viewport = defaultViewport(
      imageWidth,
      imageHeight,
      containerWidth,
      containerHeight
    );
    return buildMetrics(viewport, imageWidth, imageHeight);
  }

  it("fits the whole of a square image into a landscape container", () => {
    const metrics = visible(1466, 1466, 1005, 708);

    expect(metrics.visibleWidth).toBeGreaterThanOrEqual(1466);
    // Height is the binding constraint here, so it fits exactly.
    expect(metrics.visibleHeight).toBeCloseTo(1466);
    expect(metrics.minX).toBeCloseTo(733 - metrics.visibleWidth / 2);
    expect(metrics.minY).toBeCloseTo(733 - metrics.visibleHeight / 2);
  });

  it("fits the whole of a wide image, letterboxing the height", () => {
    const metrics = visible(4000, 1000, 1005, 708);

    expect(metrics.visibleWidth).toBeCloseTo(4000);
    expect(metrics.visibleHeight).toBeGreaterThanOrEqual(1000);
  });

  it("fits the whole of a tall image into a wide container", () => {
    const metrics = visible(1000, 4000, 1600, 400);

    expect(metrics.visibleWidth).toBeGreaterThanOrEqual(1000);
    expect(metrics.visibleHeight).toBeCloseTo(4000);
  });

  it("never zooms past 1:1 on the width, so a small image is not blown up", () => {
    expect(defaultViewport(1466, 1466, 1005, 708).zoom).toBeLessThanOrEqual(1);
    expect(defaultViewport(512, 512, 4000, 400).zoom).toBeLessThanOrEqual(1);
  });

  it("centres the image rather than pinning its corner", () => {
    const viewport = defaultViewport(4000, 1000, 1005, 708);

    expect(viewport.centerX).toBeCloseTo(0.5);
    // Centres are in `imageWidth` units; see `buildMetrics`.
    expect(viewport.centerY * 4000).toBeCloseTo(500);
  });

  it("survives a container that has not been measured yet", () => {
    const viewport = defaultViewport(1466, 1466, 0, 0);

    expect(Number.isFinite(viewport.zoom)).toBe(true);
    expect(viewport.zoom).toBeGreaterThan(0);
    expect(viewport.containerWidth).toBeGreaterThan(0);
    expect(viewport.containerHeight).toBeGreaterThan(0);
  });
});

describe("fitBoundsViewport", () => {
  it("limits fit zoom using the container aspect ratio", () => {
    const viewport = fitBoundsViewport({
      fitBounds: {
        x: 96,
        y: 196,
        width: 108,
        height: 108,
      },
      fitBoundsPaddingRatio: 0,
      containerWidth: 1600,
      containerHeight: 800,
      imageWidth: 2048,
      imageHeight: 2048,
    });

    expect(viewport.centerX).toBeCloseTo(150 / 2048);
    expect(viewport.centerY).toBeCloseTo(250 / 2048);
    expect(viewport.zoom).toBeCloseTo(2048 / 216);
  });

  it("clamps padded fits to the image bounds", () => {
    const viewport = fitBoundsViewport({
      fitBounds: {
        x: 0,
        y: 0,
        width: 40,
        height: 40,
      },
      fitBoundsPaddingRatio: 0.25,
      containerWidth: 400,
      containerHeight: 400,
      imageWidth: 100,
      imageHeight: 100,
    });

    expect(viewport.centerX).toBeCloseTo(25 / 100);
    expect(viewport.centerY).toBeCloseTo(25 / 100);
    expect(viewport.zoom).toBeCloseTo(100 / 50);
  });
});

describe("clampViewportToImage", () => {
  /**
   * Nothing bounded the pan, so a few flicks of the wrist left a black canvas
   * with the image off to one side and no way back except a reload.
   */
  function viewport(centerX: number, centerY: number): ViewportState {
    return {
      centerX,
      centerY,
      zoom: 1,
      containerWidth: 1000,
      containerHeight: 800,
    };
  }

  it("keeps the centre of the canvas over the image when panned far right", () => {
    const clamped = clampViewportToImage(viewport(9.5, 0.25), 2000, 1000);

    expect(clamped.centerX).toBe(1);
    expect(clamped.centerY).toBe(0.25);
  });

  it("keeps the centre of the canvas over the image when panned far up", () => {
    const clamped = clampViewportToImage(viewport(0.4, -6), 2000, 1000);

    expect(clamped.centerX).toBe(0.4);
    expect(clamped.centerY).toBe(0);
  });

  it("bounds the vertical centre by the image's own aspect, not by 1", () => {
    // Centres are in `imageWidth` units, so a 2000x1000 image ends at 0.5.
    const clamped = clampViewportToImage(viewport(0.5, 0.9), 2000, 1000);

    expect(clamped.centerY).toBe(0.5);
  });

  it("leaves an in-bounds viewport strictly untouched", () => {
    const inBounds = viewport(0.5, 0.25);

    expect(clampViewportToImage(inBounds, 2000, 1000)).toBe(inBounds);
  });

  it("does nothing when the image has no measured size yet", () => {
    const unbounded = viewport(9, 9);

    expect(clampViewportToImage(unbounded, 0, 0)).toBe(unbounded);
  });
});

describe("oneToOneZoom", () => {
  it("puts one image pixel on one screen pixel", () => {
    const zoom = oneToOneZoom(4000, 1000);
    const metrics = buildMetrics(
      { centerX: 0.5, centerY: 0.25, zoom, containerWidth: 1000, containerHeight: 800 },
      4000,
      2000
    );

    expect(metrics.visibleWidth).toBeCloseTo(1000);
  });

  it("survives an unmeasured container", () => {
    expect(Number.isFinite(oneToOneZoom(4000, 0))).toBe(true);
  });
});

describe("scaleBarPlan", () => {
  /**
   * The bar has to mean nanometres at the zoom actually on screen, and the
   * number under it has to be one a reader can hold in their head.
   */
  function metricsAtZoom(zoom: number) {
    return buildMetrics(
      { centerX: 0.5, centerY: 0.25, zoom, containerWidth: 1000, containerHeight: 800 },
      4000,
      2000
    );
  }

  it("reads correctly at three zoom levels on a 5 nm/px image", () => {
    // Fit: 4000 image px across 1000 screen px -> 4 image px per screen px ->
    // 20 nm per screen px, so a 160 px budget covers 3 200 nm and the largest
    // round length that fits is 2 um.
    const fit = scaleBarPlan(metricsAtZoom(1), 5, 160);
    expect(fit?.label).toBe("2 µm");
    expect(fit?.lengthPx).toBeCloseTo(2000 / 20);

    // 1:1 -> 5 nm per screen px, 160 px covers 800 nm -> 500 nm.
    const oneToOne = scaleBarPlan(metricsAtZoom(oneToOneZoom(4000, 1000)), 5, 160);
    expect(oneToOne?.label).toBe("500 nm");
    expect(oneToOne?.lengthPx).toBeCloseTo(500 / 5);

    // 8x -> 0.625 nm per screen px, 160 px covers 100 nm -> 100 nm exactly.
    const zoomedIn = scaleBarPlan(metricsAtZoom(oneToOneZoom(4000, 1000) * 8), 5, 160);
    expect(zoomedIn?.label).toBe("100 nm");
    expect(zoomedIn?.lengthPx).toBeCloseTo(100 / 0.625);
  });

  it("never draws a bar wider than its budget", () => {
    for (const zoom of [0.25, 1, 3, 17, 200]) {
      const plan = scaleBarPlan(metricsAtZoom(zoom), 5, 160);
      expect(plan).not.toBeNull();
      expect(plan!.lengthPx).toBeLessThanOrEqual(160);
    }
  });

  it("switches to millimetres on a coarse survey image", () => {
    // 20 000 nm/px at 4 image px per screen px is 80 um per screen px, so a
    // 160 px budget covers 12.8 mm and the bar reads 10 mm.
    const plan = scaleBarPlan(metricsAtZoom(1), 20000, 160);

    expect(plan?.label).toBe("10 mm");
    expect(plan?.lengthNm).toBe(1e7);
  });

  it("draws nothing at all for an uncalibrated image", () => {
    expect(scaleBarPlan(metricsAtZoom(1), null, 160)).toBeNull();
    expect(scaleBarPlan(metricsAtZoom(1), undefined, 160)).toBeNull();
    expect(scaleBarPlan(metricsAtZoom(1), 0, 160)).toBeNull();
    expect(scaleBarPlan(metricsAtZoom(1), Number.NaN, 160)).toBeNull();
  });
});
