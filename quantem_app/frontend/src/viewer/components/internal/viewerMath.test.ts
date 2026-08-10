import { describe, expect, it } from "vitest";
import {
  buildMetrics,
  defaultViewport,
  fitBoundsViewport,
} from "@/viewer/components/internal/viewerMath";

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
