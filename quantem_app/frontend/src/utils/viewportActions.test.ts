import { describe, expect, it } from "vitest";
import { createViewportActionResolver } from "@/utils/viewportActions";
import type { ViewportState } from "@/viewer/types";

describe("createViewportActionResolver", () => {
  const current: ViewportState = {
    centerX: 0.4,
    centerY: 0.3,
    zoom: 2,
    containerWidth: 800,
    containerHeight: 400,
  };

  it("resolves fitToBounds actions", () => {
    const resolve = createViewportActionResolver(1000, 500);

    const next = resolve(
      { type: "fitToBounds", x: 100, y: 50, width: 200, height: 100, padding: 0.1 },
      current
    );

    expect(next).toEqual({
      centerX: 0.2,
      centerY: 0.2,
      zoom: expect.closeTo(4.54545, 4),
      containerWidth: 800,
      containerHeight: 400,
    });
  });

  it("keeps zoom for centerOnPoint when requested", () => {
    const resolve = createViewportActionResolver(1000, 500);
    const next = resolve(
      { type: "centerOnPoint", x: 900, y: 250, keepZoom: true },
      current
    );

    expect(next).toEqual({
      centerX: 0.9,
      centerY: 0.5,
      zoom: 2,
      containerWidth: 800,
      containerHeight: 400,
    });
  });

  it("uses current center for setZoom and applies panTo values", () => {
    const resolve = createViewportActionResolver(1000, 500);

    const zoomed = resolve({ type: "setZoom", zoom: 4 }, current);
    expect(zoomed).toEqual({
      centerX: 0.4,
      centerY: 0.3,
      zoom: 4,
      containerWidth: 800,
      containerHeight: 400,
    });

    const panned = resolve(
      { type: "panTo", centerX: 0.75, centerY: 0.25, keepZoom: true },
      current
    );
    expect(panned).toEqual({
      centerX: 0.75,
      centerY: 0.25,
      zoom: 2,
      containerWidth: 800,
      containerHeight: 400,
    });
  });

  it("passes through setViewport action", () => {
    const resolve = createViewportActionResolver(1000, 500);
    const target: ViewportState = {
      centerX: 0.1,
      centerY: 0.2,
      zoom: 3,
      containerWidth: 1,
      containerHeight: 1,
    };
    expect(resolve({ type: "setViewport", viewport: target }, null)).toEqual(target);
  });
});
