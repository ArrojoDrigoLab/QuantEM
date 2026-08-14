/**
 * Who owns the left button, and where the image is allowed to go.
 *
 * Both are canvas-reform behaviours that only exist as a rule about gestures,
 * so they are tested at the gesture level: press, move, release, and what the
 * viewport was asked to become.
 */

import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useViewerPointerInteractions } from "@/viewer/components/internal/useViewerPointerInteractions";
import { buildMetrics } from "@/viewer/components/internal/viewerMath";
import { resetPanKeyStateForTests } from "@/viewer/panKeyState";
import type { ViewportState } from "@/viewer/types";

const IMAGE_WIDTH = 2000;
const IMAGE_HEIGHT = 1000;

const VIEWPORT: ViewportState = {
  centerX: 0.5,
  centerY: 0.25,
  zoom: 4,
  containerWidth: 1000,
  containerHeight: 800,
};

function makeConfig(overrides: Record<string, unknown> = {}) {
  const container = document.createElement("div");
  container.getBoundingClientRect = () =>
    ({ left: 0, top: 0, width: 1000, height: 800 }) as DOMRect;
  return {
    containerRef: { current: container },
    interactionLayerRef: { current: document.createElement("div") },
    metrics: buildMetrics(VIEWPORT, IMAGE_WIDTH, IMAGE_HEIGHT),
    localViewport: VIEWPORT,
    disablePan: false,
    resolvedImageWidth: IMAGE_WIDTH,
    resolvedImageHeight: IMAGE_HEIGHT,
    setViewport: vi.fn(),
    overlayScene: { persistent: [], transient: [] },
    drawMode: false,
    brushMode: false,
    drawState: {
      drawPoints: [],
      startBrushStroke: vi.fn(),
      appendBrushStroke: vi.fn(),
      finishBrushStroke: vi.fn(),
      setDrawPreviewPoint: vi.fn(),
      completeDraw: vi.fn(),
    },
    cursorState: {
      setIsPointerInside: vi.fn(),
      setLastMouseScreen: vi.fn(),
      updateOverlayCursor: vi.fn(),
    },
    ...overrides,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } as any;
}

function pointerEvent(clientX: number, clientY: number, button = 0) {
  return {
    pointerId: 1,
    button,
    nativeEvent: { clientX, clientY },
    target: { setPointerCapture: vi.fn() },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } as any;
}

function drag(
  result: { current: ReturnType<typeof useViewerPointerInteractions> },
  from: [number, number],
  to: [number, number],
  button = 0
) {
  act(() => {
    result.current.handlePointerDown(pointerEvent(from[0], from[1], button));
    result.current.handlePointerMove(pointerEvent(to[0], to[1], button));
    result.current.handlePointerUp(pointerEvent(to[0], to[1], button));
  });
}

function pressSpace() {
  act(() => {
    window.dispatchEvent(new KeyboardEvent("keydown", { key: " ", code: "Space" }));
  });
}

function releaseSpace() {
  act(() => {
    window.dispatchEvent(new KeyboardEvent("keyup", { key: " ", code: "Space" }));
  });
}

describe("useViewerPointerInteractions", () => {
  beforeEach(() => {
    resetPanKeyStateForTests();
  });

  afterEach(() => {
    resetPanKeyStateForTests();
  });

  describe("who owns a left drag", () => {
    it("pans on a plain left drag when no tool is listening", () => {
      const config = makeConfig();
      const { result } = renderHook(() => useViewerPointerInteractions(config));

      expect(result.current.leftDragPans).toBe(true);
      drag(result, [500, 400], [560, 400]);

      expect(config.setViewport).toHaveBeenCalledTimes(1);
      const [next] = config.setViewport.mock.calls[0];
      expect(next.centerX).toBeLessThan(VIEWPORT.centerX);
    });

    it("does not pan on a left drag once a tool has claimed the button", () => {
      // This is the defect: choosing a correction tool and dragging used to
      // slide the image out from under the stroke.
      const onImagePress = vi.fn();
      const config = makeConfig({ onImagePress, onImageDrag: vi.fn() });
      const { result } = renderHook(() => useViewerPointerInteractions(config));

      expect(result.current.leftDragPans).toBe(false);
      drag(result, [500, 400], [560, 400]);

      expect(config.setViewport).not.toHaveBeenCalled();
      expect(onImagePress).toHaveBeenCalledTimes(1);
      expect(config.onImageDrag).toHaveBeenCalledTimes(1);
    });

    it("pans on a middle drag even when a tool owns the left button", () => {
      const config = makeConfig({ onImagePress: vi.fn() });
      const { result } = renderHook(() => useViewerPointerInteractions(config));

      drag(result, [500, 400], [560, 400], 1);

      expect(config.setViewport).toHaveBeenCalledTimes(1);
      expect(config.onImagePress).not.toHaveBeenCalled();
    });

    it("pans while space is held, and hands the button back on release", () => {
      const config = makeConfig({ onImagePress: vi.fn() });
      const { result } = renderHook(() => useViewerPointerInteractions(config));

      pressSpace();
      expect(result.current.panKeyHeld).toBe(true);
      drag(result, [500, 400], [560, 400]);
      expect(config.setViewport).toHaveBeenCalledTimes(1);
      expect(config.onImagePress).not.toHaveBeenCalled();

      releaseSpace();
      expect(result.current.panKeyHeld).toBe(false);
      drag(result, [500, 400], [560, 400]);
      expect(config.setViewport).toHaveBeenCalledTimes(1);
      expect(config.onImagePress).toHaveBeenCalledTimes(1);
    });

    it("never paints a brush stroke with the pan gesture", () => {
      const config = makeConfig({ brushMode: true });
      const { result } = renderHook(() => useViewerPointerInteractions(config));

      pressSpace();
      drag(result, [500, 400], [560, 420]);

      expect(config.drawState.startBrushStroke).not.toHaveBeenCalled();
      expect(config.drawState.appendBrushStroke).not.toHaveBeenCalled();
      expect(config.setViewport).toHaveBeenCalledTimes(1);
    });

    it("swallows the click that ends a space pan, so nothing is labelled", () => {
      const onImageClick = vi.fn();
      const config = makeConfig({ onImageClick });
      const { result } = renderHook(() => useViewerPointerInteractions(config));

      pressSpace();
      act(() => {
        result.current.handlePointerDown(pointerEvent(500, 400));
        result.current.handlePointerUp(pointerEvent(500, 400));
        result.current.handleClick(pointerEvent(500, 400));
      });

      expect(onImageClick).not.toHaveBeenCalled();
    });
  });

  describe("the pan clamp", () => {
    it("cannot push the image off the canvas, however far the drag goes", () => {
      const config = makeConfig();
      const { result } = renderHook(() => useViewerPointerInteractions(config));

      // Far past the right-hand edge: 1 000 screen px at zoom 4 is 500 image
      // px, and repeated drags would previously accumulate without bound.
      act(() => {
        result.current.handlePointerDown(pointerEvent(900, 400));
        result.current.handlePointerMove(pointerEvent(-40000, -40000));
      });

      const [next] = config.setViewport.mock.calls[0];
      expect(next.centerX).toBeLessThanOrEqual(1);
      expect(next.centerX).toBeGreaterThanOrEqual(0);
      expect(next.centerY).toBeLessThanOrEqual(IMAGE_HEIGHT / IMAGE_WIDTH);
      expect(next.centerY).toBeGreaterThanOrEqual(0);
    });
  });

  it("resolves raster UUIDs locally while hovering and clears on leave", () => {
    const onShapeHover = vi.fn();
    const pickRasterObjectId = vi.fn(() => "confirmed-1");
    const config = makeConfig({ onShapeHover, pickRasterObjectId });
    const { result } = renderHook(() => useViewerPointerInteractions(config));

    act(() => {
      result.current.handlePointerMove(pointerEvent(500, 400));
    });
    expect(pickRasterObjectId).toHaveBeenCalledWith({ x: 500, y: 400 });
    expect(onShapeHover).toHaveBeenCalledWith("confirmed-1");

    act(() => {
      result.current.handleMouseLeave();
    });
    expect(onShapeHover).toHaveBeenLastCalledWith(null);
  });
});
