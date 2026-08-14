import { act, fireEvent, render, waitFor } from "@testing-library/react";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ImageViewer } from "@/viewer/components/ImageViewer";
import { resetVivOmeZarrCacheForTests } from "@/viewer/imageViewerCache";
import type {
  ViewerIdMapOverlaySpec,
  ViewerNgffOverlayLayerSpec,
} from "@/viewer/types";

type MockVivSource = {
  labels: string[];
  shape: number[];
  dtype: string;
};

type MockVivResult = {
  data: MockVivSource[];
};

type MockLayerInstance = {
  id: string;
  props: Record<string, unknown>;
};

type MockDeckProps = {
  layers?: MockLayerInstance[];
};

const { deckPropsSpy, loadOmeZarrMock } = vi.hoisted(() => ({
  deckPropsSpy: vi.fn<(props: MockDeckProps) => void>(),
  loadOmeZarrMock: vi.fn<(url: string) => Promise<MockVivResult>>(),
}));

vi.mock("@deck.gl/react", () => ({
  DeckGL: React.forwardRef<object, MockDeckProps>((props, ref) => {
    React.useImperativeHandle(ref, () => ({
      pickObject: () => null,
    }));
    deckPropsSpy(props);
    return <div data-testid="deckgl" />;
  }),
}));

vi.mock("@deck.gl/core", () => ({
  OrthographicView: class OrthographicView {},
  // IdMapLabelLayer (pulled in transitively via buildViewerDeckLayers) extends
  // CompositeLayer at module-eval time, so the mock must provide it.
  CompositeLayer: class CompositeLayer {
    id: string;
    props: Record<string, unknown>;

    constructor(props: Record<string, unknown>) {
      this.id = String(props.id);
      this.props = props;
    }
  },
}));

vi.mock("@hms-dbmi/viv", () => ({
  loadOmeZarr: loadOmeZarrMock,
  getImageSize: vi.fn(() => ({ width: 256, height: 256 })),
  MultiscaleImageLayer: class MockMultiscaleImageLayer {
    id: string;
    props: Record<string, unknown>;

    constructor(props: Record<string, unknown>) {
      this.id = String(props.id);
      this.props = props;
    }
  },
}));

function makeVivResult(channelCount = 1): MockVivResult {
  return {
    data: [
      {
        labels: ["c", "y", "x"],
        shape: [channelCount, 256, 256],
        dtype: "uint8",
      },
    ],
  };
}

function makeVivResult3d(depth = 20): MockVivResult {
  return {
    data: [
      {
        labels: ["c", "z", "y", "x"],
        shape: [1, depth, 256, 256],
        dtype: "uint8",
      },
    ],
  };
}

function makeOverlaySpec(id: string, ngffUrl: string): ViewerNgffOverlayLayerSpec {
  return {
    id,
    ngffUrl,
    color: "#33cc66",
    opacity: 0.4,
    channelIndices: [0],
  };
}

function makeIdMapSpec(revision: number): ViewerIdMapOverlaySpec {
  return {
    id: "review-id-map",
    ngffUrl: `/review.zarr?rev=1-${revision}`,
    revision,
    lut: new Uint8Array(8),
    maxLabel: 1,
    lutRevision: revision,
    fillOpacity: 0.25,
    borderOpacity: 0.95,
    showBorders: true,
  };
}

function getLayerIds(): string[] {
  const lastCall = deckPropsSpy.mock.calls.at(-1)?.[0];
  return (lastCall?.layers ?? []).map((layer) => layer.id);
}

function getLayerById(id: string): MockLayerInstance {
  const lastCall = deckPropsSpy.mock.calls.at(-1)?.[0];
  const layer = (lastCall?.layers ?? []).find((entry) => entry.id === id);
  if (!layer) {
    throw new Error(`Layer ${id} was not rendered.`);
  }
  return layer;
}

function createDeferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function getViewerEventTarget(container: HTMLElement): SVGElement {
  const eventTarget = container.querySelector(".image-viewer svg");
  if (!(eventTarget instanceof SVGElement)) {
    throw new Error("Viewer event target was not rendered.");
  }
  return eventTarget;
}

describe("ImageViewer", () => {
  beforeEach(() => {
    resetVivOmeZarrCacheForTests();
    deckPropsSpy.mockReset();
    loadOmeZarrMock.mockReset();
    if (!Element.prototype.setPointerCapture) {
      Element.prototype.setPointerCapture = vi.fn();
    }
    vi.stubGlobal(
      "ResizeObserver",
      class MockResizeObserver {
        private readonly callback: ResizeObserverCallback;

        constructor(callback: ResizeObserverCallback) {
          this.callback = callback;
        }

        observe() {
          this.callback(
            [
              {
                contentRect: {
                  width: 400,
                  height: 400,
                },
              } as ResizeObserverEntry,
            ],
            this as unknown as ResizeObserver
          );
        }

        disconnect() {}

        unobserve() {}
      }
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("keeps the current overlay visible until the replacement revision loads", async () => {
    const nextOverlayDeferred = createDeferred<MockVivResult>();

    loadOmeZarrMock.mockImplementation((url: string) => {
      if (url === "/image.zarr") {
        return Promise.resolve(makeVivResult());
      }
      if (url === "/overlay.zarr?rev=1") {
        return Promise.resolve(makeVivResult());
      }
      if (url === "/overlay.zarr?rev=2") {
        return nextOverlayDeferred.promise;
      }
      return Promise.reject(new Error(`Unexpected URL: ${url}`));
    });

    const { rerender } = render(
      <ImageViewer
        image={{ ngffUrl: "/image.zarr", width: 256, height: 256 }}
        overlays={{ rasterLayers: [makeOverlaySpec("rev-1", "/overlay.zarr?rev=1")] }}
      />
    );

    await waitFor(() => {
      expect(getLayerIds()).toContain("viv-overlay-rev-1");
    });

    rerender(
      <ImageViewer
        image={{ ngffUrl: "/image.zarr", width: 256, height: 256 }}
        overlays={{ rasterLayers: [makeOverlaySpec("rev-2", "/overlay.zarr?rev=2")] }}
      />
    );

    await waitFor(() => {
      expect(loadOmeZarrMock).toHaveBeenCalledWith("/overlay.zarr?rev=2", {
        type: "multiscales",
      });
    });

    expect(getLayerIds()).toContain("viv-overlay-rev-1");
    expect(getLayerIds()).not.toContain("viv-overlay-rev-2");

    await act(async () => {
      nextOverlayDeferred.resolve(makeVivResult());
      await nextOverlayDeferred.promise;
    });

    await waitFor(() => {
      expect(getLayerIds()).toContain("viv-overlay-rev-2");
      expect(getLayerIds()).not.toContain("viv-overlay-rev-1");
    });
  });

  it("reports an ID-map revision only after that replacement has loaded", async () => {
    const nextRevision = createDeferred<MockVivResult>();
    const onRevisionDisplayed = vi.fn();
    loadOmeZarrMock.mockImplementation((url: string) => {
      if (url === "/image.zarr") return Promise.resolve(makeVivResult());
      if (url.includes("/review.zarr/") && url.endsWith("?rev=1-5")) {
        return Promise.resolve(makeVivResult());
      }
      if (url.includes("/review.zarr/") && url.endsWith("?rev=1-6")) {
        return nextRevision.promise;
      }
      return Promise.reject(new Error(`Unexpected URL: ${url}`));
    });

    const { rerender } = render(
      <ImageViewer
        image={{ ngffUrl: "/image.zarr", width: 256, height: 256 }}
        overlays={{
          idMapOverlays: [makeIdMapSpec(5)],
          onRasterRevisionDisplayed: onRevisionDisplayed,
        }}
      />
    );

    await waitFor(() => expect(onRevisionDisplayed).toHaveBeenCalledWith(5));
    rerender(
      <ImageViewer
        image={{ ngffUrl: "/image.zarr", width: 256, height: 256 }}
        overlays={{
          idMapOverlays: [makeIdMapSpec(6)],
          onRasterRevisionDisplayed: onRevisionDisplayed,
        }}
      />
    );

    await waitFor(() =>
      expect(loadOmeZarrMock).toHaveBeenCalledWith(
        "/review.zarr/labels?rev=1-6",
        { type: "multiscales" }
      )
    );
    expect(onRevisionDisplayed).not.toHaveBeenCalledWith(6);

    await act(async () => {
      nextRevision.resolve(makeVivResult());
      await nextRevision.promise;
    });
    await waitFor(() => expect(onRevisionDisplayed).toHaveBeenCalledWith(6));
  });

  it("shares OME-Zarr loads across viewers for identical URLs", async () => {
    loadOmeZarrMock.mockResolvedValue(makeVivResult());

    render(
      <>
        <ImageViewer
          image={{ ngffUrl: "/image.zarr", width: 256, height: 256 }}
          overlays={{ rasterLayers: [makeOverlaySpec("left", "/overlay.zarr?rev=1")] }}
        />
        <ImageViewer
          image={{ ngffUrl: "/image.zarr", width: 256, height: 256 }}
          overlays={{ rasterLayers: [makeOverlaySpec("right", "/overlay.zarr?rev=1")] }}
        />
      </>
    );

    await waitFor(() => {
      expect(loadOmeZarrMock).toHaveBeenCalledTimes(2);
    });
    expect(loadOmeZarrMock.mock.calls).toEqual(
      expect.arrayContaining([
        ["/image.zarr", { type: "multiscales" }],
        ["/overlay.zarr?rev=1", { type: "multiscales" }],
      ])
    );
  });

  it("renders rgb overlays with identity channel colors and transparent black", async () => {
    loadOmeZarrMock.mockImplementation((url: string) => {
      if (url === "/image.zarr") {
        return Promise.resolve(makeVivResult());
      }
      if (url === "/overlay-rgb.zarr") {
        return Promise.resolve(makeVivResult(3));
      }
      return Promise.reject(new Error(`Unexpected URL: ${url}`));
    });

    render(
      <ImageViewer
        image={{ ngffUrl: "/image.zarr", width: 256, height: 256 }}
        overlays={{
          rasterLayers: [
            {
              id: "debug-rgb",
              ngffUrl: "/overlay-rgb.zarr",
              color: "#ffffff",
              opacity: 0.65,
              channelIndices: [0, 1, 2],
              renderMode: "rgb",
            },
          ],
        }}
      />
    );

    await waitFor(() => {
      expect(getLayerIds()).toContain("viv-overlay-debug-rgb");
    });

    const overlayLayer = getLayerById("viv-overlay-debug-rgb");
    expect(overlayLayer.props.colors).toEqual([
      [255, 0, 0],
      [0, 255, 0],
      [0, 0, 255],
    ]);
    expect(overlayLayer.props.contrastLimits).toEqual([
      [0, 255],
      [0, 255],
      [0, 255],
    ]);
    expect(overlayLayer.props.transparentColor).toEqual([0, 0, 0]);
    expect(overlayLayer.props.useTransparentColor).toBe(true);
  });

  it("renders tint overlays with per-channel colors when provided", async () => {
    loadOmeZarrMock.mockImplementation((url: string) => {
      if (url === "/image.zarr") {
        return Promise.resolve(makeVivResult());
      }
      if (url === "/overlay-tint.zarr") {
        return Promise.resolve(makeVivResult(2));
      }
      return Promise.reject(new Error(`Unexpected URL: ${url}`));
    });

    render(
      <ImageViewer
        image={{ ngffUrl: "/image.zarr", width: 256, height: 256 }}
        overlays={{
          rasterLayers: [
            {
              id: "debug-tint",
              ngffUrl: "/overlay-tint.zarr",
              color: "#ffffff",
              channelColors: ["#33cc66", "#3b82f6"],
              opacity: 0.65,
              channelIndices: [0, 1],
            },
          ],
        }}
      />
    );

    await waitFor(() => {
      expect(getLayerIds()).toContain("viv-overlay-debug-tint");
    });

    const overlayLayer = getLayerById("viv-overlay-debug-tint");
    expect(overlayLayer.props.colors).toEqual([
      [51, 204, 102],
      [59, 130, 246],
    ]);
  });

  it("keeps bitmap probability previews below confirmed ID-map labels", async () => {
    loadOmeZarrMock.mockResolvedValue(makeVivResult());

    render(
      <ImageViewer
        image={{ ngffUrl: "/image.zarr", width: 256, height: 256 }}
        overlays={{
          bitmapOverlays: [
            {
              id: "threshold-preview",
              image: document.createElement("canvas"),
              bounds: [0, 0, 256, 256],
              opacity: 1,
            },
          ],
          idMapOverlays: [makeIdMapSpec(1)],
        }}
      />
    );

    await waitFor(() => {
      expect(getLayerIds()).toContain("idmap-overlay-review-id-map");
    });
    const layerIds = getLayerIds();
    expect(layerIds.indexOf("viewer-bitmap-threshold-preview")).toBeLessThan(
      layerIds.indexOf("idmap-overlay-review-id-map")
    );
  });

  it("does not render a z-slider for 2D images", async () => {
    loadOmeZarrMock.mockResolvedValue(makeVivResult());
    const { container } = render(
      <ImageViewer image={{ ngffUrl: "/image.zarr", width: 256, height: 256 }} />
    );
    await waitFor(() => {
      expect(getLayerIds()).toContain("viv-multiscale-image");
    });
    expect(container.querySelector(".viewer-z-slider")).toBeNull();
  });

  it("renders a z-slider for 3D volumes and updates the selected plane", async () => {
    loadOmeZarrMock.mockResolvedValue(makeVivResult3d(20));

    const { container } = render(
      <ImageViewer
        image={{
          ngffUrl: "/volume.zarr",
          width: 256,
          height: 256,
          zPlaneIndices: Array.from({ length: 20 }, (_, i) => i * 2),
        }}
      />
    );

    await waitFor(() => {
      expect(getLayerIds()).toContain("viv-multiscale-image");
    });

    // Slider present, base image starts at z=0.
    const slider = container.querySelector(".viewer-z-slider__input") as HTMLInputElement;
    expect(slider).not.toBeNull();
    expect(slider.max).toBe("19");
    expect(getLayerById("viv-multiscale-image").props.selections).toEqual([{ z: 0 }]);

    // Scrubbing updates the viv selection for the base image layer.
    await act(async () => {
      fireEvent.change(slider, { target: { value: "5" } });
    });

    await waitFor(() => {
      expect(getLayerById("viv-multiscale-image").props.selections).toEqual([{ z: 5 }]);
    });
  });

  it("prevents page scrolling while wheel-zooming inside the viewer", async () => {
    loadOmeZarrMock.mockResolvedValue(makeVivResult());

    const { container } = render(
      <ImageViewer image={{ ngffUrl: "/image.zarr", width: 256, height: 256 }} />
    );

    await waitFor(() => {
      expect(getLayerIds()).toContain("viv-multiscale-image");
    });

    const wheelEvent = new WheelEvent("wheel", {
      bubbles: true,
      cancelable: true,
      clientX: 100,
      clientY: 100,
      deltaY: 120,
    });
    const preventDefaultSpy = vi.spyOn(wheelEvent, "preventDefault");
    const stopPropagationSpy = vi.spyOn(wheelEvent, "stopPropagation");

    getViewerEventTarget(container).dispatchEvent(wheelEvent);

    expect(preventDefaultSpy).toHaveBeenCalled();
    expect(stopPropagationSpy).toHaveBeenCalled();
  });

  it("still treats minor pointer jitter as a click", async () => {
    loadOmeZarrMock.mockResolvedValue(makeVivResult());
    const onImageClick = vi.fn();

    const { container } = render(
      <ImageViewer
        image={{ ngffUrl: "/image.zarr", width: 256, height: 256 }}
        viewport={{ disablePan: true }}
        interactions={{ onImageClick }}
      />
    );

    await waitFor(() => {
      expect(getLayerIds()).toContain("viv-multiscale-image");
    });

    const eventTarget = getViewerEventTarget(container);

    firePointerEvent(eventTarget, "pointerDown", { clientX: 100, clientY: 100, pointerId: 1 });
    firePointerEvent(eventTarget, "pointerMove", { clientX: 102, clientY: 101, pointerId: 1 });
    firePointerEvent(eventTarget, "pointerUp", { clientX: 102, clientY: 101, pointerId: 1 });
    fireMouseEvent(eventTarget, "click", { clientX: 102, clientY: 101 });

    expect(onImageClick).toHaveBeenCalledTimes(1);
  });

  /**
   * The canvas bar is mounted here rather than by each screen, which is the
   * whole reason both `/viewer` and `/labeling` get it: they both mount this
   * component and neither has to know the bar exists.
   */
  it("carries the view controls, so every screen that mounts the canvas gets them", async () => {
    loadOmeZarrMock.mockResolvedValue(makeVivResult());

    const { container } = render(
      <ImageViewer image={{ ngffUrl: "/image.zarr", width: 256, height: 256 }} />
    );

    await waitFor(() => {
      expect(getLayerIds()).toContain("viv-multiscale-image");
    });

    const labels = Array.from(
      container.querySelectorAll(".viewer-view-controls button")
    ).map((button) => button.textContent);
    expect(labels).toEqual(["Fit", "1:1", "Reset"]);
  });

  it("draws a scale bar when the image is calibrated, and none when it is not", async () => {
    loadOmeZarrMock.mockResolvedValue(makeVivResult());

    const calibrated = render(
      <ImageViewer
        image={{ ngffUrl: "/image.zarr", width: 256, height: 256, pixelSizeNm: 5 }}
      />
    );
    await waitFor(() => {
      expect(getLayerIds()).toContain("viv-multiscale-image");
    });
    expect(
      calibrated.container.querySelector(".viewer-scale-bar-label")?.textContent
    ).toMatch(/(nm|µm|mm)$/);
    calibrated.unmount();

    const uncalibrated = render(
      <ImageViewer image={{ ngffUrl: "/image.zarr", width: 256, height: 256 }} />
    );
    await waitFor(() => {
      expect(getLayerIds()).toContain("viv-multiscale-image");
    });
    expect(uncalibrated.container.querySelector(".viewer-scale-bar")).toBeNull();
  });

  it("says how to pan only once a tool has taken the left button", async () => {
    loadOmeZarrMock.mockResolvedValue(makeVivResult());

    const plain = render(
      <ImageViewer image={{ ngffUrl: "/image.zarr", width: 256, height: 256 }} />
    );
    await waitFor(() => {
      expect(getLayerIds()).toContain("viv-multiscale-image");
    });
    expect(plain.container.querySelector(".viewer-pan-hint")).toBeNull();
    plain.unmount();

    const armed = render(
      <ImageViewer
        image={{ ngffUrl: "/image.zarr", width: 256, height: 256 }}
        interactions={{ onImageClick: vi.fn() }}
      />
    );
    await waitFor(() => {
      expect(getLayerIds()).toContain("viv-multiscale-image");
    });
    expect(armed.container.querySelector(".viewer-pan-hint")?.textContent).toBe(
      "Hold space or the middle button to move the image"
    );
  });
});

function firePointerEvent(
  target: Element,
  type: "pointerDown" | "pointerMove" | "pointerUp",
  init: PointerEventInit
) {
  const event = new MouseEvent(type.toLowerCase(), {
    bubbles: true,
    cancelable: true,
    clientX: init.clientX ?? 0,
    clientY: init.clientY ?? 0,
    ...init,
  });
  Object.defineProperty(event, "pointerId", {
    configurable: true,
    value: init.pointerId ?? 1,
  });
  Object.defineProperty(event, "pointerType", {
    configurable: true,
    value: init.pointerType ?? "mouse",
  });
  target.dispatchEvent(event);
}

function fireMouseEvent(target: Element, type: "click", init: MouseEventInit) {
  const event = new MouseEvent(type, {
    bubbles: true,
    cancelable: true,
    ...init,
  });
  target.dispatchEvent(event);
}
