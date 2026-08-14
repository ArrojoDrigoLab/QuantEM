import { afterEach, describe, expect, it, vi } from "vitest";
import {
  IdMapLabelLayer,
  type IdMapLabelLayerProps,
} from "@/viewer/components/internal/IdMapLabelLayer";

vi.mock("@deck.gl/core", () => ({
  CompositeLayer: class CompositeLayer {
    props: IdMapLabelLayerProps;

    constructor(props: IdMapLabelLayerProps) {
      this.props = props;
    }

    getSubLayerProps(props: Record<string, unknown>) {
      return props;
    }
  },
}));

vi.mock("@deck.gl/layers", () => ({
  BitmapLayer: class BitmapLayer {},
}));

vi.mock("@hms-dbmi/viv", () => ({
  MultiscaleImageLayer: class MultiscaleImageLayer {
    props: Record<string, unknown>;

    constructor(props: Record<string, unknown>) {
      this.props = props;
    }
  },
}));

interface SharedTile {
  data: Uint32Array[];
  width: number;
  height: number;
}

function makeLayer(id: string, color: [number, number, number]): IdMapLabelLayer {
  return new IdMapLabelLayer({
    id,
    labelsData: [],
    lut: new Uint8Array([0, 0, 0, 0, ...color, 255]),
    maxLabel: 1,
    lutRevision: 7,
    visualRevision: 0,
    highlightedSegmentId: null,
    highlightRevision: 0,
    imageWidth: 1,
    imageHeight: 1,
    fillOpacity: 0.5,
    borderOpacity: 1,
    showBorders: false,
  });
}

function makeRenderableLayer(
  fillOpacity: number,
  showBorders: boolean
): IdMapLabelLayer {
  return new IdMapLabelLayer({
    id: "review-layer",
    labelsData: [{ dtype: "uint32" }] as unknown as IdMapLabelLayerProps["labelsData"],
    lut: new Uint8Array([0, 0, 0, 0, 0, 255, 0, 255]),
    maxLabel: 1,
    lutRevision: 7,
    visualRevision: 0,
    highlightedSegmentId: null,
    highlightRevision: 0,
    imageWidth: 1,
    imageHeight: 1,
    fillOpacity,
    borderOpacity: 0.9,
    showBorders,
  });
}

function colorize(layer: IdMapLabelLayer, tile: SharedTile): ImageData {
  return (
    layer as unknown as {
      _colorize: (sourceTile: SharedTile) => ImageData;
    }
  )._colorize(tile);
}

describe("IdMapLabelLayer", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("keeps colorized tile caches separate for overlays sharing one source tile", () => {
    vi.stubGlobal(
      "ImageData",
      class ImageData {
        data: Uint8ClampedArray;
        width: number;
        height: number;

        constructor(data: Uint8ClampedArray, width: number, height: number) {
          this.data = data;
          this.width = width;
          this.height = height;
        }
      }
    );
    const sharedTile: SharedTile = {
      data: [new Uint32Array([1])],
      width: 1,
      height: 1,
    };

    const candidateImage = colorize(
      makeLayer("candidate-layer", [255, 0, 0]),
      sharedTile
    );
    const confirmedImage = colorize(
      makeLayer("confirmed-layer", [0, 255, 0]),
      sharedTile
    );

    expect(Array.from(candidateImage.data)).toEqual([255, 0, 0, 128]);
    expect(Array.from(confirmedImage.data)).toEqual([0, 255, 0, 128]);
  });

  it("changes Viv's no-refetch render revision when visual controls change", () => {
    const first = makeRenderableLayer(0.2, true).renderLayers() as unknown as {
      props: { contrastLimits: Array<[number, number]> };
    };
    const second = makeRenderableLayer(0, false).renderLayers() as unknown as {
      props: { contrastLimits: Array<[number, number]> };
    };

    expect(first.props.contrastLimits).toEqual([
      [7, 0.2],
      [0.9, 1],
      [0, 0],
      [0, 0],
    ]);
    expect(second.props.contrastLimits).toEqual([
      [7, 0],
      [0.9, 0],
      [0, 0],
      [0, 0],
    ]);
  });

  it("highlights a picked UUID in cyan without changing the LUT", () => {
    vi.stubGlobal(
      "ImageData",
      class ImageData {
        data: Uint8ClampedArray;
        width: number;
        height: number;

        constructor(data: Uint8ClampedArray, width: number, height: number) {
          this.data = data;
          this.width = width;
          this.height = height;
        }
      }
    );
    const layer = makeLayer("confirmed-layer", [0, 255, 0]);
    layer.props.highlightedSegmentId = "object-1";
    layer.props.highlightRevision = 1;
    layer.props.pickMap = new Map([[1, "object-1"]]);

    const image = colorize(layer, {
      data: [new Uint32Array([1])],
      width: 1,
      height: 1,
    });

    expect(Array.from(image.data)).toEqual([0, 255, 255, 128]);
  });
});
