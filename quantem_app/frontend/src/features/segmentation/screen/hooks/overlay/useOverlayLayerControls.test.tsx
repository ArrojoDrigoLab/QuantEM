import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useOverlayLayerControls } from "@/features/segmentation/screen/hooks/overlay/useOverlayLayerControls";
import type { SegmentationOverlayManifest } from "@/shared/types/segmentation";

vi.mock("@/features/segmentation/screen/hooks/overlay/useOverlayLut", () => ({
  useOverlayLut: ({ hiddenStates = [] }: { hiddenStates?: string[] }) => ({
    rgba: new Uint8Array([
      0,
      0,
      0,
      0,
      hiddenStates.includes("confirmed") ? 1 : 2,
      0,
      0,
      255,
    ]),
    maxLabel: 1,
    lutRevision: 7,
  }),
  useOverlayPickMap: () => new Map([[1, "object-1"]]),
}));

function manifest(): SegmentationOverlayManifest {
  return {
    status: "READY",
    ngff_url: "/ngff/overlay.zarr",
    lut_url: "/api/segmentations/seg-1/overlay-lut/",
    arrays: ["labels", "border"],
    label_dtype: "uint32",
    source_model: "quantem:mito",
    bundle_version: 4,
    applied_revision: 8,
    desired_revision: 8,
    lut_revision: 7,
    chunk_size: [256, 256],
    level_count: 4,
    width: 1024,
    height: 1024,
  };
}

describe("useOverlayLayerControls", () => {
  it("builds independent candidate, left-confirmed, and right-confirmed raster layers", () => {
    const { result } = renderHook(() =>
      useOverlayLayerControls({
        segmentationId: "seg-1",
        modelManifest: manifest(),
        confirmedManifest: manifest(),
      })
    );

    expect(result.current.leftIdMapOverlays.map((overlay) => overlay.id)).toEqual([
      "label-left-candidates-idmap",
      "label-left-confirmed-idmap",
    ]);
    expect(result.current.rightIdMapOverlays.map((overlay) => overlay.id)).toEqual([
      "label-right-confirmed-idmap",
    ]);
    expect(result.current.leftIdMapOverlays[0].lut[4]).toBe(1);
    expect(result.current.leftIdMapOverlays[1].lut[4]).toBe(2);
    expect(result.current.rightIdMapOverlays[0].lut[4]).toBe(2);

    act(() => {
      result.current.updateLayerStyles.setCandidateFillOpacity(0);
      result.current.updateLayerStyles.setConfirmedFillOpacity(0.4);
      result.current.updateRightLayerStyle.setFillOpacity(0.75);
      result.current.setShowCandidateBorders(false);
      result.current.setShowRightConfirmedBorders(false);
    });

    expect(result.current.leftIdMapOverlays[0]).toMatchObject({
      fillOpacity: 0,
      showBorders: false,
    });
    expect(result.current.leftIdMapOverlays[1]).toMatchObject({
      fillOpacity: 0.4,
      showBorders: true,
    });
    expect(result.current.rightIdMapOverlays[0]).toMatchObject({
      fillOpacity: 0.75,
      showBorders: false,
    });
  });

  it("hides an optimistically deleted UUID from both panes before the server LUT changes", () => {
    const { result } = renderHook(() =>
      useOverlayLayerControls({
        segmentationId: "seg-1",
        modelManifest: manifest(),
        confirmedManifest: manifest(),
        hiddenSegmentIds: new Set(["object-1"]),
        hiddenSegmentVisualRevision: 3,
      })
    );

    expect(result.current.leftIdMapOverlays[0].lut[7]).toBe(0);
    expect(result.current.leftIdMapOverlays[1].lut[7]).toBe(0);
    expect(result.current.rightIdMapOverlays[0].lut[7]).toBe(0);
    expect(result.current.leftIdMapOverlays[0].visualRevision).toBe(3);
    expect(result.current.rightIdMapOverlays[0].visualRevision).toBe(3);
  });
});
