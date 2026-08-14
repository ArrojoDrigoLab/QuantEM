import { describe, expect, it } from "vitest";
import {
  generateLeftPanelOverlays,
  generateRightPanelOverlays,
} from "@/features/segmentation/overlays/segments";
import type { SegmentObject } from "@/shared/types/segmentation";

function makeSegment(id: string, label: SegmentObject["label_state"]): SegmentObject {
  return {
    id,
    segmentation: "seg-1",
    label_state: label,
    confidence_score: 0.75,
    geometry_coords: [
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10],
    ],
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
}

describe("segmentation overlay segments", () => {
  it("maps left-panel colors and opacities by label state", () => {
    const overlays = generateLeftPanelOverlays(
      [
        makeSegment("c", "CONFIRMED"),
        makeSegment("e", "EXCLUDED"),
        makeSegment("i", "INFERRED"),
      ],
      false
    );

    expect(overlays).toHaveLength(3);
    expect(overlays[0]).toMatchObject({ id: "i", fillColor: "#ff0000", fillOpacity: 0 });
    expect(overlays[1]).toMatchObject({ id: "e", fillColor: "#5c677d", fillOpacity: 0.05 });
    expect(overlays[2]).toMatchObject({ id: "c", fillColor: "#33cc66", fillOpacity: 0.15 });
  });

  it("always paints confirmed outlines after overlapping candidates", () => {
    const overlays = generateLeftPanelOverlays(
      [makeSegment("confirmed", "CONFIRMED"), makeSegment("candidate", "CANDIDATE")],
      false
    );

    expect(overlays.map((overlay) => overlay.id)).toEqual(["candidate", "confirmed"]);

    const right = generateRightPanelOverlays(
      [makeSegment("confirmed", "CONFIRMED")],
      [makeSegment("candidate", "INFERRED")],
      [],
      null
    );
    expect(right.map((overlay) => overlay.id)).toEqual(["candidate", "confirmed"]);
  });

  it("applies custom left-panel layer styles", () => {
    const overlays = generateLeftPanelOverlays(
      [makeSegment("candidate", "CANDIDATE"), makeSegment("confirmed", "CONFIRMED")],
      false,
      undefined,
      undefined,
      false,
      {
        candidateStrokeWidth: 3.5,
        candidateFillOpacity: 0.2,
        confirmedStrokeWidth: 5,
        confirmedFillOpacity: 0.45,
      }
    );

    expect(overlays.find((overlay) => overlay.id === "candidate")).toMatchObject({
      strokeWidth: 3.5,
      fillOpacity: 0.2,
    });
    expect(overlays.find((overlay) => overlay.id === "confirmed")).toMatchObject({
      strokeWidth: 5,
      fillOpacity: 0.45,
    });
  });

  it("keeps bbox-highlighted left-panel stroke overrides fixed", () => {
    const overlays = generateLeftPanelOverlays(
      [makeSegment("candidate", "CANDIDATE")],
      false,
      undefined,
      new Set(["candidate"]),
      false,
      {
        candidateStrokeWidth: 7,
        candidateFillOpacity: 0.25,
        confirmedStrokeWidth: 7,
        confirmedFillOpacity: 0.5,
      }
    );

    expect(overlays[0]).toMatchObject({
      strokeColor: "#00ffff",
      strokeWidth: 4,
      fillOpacity: 0.25,
    });
  });

  it("applies an independent right-panel confirmed style, including zero fill", () => {
    const overlays = generateRightPanelOverlays(
      [makeSegment("confirmed", "CONFIRMED")],
      [],
      [],
      null,
      undefined,
      undefined,
      false,
      { strokeWidth: 3.5, fillOpacity: 0 }
    );

    expect(overlays[0]).toMatchObject({
      id: "confirmed",
      strokeWidth: 3.5,
      fillOpacity: 0,
    });
  });

  it("highlights selected and bbox-highlighted right-panel segments", () => {
    const overlays = generateRightPanelOverlays(
      [makeSegment("confirmed", "CONFIRMED")],
      [makeSegment("inferred", "INFERRED")],
      [makeSegment("excluded", "EXCLUDED")],
      "inferred",
      undefined,
      new Set(["inferred"])
    );

    expect(overlays.find((overlay) => overlay.id === "inferred")).toMatchObject({
      strokeColor: "#00ffff",
      strokeWidth: 8,
    });
  });

  it("uses deterministic per-object colors for mito overlays", () => {
    const overlaysA = generateRightPanelOverlays(
      [makeSegment("obj-1", "CONFIRMED"), makeSegment("obj-2", "CONFIRMED")],
      [],
      [makeSegment("obj-ex", "EXCLUDED")],
      null,
      "quantem_internal_mito"
    );
    const overlaysB = generateRightPanelOverlays(
      [makeSegment("obj-1", "CONFIRMED"), makeSegment("obj-2", "CONFIRMED")],
      [],
      [makeSegment("obj-ex", "EXCLUDED")],
      null,
      "quantem_internal_mito"
    );

    const firstStroke = overlaysA.find((overlay) => overlay.id === "obj-1")?.strokeColor;
    const secondStroke = overlaysA.find((overlay) => overlay.id === "obj-2")?.strokeColor;
    const firstStrokeAgain = overlaysB.find((overlay) => overlay.id === "obj-1")?.strokeColor;

    expect(firstStroke).toBeDefined();
    expect(secondStroke).toBeDefined();
    expect(firstStroke).not.toBe(secondStroke);
    expect(firstStrokeAgain).toBe(firstStroke);
  });
});
