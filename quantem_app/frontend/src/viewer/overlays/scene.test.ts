import { describe, expect, it } from "vitest";
import type { SegmentOverlay } from "@/viewer/types";
import { findOverlayIdAtPoint, findSceneOverlayIdAtPoint } from "@/viewer/overlays/hitTest";
import {
  composeOverlayScene,
  sceneToOverlayList,
} from "@/viewer/overlays/scene";

function makeOverlay(id: string, x: number, y: number): SegmentOverlay {
  return {
    id,
    geometry: [
      { x, y },
      { x: x + 10, y },
      { x: x + 10, y: y + 10 },
      { x, y: y + 10 },
      { x, y },
    ],
    fillColor: "#ff0000",
    fillOpacity: 0.1,
    strokeColor: "#ff0000",
    strokeOpacity: 0.7,
    strokeWidth: 2,
  };
}

describe("overlay scene helpers", () => {
  it("composes persistent and transient layers into a stable scene", () => {
    const persistentA = makeOverlay("persistent-a", 0, 0);
    const persistentB = makeOverlay("persistent-b", 20, 20);
    const transientA = makeOverlay("transient-a", 40, 40);
    const transientB = makeOverlay("transient-b", 60, 60);

    const scene = composeOverlayScene({
      persistentLayers: [[persistentA], [persistentB]],
      transientLayers: [[transientA], [transientB]],
    });

    expect(scene.persistent.map((overlay) => overlay.id)).toEqual([
      "persistent-a",
      "persistent-b",
    ]);
    expect(scene.transient.map((overlay) => overlay.id)).toEqual([
      "transient-a",
      "transient-b",
    ]);
    expect(sceneToOverlayList(scene).map((overlay) => overlay.id)).toEqual([
      "persistent-a",
      "persistent-b",
      "transient-a",
      "transient-b",
    ]);
  });

});

describe("overlay hit testing", () => {
  it("finds hits from a flat overlay list", () => {
    const overlays = [makeOverlay("one", 0, 0), makeOverlay("two", 20, 20)];
    expect(findOverlayIdAtPoint({ x: 5, y: 5 }, overlays)).toBe("one");
    expect(findOverlayIdAtPoint({ x: 25, y: 25 }, overlays)).toBe("two");
    expect(findOverlayIdAtPoint({ x: 100, y: 100 }, overlays)).toBeNull();
  });

  it("finds hits across persistent and transient scene overlays", () => {
    const scene = composeOverlayScene({
      persistentLayers: [[makeOverlay("persistent", 0, 0)]],
      transientLayers: [[makeOverlay("transient", 20, 20)]],
    });

    expect(findSceneOverlayIdAtPoint({ x: 2, y: 2 }, scene)).toBe("persistent");
    expect(findSceneOverlayIdAtPoint({ x: 22, y: 22 }, scene)).toBe("transient");
  });
});
