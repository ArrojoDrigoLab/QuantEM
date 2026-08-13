import { describe, expect, it } from "vitest";
import { findOverlayIdAtPoint } from "@/viewer/overlays/hitTest";

describe("overlay hit testing with holes", () => {
  const overlay = {
    id: "ring",
    geometry: [
      { x: 0, y: 0 },
      { x: 100, y: 0 },
      { x: 100, y: 100 },
      { x: 0, y: 100 },
    ],
    holes: [[
      { x: 30, y: 30 },
      { x: 70, y: 30 },
      { x: 70, y: 70 },
      { x: 30, y: 70 },
    ]],
    fillColor: "#00ff00",
    fillOpacity: 0.2,
    strokeColor: "#00ff00",
    strokeOpacity: 1,
  };

  it("selects the foreground and not the excluded hole", () => {
    expect(findOverlayIdAtPoint({ x: 10, y: 10 }, [overlay])).toBe("ring");
    expect(findOverlayIdAtPoint({ x: 50, y: 50 }, [overlay])).toBeNull();
  });
});
