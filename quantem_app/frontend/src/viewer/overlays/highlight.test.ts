import { describe, expect, it } from "vitest";
import {
  applyOverlayHighlight,
  applyShapeHighlight,
  updateOverlayHighlight,
} from "@/viewer/overlays/highlight";

describe("viewer overlay highlight", () => {
  it("applies highlighted overlay colors", () => {
    const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    applyOverlayHighlight(polygon, true, {
      fillColor: "#111111",
      fillOpacity: 0.2,
      strokeColor: "#222222",
      strokeOpacity: 0.3,
    });

    expect(polygon.getAttribute("fill")).toBe("#00ffff");
    expect(polygon.getAttribute("stroke")).toBe("#00ffff");
  });

  it("applies base colors when highlight is removed", () => {
    const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    applyShapeHighlight(polygon, false, "#ff0000");
    expect(polygon.getAttribute("fill")).toBe("#ff0000");
    expect(polygon.getAttribute("stroke")).toBe("#ff0000");
  });

  it("updates the expected overlay shape by data-segment-id", () => {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    polygon.setAttribute("data-segment-id", "seg-1");
    svg.appendChild(polygon);

    updateOverlayHighlight(svg, true, "#ff0000", "seg-1");

    expect(polygon.getAttribute("fill")).toBe("#00ffff");
  });
});
