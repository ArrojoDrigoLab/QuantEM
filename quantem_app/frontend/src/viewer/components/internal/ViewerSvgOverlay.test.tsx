import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ViewerSvgOverlay } from "@/viewer/components/internal/ViewerSvgOverlay";

describe("ViewerSvgOverlay", () => {
  it("keeps polyline stroke widths non-scaling like polygon overlays", () => {
    const { container } = render(
      <ViewerSvgOverlay
        metrics={{
          imageWidth: 256,
          imageHeight: 256,
          containerWidth: 256,
          containerHeight: 256,
          visibleWidth: 256,
          visibleHeight: 256,
          minX: 0,
          minY: 0,
        }}
        persistentOverlays={[
          {
            id: "polyline-1",
            geometry: [
              { x: 10, y: 10 },
              { x: 50, y: 50 },
            ],
            fillColor: "transparent",
            fillOpacity: 0,
            strokeColor: "#ff0000",
            strokeOpacity: 1,
            strokeWidth: 2.5,
            shape: "polyline",
          },
        ]}
        transientOverlays={[]}
      />
    );

    const polyline = container.querySelector("polyline");
    expect(polyline).toHaveAttribute("vector-effect", "non-scaling-stroke");
  });

  it("renders dashed overlay outlines", () => {
    const { container } = render(
      <ViewerSvgOverlay
        metrics={null}
        persistentOverlays={[
          {
            id: "dashed-box",
            geometry: [
              { x: 0, y: 0 },
              { x: 1, y: 0 },
              { x: 1, y: 1 },
              { x: 0, y: 1 },
            ],
            fillColor: "transparent",
            fillOpacity: 0,
            strokeColor: "#ffd166",
            strokeOpacity: 0.4,
            strokeWidth: 3,
            strokeDasharray: "8 6",
          },
        ]}
        transientOverlays={[]}
      />
    );

    expect(container.querySelector("polygon")).toHaveAttribute(
      "stroke-dasharray",
      "8 6"
    );
  });
});
