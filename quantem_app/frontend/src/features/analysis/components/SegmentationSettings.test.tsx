import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SegmentationSettings } from "./SegmentationSettings";
import { metricLabel } from "../metricLabel";

describe("saved segmentation settings", () => {
  it("shows the recorded model and resolution without guidance messages", () => {
    render(<SegmentationSettings images={[]} settings={[{ compartment: "mito", manual_objects: 2, runs: [{
      run_id: "original-run", pack_id: "quantem:mito", adapter_id: "adapter-a", adapter: { name: "T1D model" },
      threshold: 0.45, native_pixel_size_nm: 2.5, ran_at_nm: 8, resampled: true, n_objects: 52,
    }] }]} />);
    expect(screen.getByRole("region", { name: "Segmentation settings" })).toHaveTextContent("T1D model");
    expect(screen.getByText("Image resolution at inference (nm/px)")).toBeVisible();
    expect(screen.getByText("0.45")).toBeVisible();
    expect(screen.getByText("Yes")).toBeVisible();
    expect(screen.queryByText(/Re-run inference/)).not.toBeInTheDocument();
  });

  it("shows unknown for historical missing settings and native pixels for a known unresized run", () => {
    render(<SegmentationSettings images={[]} settings={[{ compartment: "mito", runs: [{ pack_id: "quantem:mito", run_id: "native-run", resampled: false }] }]} />);
    expect(screen.getByText("Native pixels")).toBeVisible();
    expect(screen.getAllByText("Unknown").length).toBeGreaterThan(0);
    expect(screen.getByText("No")).toBeVisible();
  });

  it("labels measurement units", () => {
    expect(metricLabel("mean_area_um2")).toBe("Mean area (µm²)");
    expect(metricLabel("perimeter_px")).toBe("Perimeter (px)");
    expect(metricLabel("area_px")).toBe("Area (px²)");
    expect(metricLabel("distance_median_nm")).toBe("Distance median (nm)");
  });
});
