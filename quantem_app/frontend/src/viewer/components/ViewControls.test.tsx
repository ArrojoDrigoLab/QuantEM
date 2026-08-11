import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ScaleBar } from "@/viewer/components/ScaleBar";
import { ViewControls } from "@/viewer/components/ViewControls";
import { buildMetrics } from "@/viewer/components/internal/viewerMath";

describe("ViewControls", () => {
  it("offers the three ways back, by name", () => {
    render(<ViewControls onFit={vi.fn()} onOneToOne={vi.fn()} onReset={vi.fn()} />);

    expect(screen.getByRole("button", { name: "Fit" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "1:1" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Reset" })).toBeTruthy();
  });

  it("calls the handler the user pressed", async () => {
    const onFit = vi.fn();
    const onOneToOne = vi.fn();
    const onReset = vi.fn();
    render(<ViewControls onFit={onFit} onOneToOne={onOneToOne} onReset={onReset} />);

    await userEvent.click(screen.getByRole("button", { name: "Fit" }));
    await userEvent.click(screen.getByRole("button", { name: "1:1" }));
    await userEvent.click(screen.getByRole("button", { name: "Reset" }));

    expect(onFit).toHaveBeenCalledTimes(1);
    expect(onOneToOne).toHaveBeenCalledTimes(1);
    expect(onReset).toHaveBeenCalledTimes(1);
  });
});

describe("ScaleBar", () => {
  const metrics = buildMetrics(
    { centerX: 0.5, centerY: 0.25, zoom: 1, containerWidth: 1000, containerHeight: 800 },
    4000,
    2000
  );

  it("states the physical length in the accessible name, not just the picture", () => {
    render(<ScaleBar metrics={metrics} pixelSizeNm={5} />);

    expect(screen.getByLabelText("Scale bar: 2 µm")).toBeTruthy();
    expect(screen.getByText("2 µm")).toBeTruthy();
  });

  it("draws nothing when the image has no pixel size", () => {
    // An uncalibrated image gets no bar rather than a bar that would silently
    // mean pixels.
    const { container } = render(<ScaleBar metrics={metrics} pixelSizeNm={null} />);

    expect(container.querySelector(".viewer-scale-bar")).toBeNull();
  });

  it("draws nothing before the canvas has been measured", () => {
    const { container } = render(<ScaleBar metrics={null} pixelSizeNm={5} />);

    expect(container.querySelector(".viewer-scale-bar")).toBeNull();
  });
});
