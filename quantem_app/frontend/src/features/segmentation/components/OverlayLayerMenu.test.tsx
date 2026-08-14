import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import {
  OverlayLayerMenu,
  type OverlayLayerControl,
} from "@/features/segmentation/components/OverlayLayerMenu";

function controls(overrides: Partial<OverlayLayerControl> = {}): OverlayLayerControl {
  return {
    strokeWidth: 2,
    fillOpacity: 0.2,
    showBorders: true,
    onStrokeWidthChange: vi.fn(),
    onFillOpacityChange: vi.fn(),
    onShowBordersChange: vi.fn(),
    ...overrides,
  };
}

describe("OverlayLayerMenu", () => {
  it("starts collapsed and shows both left-pane overlay groups when opened", async () => {
    const user = userEvent.setup();
    const candidateControls = controls();
    const confirmedControls = controls();
    render(
      <OverlayLayerMenu
        idPrefix="left"
        paneLabel="Left pane"
        usesRasterOverlay
        candidates={candidateControls}
        confirmed={confirmedControls}
      />
    );

    const summary = screen.getByLabelText("Left pane overlay options");
    expect(summary.parentElement).not.toHaveAttribute("open");
    await user.click(summary);
    expect(summary.parentElement).toHaveAttribute("open");
    expect(screen.getByRole("heading", { name: "Candidates" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Confirmed" })).toBeInTheDocument();

    await user.click(screen.getByRole("checkbox", { name: "Candidates borders" }));
    expect(candidateControls.onShowBordersChange).toHaveBeenCalledWith(false);
  });

  it("can show only confirmed options for the right pane", async () => {
    const user = userEvent.setup();
    render(
      <OverlayLayerMenu
        idPrefix="right"
        paneLabel="Right pane"
        usesRasterOverlay={false}
        confirmed={controls()}
      />
    );

    await user.click(screen.getByLabelText("Right pane overlay options"));
    expect(screen.queryByRole("heading", { name: "Candidates" })).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Confirmed" })).toBeInTheDocument();
    expect(
      screen.getByRole("slider", { name: "Confirmed border thickness" })
    ).toBeInTheDocument();
  });
});
