/**
 * The slot contract.
 *
 * Two other packages render controls into this toolbar without opening it, so
 * the slots are the interface between them and the file's owner. If a slot
 * silently stops rendering, the control it carries disappears from the app with
 * nothing failing anywhere near the package that owns it.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { WorkflowModeToolbar } from "@/features/segmentation/components/WorkflowModeToolbar";

function renderToolbar(extra: Partial<Parameters<typeof WorkflowModeToolbar>[0]> = {}) {
  return render(
    <WorkflowModeToolbar
      workflowMode="review"
      reviewPhase="model"
      correctionTool="draw"
      hoverActionMode="confirm"
      drawBrushSize={24}
      hasDrawStrokes={false}
      onReviewPhaseChange={vi.fn()}
      onCorrectionToolChange={vi.fn()}
      onHoverActionModeChange={vi.fn()}
      onDrawBrushSizeChange={vi.fn()}
      onConfirmShape={vi.fn()}
      onClearDrawing={vi.fn()}
      {...extra}
    />
  );
}

describe("WorkflowModeToolbar slots", () => {
  it("renders an extra mode as a Correct sub-tool, not beside Review and Correct", () => {
    renderToolbar({
      reviewPhase: "correction",
      extraModes: <button type="button">Focus queue</button>,
    });

    const topLevelGroup = screen.getByRole("button", { name: "Review" }).parentElement;
    expect(topLevelGroup?.textContent).not.toContain("Focus queue");
    expect(screen.getByRole("button", { name: "Focus queue" })).toBeInTheDocument();
  });

  it("renders extra tools at the end of the toolbar", () => {
    renderToolbar({ extraTools: <button type="button">Mark checked area</button> });

    expect(screen.getByRole("button", { name: "Mark checked area" })).toBeTruthy();
  });

  it("adds nothing to the toolbar when no package fills a slot", () => {
    const { container } = renderToolbar();

    expect(container.textContent).not.toContain("undefined");
    expect(screen.getByRole("button", { name: "Review" })).toBeTruthy();
  });
});
