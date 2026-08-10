import type { ComponentProps } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { WorkflowModeToolbar } from "@/features/segmentation/components/WorkflowModeToolbar";

function makeProps(
  overrides: Partial<ComponentProps<typeof WorkflowModeToolbar>> = {}
): ComponentProps<typeof WorkflowModeToolbar> {
  return {
    workflowMode: "review",
    reviewPhase: "model",
    correctionTool: "draw",
    hoverActionMode: "confirm",
    drawBrushSize: 24,
    hasDrawStrokes: false,
    onReviewPhaseChange: vi.fn(),
    onCorrectionToolChange: vi.fn(),
    onHoverActionModeChange: vi.fn(),
    onDrawBrushSizeChange: vi.fn(),
    onConfirmShape: vi.fn(),
    onClearDrawing: vi.fn(),
    ...overrides,
  };
}

describe("WorkflowModeToolbar", () => {
  it("shows model-phase hover actions and hides correction controls", () => {
    render(<WorkflowModeToolbar {...makeProps()} />);
    expect(screen.getByRole("button", { name: "Confirm Object" })).toHaveAttribute(
      "aria-pressed",
      "true"
    );
    expect(screen.getByRole("button", { name: "Reject Object" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Confirm Group" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reject Group" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Draw" })).not.toBeInTheDocument();
  });

  it("shows test point control when enabled", async () => {
    const user = userEvent.setup();
    const onHoverActionModeChange = vi.fn();
    render(
      <WorkflowModeToolbar
        {...makeProps({
          showTestPoint: true,
          onHoverActionModeChange,
        })}
      />
    );

    await user.click(screen.getByRole("button", { name: "Test Point" }));
    expect(onHoverActionModeChange).toHaveBeenCalledWith("test");
  });

  it("switches back to point mode from a group mode", async () => {
    const user = userEvent.setup();
    const onHoverActionModeChange = vi.fn();
    render(
      <WorkflowModeToolbar
        {...makeProps({
          hoverActionMode: "group-confirm",
          onHoverActionModeChange,
        })}
      />
    );

    await user.click(screen.getByRole("button", { name: "Confirm Object" }));
    expect(onHoverActionModeChange).toHaveBeenCalledWith("confirm");
  });

  it("applies the active group action when clicked again with a selection", async () => {
    const user = userEvent.setup();
    const onApplyGroupAction = vi.fn();
    const onHoverActionModeChange = vi.fn();
    render(
      <WorkflowModeToolbar
        {...makeProps({
          hoverActionMode: "group-reject",
          canApplyGroupAction: true,
          onApplyGroupAction,
          onHoverActionModeChange,
        })}
      />
    );

    await user.click(screen.getByRole("button", { name: "Reject Group" }));
    expect(onApplyGroupAction).toHaveBeenCalledWith("group-reject");
    expect(onHoverActionModeChange).not.toHaveBeenCalled();
  });

  it("shows draw confirm/clear actions once strokes exist", () => {
    render(
      <WorkflowModeToolbar
        {...makeProps({
          reviewPhase: "correction",
          correctionTool: "draw",
          hasDrawStrokes: true,
        })}
      />
    );

    expect(screen.getByRole("button", { name: "Confirm Drawn Area" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Clear" })).toBeEnabled();
  });

  it("shows the confirmed-area tool and explains what it is for", () => {
    render(
      <WorkflowModeToolbar
        {...makeProps({
          reviewPhase: "correction",
        })}
      />
    );

    const button = screen.getByRole("button", { name: "Confirmed area" });
    expect(button).toBeInTheDocument();

    // The concept had three names and its only written rationale lived in a 400
    // response body. The "why" now sits next to the control, and the API's own
    // term is given so `blockers` and the Adapt wizard are recognisable.
    const explainer = document.getElementById(button.getAttribute("aria-describedby") ?? "");
    expect(explainer?.textContent).toContain("counts as true background");
    expect(explainer?.textContent).toContain("pixels are ignored");
    expect(explainer?.textContent).toContain("completed ROI");
  });

  it("uses the same explanation for the ER toolbar variant", () => {
    render(
      <WorkflowModeToolbar
        {...makeProps({
          reviewPhase: "correction",
          isErSegmentation: true,
        })}
      />
    );

    const button = screen.getByRole("button", { name: "Confirmed area" });
    const explainer = document.getElementById(button.getAttribute("aria-describedby") ?? "");
    expect(explainer?.textContent).toContain("counts as true background");
  });

  it("offers polygon, draw and erase tools for ER", () => {
    render(
      <WorkflowModeToolbar
        {...makeProps({
          reviewPhase: "correction",
          isErSegmentation: true,
        })}
      />
    );

    expect(screen.getByRole("button", { name: "Polygon" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Draw" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Erase" })).toBeInTheDocument();
  });

  it("does not offer a SAM prompt tool", () => {
    render(
      <WorkflowModeToolbar
        {...makeProps({
          reviewPhase: "correction",
        })}
      />
    );

    expect(screen.queryByRole("button", { name: "Prompt" })).not.toBeInTheDocument();
    expect(screen.queryByRole("checkbox", { name: "Auto" })).not.toBeInTheDocument();
  });

  it("fires phase/tool callbacks", async () => {
    const user = userEvent.setup();
    const onReviewPhaseChange = vi.fn();
    const onCorrectionToolChange = vi.fn();
    render(
      <WorkflowModeToolbar
        {...makeProps({
          reviewPhase: "correction",
          onReviewPhaseChange,
          onCorrectionToolChange,
        })}
      />
    );

    await user.click(screen.getByRole("button", { name: "Review" }));
    await user.click(screen.getByRole("button", { name: "Draw" }));
    await user.click(screen.getByRole("button", { name: "Confirmed area" }));
    expect(onReviewPhaseChange).toHaveBeenCalledWith("model");
    expect(onCorrectionToolChange).toHaveBeenCalledWith("draw");
    expect(onCorrectionToolChange).toHaveBeenCalledWith("completed_roi");
  });
});
