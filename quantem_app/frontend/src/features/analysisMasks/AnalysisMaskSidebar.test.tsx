import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { ImageSegmentation } from "@/shared/types/images";
import { AnalysisMaskSidebar } from "./AnalysisMaskSidebar";
import type { AnalysisMaskObject } from "./types";

const object: AnalysisMaskObject = {
  id: "object-1",
  segmentation: "seg-1",
  name: "Object 1",
  color: "#38bdf8",
  sort_order: 1,
  geometry: {
    type: "Polygon",
    coordinates: [
      [
        [0, 0],
        [20, 0],
        [20, 20],
        [0, 20],
        [0, 0],
      ],
    ],
  },
  created_at: "2026-08-13T00:00:00Z",
  updated_at: "2026-08-13T00:00:00Z",
};

function renderSidebar(
  overrides: Partial<Parameters<typeof AnalysisMaskSidebar>[0]> = {}
) {
  const props: Parameters<typeof AnalysisMaskSidebar>[0] = {
    tool: "polygon",
    onToolChange: vi.fn(),
    operation: "include",
    onOperationChange: vi.fn(),
    canExclude: false,
    brushSize: 24,
    onBrushSizeChange: vi.fn(),
    polygonHasDraft: false,
    polygonCanClose: false,
    polygonSaving: false,
    onClosePolygon: vi.fn(),
    onClearPolygon: vi.fn(),
    navigateMode: false,
    onNavigateModeChange: vi.fn(),
    objects: [object],
    activeObjectId: object.id,
    busy: false,
    onEditObject: vi.fn(),
    onSaveObject: vi.fn(),
    onRenameObject: vi.fn().mockResolvedValue(undefined),
    onRequestDeleteObject: vi.fn(),
    existingMaskLayers: [],
    onToggleExistingMask: vi.fn(),
    ...overrides,
  };
  return { ...render(<AnalysisMaskSidebar {...props} />), props };
}

describe("AnalysisMaskSidebar", () => {
  it("starts from a polygon-focused object editor without review or layer controls", () => {
    renderSidebar();

    expect(screen.getByRole("button", { name: "Polygon" })).toHaveAttribute(
      "aria-pressed",
      "true"
    );
    expect(screen.getByRole("button", { name: "Save" })).toBeInTheDocument();
    expect(screen.queryByText("View confirmed")).not.toBeInTheDocument();
    expect(screen.queryByText("Layers")).not.toBeInTheDocument();
    expect(screen.queryByText("Model to run")).not.toBeInTheDocument();
  });

  it("keeps rename, delete, and Edit/Save controls on the second row", async () => {
    const user = userEvent.setup();
    const onRenameObject = vi.fn().mockResolvedValue(undefined);
    const onRequestDeleteObject = vi.fn();
    const onSaveObject = vi.fn();
    renderSidebar({ onRenameObject, onRequestDeleteObject, onSaveObject });

    await user.click(screen.getByRole("button", { name: "Rename Object 1" }));
    const input = screen.getByRole("textbox", { name: "Name for Object 1" });
    await user.clear(input);
    await user.type(input, "Portal cell{Enter}");
    expect(onRenameObject).toHaveBeenCalledWith("object-1", "Portal cell");

    await user.click(screen.getByRole("button", { name: "Delete Object 1" }));
    expect(onRequestDeleteObject).toHaveBeenCalledWith(object);

    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(onSaveObject).toHaveBeenCalledTimes(1);
  });

  it("offers only other Analysis Masks as secondary layer toggles", async () => {
    const user = userEvent.setup();
    const otherMask = {
      id: "seg-2",
      asset: "image-1",
      display_name: "Cell boundary",
      segmentation_type: {
        id: "type-analysis",
        internal_name: "quantem_internal_analysis_mask",
        short_name: "Analysis mask",
        long_name: "Analysis Segmentation Mask",
        tags: [],
        created_at: "2026-08-13T00:00:00Z",
        updated_at: "2026-08-13T00:00:00Z",
      },
      status_stage: "CANDIDATES_READY",
      status_progress: 100,
      created_at: "2026-08-13T00:00:00Z",
      updated_at: "2026-08-13T00:00:00Z",
    } satisfies ImageSegmentation;
    const onToggleExistingMask = vi.fn();
    renderSidebar({
      existingMaskLayers: [{ segmentation: otherMask, enabled: true }],
      onToggleExistingMask,
    });

    await user.click(screen.getByRole("checkbox", { name: "Cell boundary" }));
    expect(onToggleExistingMask).toHaveBeenCalledWith("seg-2");
  });
});
