import { render, screen, waitFor } from "@testing-library/react";
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
  tags: [],
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
    objectNamesRequired: false,
    cellsMode: false,
    activeObjectId: object.id,
    metadataEditingObjectId: null,
    onMetadataEditingObjectChange: vi.fn(),
    metadataEditDisabled: false,
    tagSuggestions: [],
    busy: false,
    onEditObject: vi.fn(),
    onSaveObject: vi.fn(),
    onUpdateObjectMetadata: vi.fn().mockResolvedValue(undefined),
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

  it("keeps metadata, delete, and Edit/Save controls on the second row", async () => {
    const user = userEvent.setup();
    const onMetadataEditingObjectChange = vi.fn();
    const onRequestDeleteObject = vi.fn();
    const onSaveObject = vi.fn();
    renderSidebar({
      onMetadataEditingObjectChange,
      onRequestDeleteObject,
      onSaveObject,
    });

    await user.click(screen.getByRole("button", { name: "Rename Object 1" }));
    expect(onMetadataEditingObjectChange).toHaveBeenCalledWith("object-1");

    await user.click(screen.getByRole("button", { name: "Delete Object 1" }));
    expect(onRequestDeleteObject).toHaveBeenCalledWith(object);

    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(onSaveObject).toHaveBeenCalledTimes(1);
  });

  it("saves a Cell name and multiple key/value tags from the check button", async () => {
    const user = userEvent.setup();
    const onUpdateObjectMetadata = vi.fn().mockResolvedValue(undefined);
    const onMetadataEditingObjectChange = vi.fn();
    const taggedObject: AnalysisMaskObject = {
      ...object,
      name: "Portal cell",
      tags: [{ name: "Phenotype", value: "Beta" }],
    };
    const { container } = renderSidebar({
      cellsMode: true,
      objectNamesRequired: true,
      objects: [taggedObject],
      activeObjectId: null,
      metadataEditingObjectId: taggedObject.id,
      onMetadataEditingObjectChange,
      onUpdateObjectMetadata,
      tagSuggestions: [
        { name: "Phenotype", values: ["Beta", "Alpha"] },
        { name: "Treatment", values: ["Control"] },
      ],
    });

    expect(
      screen.getByRole("heading", { name: "Edit cell details" })
    ).toBeInTheDocument();
    expect(container.querySelector('option[value="Phenotype"]')).toBeInTheDocument();
    expect(container.querySelector('option[value="Beta"]')).toBeInTheDocument();

    const name = screen.getByRole("textbox", { name: "Name for Portal cell" });
    await user.clear(name);
    await user.type(name, "Pericentral cell");
    await user.type(
      screen.getByRole("combobox", { name: "Tag name 2 for Portal cell" }),
      "Treatment"
    );
    await user.type(
      screen.getByRole("combobox", { name: "Tag value 2 for Portal cell" }),
      "Control"
    );
    await user.click(screen.getByRole("button", { name: "Add tag" }));
    expect(
      screen.getByRole("combobox", { name: "Tag name 3 for Portal cell" })
    ).toBeInTheDocument();

    await user.click(
      screen.getByRole("button", { name: "Save name and tags for Portal cell" })
    );

    await waitFor(() => {
      expect(onUpdateObjectMetadata).toHaveBeenCalledWith(
        taggedObject.id,
        "Pericentral cell",
        [
          { name: "Phenotype", value: "Beta" },
          { name: "Treatment", value: "Control" },
        ]
      );
      expect(onMetadataEditingObjectChange).toHaveBeenCalledWith(null);
    });
  });

  it("rejects duplicate Cell tag names without regard to casing", async () => {
    const user = userEvent.setup();
    const onUpdateObjectMetadata = vi.fn().mockResolvedValue(undefined);
    renderSidebar({
      cellsMode: true,
      objectNamesRequired: true,
      objects: [{ ...object, name: "Cell 1", tags: [] }],
      activeObjectId: null,
      metadataEditingObjectId: object.id,
      onUpdateObjectMetadata,
    });

    await user.type(
      screen.getByRole("combobox", { name: "Tag name 1 for Cell 1" }),
      "State"
    );
    await user.click(screen.getByRole("button", { name: "Add tag" }));
    await user.type(
      screen.getByRole("combobox", { name: "Tag name 2 for Cell 1" }),
      "state"
    );
    await user.click(
      screen.getByRole("button", { name: "Save name and tags for Cell 1" })
    );

    expect(screen.getByRole("alert")).toHaveTextContent(
      "A cell cannot contain the same tag name twice."
    );
    expect(onUpdateObjectMetadata).not.toHaveBeenCalled();
  });

  it("shows saved tags as pills and disables labeling while metadata is open", () => {
    const taggedObject: AnalysisMaskObject = {
      ...object,
      name: "Cell 1",
      tags: [{ name: "Phenotype", value: "Beta" }],
    };
    const { rerender, props } = renderSidebar({
      cellsMode: true,
      objects: [taggedObject],
      activeObjectId: null,
    });
    expect(screen.getByLabelText("Tags for Cell 1")).toHaveTextContent(
      "Phenotype: Beta"
    );

    rerender(
      <AnalysisMaskSidebar
        {...props}
        activeObjectId={taggedObject.id}
        metadataEditingObjectId={taggedObject.id}
      />
    );
    expect(screen.getByRole("button", { name: "Include" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Exclude" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Polygon" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Brush" })).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: "Navigate (A)" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Done" })).toBeDisabled();
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

  it("uses polygon-only automatic creation for Cells", () => {
    renderSidebar({ cellsMode: true, activeObjectId: null });

    expect(screen.getByRole("heading", { name: "Draw cells" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Brush" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Include" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Exclude" })).not.toBeInTheDocument();
    expect(screen.getByText(/saves automatically/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Save" })).not.toBeInTheDocument();
  });

  it("restores include, exclude, polygon, and brush tools while editing a Cell", () => {
    renderSidebar({ cellsMode: true, activeObjectId: object.id, canExclude: true });

    expect(screen.getByRole("heading", { name: "Edit cell" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Include" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Exclude" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Polygon" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Brush" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Done" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Save" })).not.toBeInTheDocument();
  });
});
