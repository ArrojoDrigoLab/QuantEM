import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  getAssetRasterExportUrl,
  getAssetSegmentations,
} from "@/shared/api/assets";
import type { ImageSegmentation } from "@/shared/types/images";
import { ExportDialog } from "./ExportDialog";

vi.mock("@/shared/api/assets", async () => {
  const actual = await vi.importActual<typeof import("@/shared/api/assets")>(
    "@/shared/api/assets"
  );
  return {
    ...actual,
    getAssetRasterExportUrl: vi.fn(),
    getAssetSegmentations: vi.fn(),
  };
});

function segmentation(
  id: string,
  name: string,
  measurementMode: "objects" | "global",
  internalName = `custom_${id}`
): ImageSegmentation {
  return {
    id,
    display_name: name,
    segmentation_type: {
      id: `type-${id}`,
      internal_name: internalName,
      short_name: name,
      long_name: name,
      measurement_mode: measurementMode,
      tags: [],
      created_at: "2026-08-13T00:00:00Z",
      updated_at: "2026-08-13T00:00:00Z",
    },
    status_stage: "CANDIDATES_READY",
    status_progress: 100,
    created_at: "2026-08-13T00:00:00Z",
    updated_at: "2026-08-13T00:00:00Z",
  };
}

describe("ExportDialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getAssetSegmentations).mockResolvedValue([
      segmentation("mito", "Mitochondria", "objects"),
      segmentation(
        "mask",
        "Cell compartments",
        "global",
        "quantem_internal_analysis_mask"
      ),
    ]);
    vi.mocked(getAssetRasterExportUrl).mockReturnValue("/download/export.png");
  });

  it("offers the original image, every segmentation, and the overlap note", async () => {
    render(
      <ExportDialog
        asset={{ id: "asset-1", displayName: "Portal field" }}
        onClose={vi.fn()}
      />
    );

    expect(screen.getByRole("radio", { name: /Original EM image/i })).toBeChecked();
    expect(await screen.findByRole("radio", { name: /Mitochondria/i })).toBeInTheDocument();
    expect(
      screen.getByRole("radio", { name: /Cell compartments/i })
    ).toBeInTheDocument();
    expect(screen.getByText(/later in the list replaces the earlier object/i)).toBeInTheDocument();
  });

  it("downloads the selected segmentation through the shared endpoint", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    render(
      <ExportDialog
        asset={{ id: "asset-1", displayName: "Portal field" }}
        onClose={onClose}
      />
    );

    await user.click(await screen.findByRole("radio", { name: /Cell compartments/i }));
    await user.click(screen.getByRole("button", { name: "Export" }));

    expect(getAssetRasterExportUrl).toHaveBeenCalledWith("asset-1", {
      source: "segmentation",
      segmentationId: "mask",
    });
    expect(click).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(onClose).toHaveBeenCalled());
    click.mockRestore();
  });
});
