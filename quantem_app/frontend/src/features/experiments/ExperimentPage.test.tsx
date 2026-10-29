import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ExperimentPage } from "@/features/experiments/ExperimentPage";
import {
  getExperiment,
  getExperiments,
  getHomeEntryPage,
  getAssetSegmentations,
  updateAssetLibraryDetails,
  updateDataset,
  updateExperiment,
} from "@/shared/api/assets";
import type { Dataset, Experiment } from "@/shared/types/common";
import type { HomeEntry, HomeEntryPage } from "@/shared/types/images";
import { getWorkspace } from "@/shared/api/workspace";
import { getSystemStatus } from "@/shared/api/jobs";
vi.mock("@/shared/api/workspace", () => ({ getWorkspace: vi.fn(), saveSample: vi.fn() }));

vi.mock("@/shared/api/jobs", async () => {
  const actual = await vi.importActual<typeof import("@/shared/api/jobs")>(
    "@/shared/api/jobs"
  );
  return { ...actual, getSystemStatus: vi.fn() };
});

vi.mock("@/shared/api/assets", async () => {
  const actual = await vi.importActual<typeof import("@/shared/api/assets")>(
    "@/shared/api/assets"
  );
  return {
    ...actual,
    deleteAsset: vi.fn(),
    getExperiment: vi.fn(),
    getExperiments: vi.fn(),
    getHomeEntryPage: vi.fn(),
    getAssetSegmentations: vi.fn(),
    updateAssetLibraryDetails: vi.fn(),
    updateDataset: vi.fn(),
    updateExperiment: vi.fn(),
  };
});

const DATASET_A: Dataset = {
  id: "dataset-a",
  experiment: "experiment-1",
  name: "Baseline",
  notes: "Untreated samples",
  asset_count: 1,
  created_at: null,
  updated_at: null,
};

const DATASET_B: Dataset = {
  ...DATASET_A,
  id: "dataset-b",
  name: "Stimulated",
  notes: "",
  asset_count: 0,
};

const EXPERIMENT: Experiment = {
  id: "experiment-1",
  name: "Liver study",
  notes: "Glucose-infusion experiment",
  datasets: [DATASET_A, DATASET_B],
  asset_count: 1,
  ungrouped_asset_count: 0,
  created_at: null,
  updated_at: null,
};

const IMAGE: HomeEntry = {
  id: "asset-1",
  display_name: "Liver 01",
  original_filename: "liver-01.tif",
  notes: "Periportal field with visible glycogen.",
  metadata_summary: "1024x1024",
  width: 1024,
  height: 1024,
  pixel_size_nm: 5,
  experiment_id: EXPERIMENT.id,
  experiment_name: EXPERIMENT.name,
  dataset_ids: [DATASET_A.id],
  dataset_names: [DATASET_A.name],
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  preprocess_stage: "DONE",
  preprocess_progress: 100,
  ngff_ready: true,
  can_open: true,
};

function page(results: HomeEntry[]): HomeEntryPage {
  return {
    results,
    total: results.length,
    limit: 5,
    offset: 0,
    has_more: false,
  };
}

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/experiments/experiment-1"]}>
      <Routes>
        <Route path="/experiments/:experimentId" element={<ExperimentPage />} />
        <Route path="/assets/:assetId/viewer" element={<p>Viewer</p>} />
      </Routes>
    </MemoryRouter>
  );
}

describe("ExperimentPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    sessionStorage.clear();
    sessionStorage.setItem("quantem.workspace.experiment-1.controls", JSON.stringify({ layout: "cards" }));
    vi.spyOn(window, "scrollTo").mockImplementation(() => {});
    vi.mocked(getWorkspace).mockResolvedValue({ images: [] });
    vi.mocked(getSystemStatus).mockResolvedValue({
      cuda_available: false,
      supported_upload_formats: [".tif", ".tiff", ".png"],
    });
    vi.mocked(getExperiment).mockResolvedValue(EXPERIMENT);
    vi.mocked(getExperiments).mockResolvedValue([EXPERIMENT]);
    vi.mocked(getHomeEntryPage).mockImplementation(async (params) =>
      params?.dataset === DATASET_A.id ? page([IMAGE]) : page([])
    );
    vi.mocked(getAssetSegmentations).mockResolvedValue([]);
    vi.mocked(updateExperiment).mockImplementation(async (_id, updates) => ({
      ...EXPERIMENT,
      ...updates,
    }));
    vi.mocked(updateDataset).mockImplementation(async (_id, updates) => ({
      ...DATASET_A,
      ...updates,
    }));
    vi.mocked(updateAssetLibraryDetails).mockResolvedValue({ id: IMAGE.id } as never);
  });

  it("shows experiment and dataset notes with image cards", async () => {
    renderPage();

    expect(await screen.findByRole("heading", { name: "Liver study" })).toBeInTheDocument();
    expect(screen.getByText("Glucose-infusion experiment")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Baseline" })).toBeInTheDocument();
    expect(screen.getByText("Untreated samples")).toBeInTheDocument();
    expect(await screen.findByText("Periportal field with visible glycogen.")).toHaveAttribute(
      "title",
      "Periportal field with visible glycogen."
    );
    expect(
      screen.getByRole("button", { name: "Options for Liver 01" })
    ).toBeInTheDocument();
  });

  it("opens a prominent right-aligned Add images dialog from the selection row", async () => {
    const user = userEvent.setup();
    renderPage();

    const addImages = await screen.findByRole("button", { name: "Add images" });
    expect(addImages).toHaveClass("ml-auto");
    await user.click(addImages);

    const dialog = await screen.findByRole("dialog", { name: "Add images" });
    expect(dialog).toHaveTextContent("Add images to Liver study");
    await user.upload(
      within(dialog).getByLabelText("Image file"),
      new File(["fake"], "new-image.png", { type: "image/png" })
    );
    expect(await within(dialog).findByLabelText("New dataset name")).toHaveValue("Dataset 1");
    expect(within(dialog).getByRole("radio", { name: "Existing datasets" })).toBeEnabled();
  });

  it("edits image metadata and dataset membership atomically", async () => {
    const user = userEvent.setup();
    renderPage();
    await user.click(
      await screen.findByRole("button", { name: "Options for Liver 01" })
    );
    await user.click(screen.getByRole("menuitem", { name: "Edit" }));

    const dialog = await screen.findByRole("dialog", { name: "Edit image" });
    await user.clear(within(dialog).getByLabelText("Display name"));
    await user.type(within(dialog).getByLabelText("Display name"), "Liver ROI A");
    await user.clear(within(dialog).getByLabelText("Resolution (nm/px)"));
    await user.type(within(dialog).getByLabelText("Resolution (nm/px)"), "6.5");
    await user.selectOptions(within(dialog).getByLabelText("Dataset"), DATASET_B.id);
    await user.clear(within(dialog).getByLabelText("Image notes"));
    await user.type(within(dialog).getByLabelText("Image notes"), "Updated note");
    await user.click(within(dialog).getByRole("button", { name: "Save changes" }));

    await waitFor(() =>
      expect(updateAssetLibraryDetails).toHaveBeenCalledWith(IMAGE.id, {
        display_name: "Liver ROI A",
        pixel_size_nm: 6.5,
        notes: "Updated note",
        datasets: [DATASET_B.id],
      })
    );
  });

  it("opens Export from an image's vertical-dots menu", async () => {
    const user = userEvent.setup();
    renderPage();
    await user.click(
      await screen.findByRole("button", { name: "Options for Liver 01" })
    );
    await user.click(screen.getByRole("menuitem", { name: "Export" }));

    expect(
      await screen.findByRole("dialog", { name: "Export image or mask · Liver 01" })
    ).toBeInTheDocument();
    expect(getAssetSegmentations).toHaveBeenCalledWith(IMAGE.id);
  });

  it("edits experiment and dataset names and notes inline", async () => {
    const user = userEvent.setup();
    renderPage();
    await screen.findByRole("heading", { name: "Liver study" });

    await user.click(screen.getByRole("button", { name: "Edit experiment" }));
    await user.clear(screen.getByLabelText("Experiment name"));
    await user.type(screen.getByLabelText("Experiment name"), "Liver follow-up");
    await user.clear(screen.getByLabelText("Notes"));
    await user.type(screen.getByLabelText("Notes"), "Second cohort");
    await user.click(screen.getByRole("button", { name: "Confirm" }));
    await waitFor(() =>
      expect(updateExperiment).toHaveBeenCalledWith(EXPERIMENT.id, {
        name: "Liver follow-up",
        notes: "Second cohort",
      })
    );

    await user.click(screen.getAllByRole("button", { name: "Edit dataset" })[0]);
    await user.clear(screen.getByLabelText("Dataset name"));
    await user.type(screen.getByLabelText("Dataset name"), "Control");
    await user.click(screen.getByRole("button", { name: "Confirm" }));
    await waitFor(() =>
      expect(updateDataset).toHaveBeenCalledWith(
        DATASET_A.id,
        expect.objectContaining({ name: "Control" })
      )
    );
  });
});
