import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { LibraryPage } from "@/features/library/LibraryPage";
import {
  getAsset,
  getHomeEntryPage,
  recoverDeferredUploadedAssetPipelines,
  uploadAsset,
} from "@/shared/api/assets";
import { getSystemStatus } from "@/shared/api/jobs";
import type { AssetDetail, HomeEntry, HomeEntryPage } from "@/shared/types/images";
import { server } from "@/test/msw/server";

vi.mock("@/shared/api/assets", async () => {
  const actual = await vi.importActual<typeof import("@/shared/api/assets")>(
    "@/shared/api/assets"
  );
  return {
    ...actual,
    getAsset: vi.fn(),
    getHomeEntryPage: vi.fn(),
    recoverDeferredUploadedAssetPipelines: vi.fn(),
    uploadAsset: vi.fn(),
  };
});

vi.mock("@/shared/api/jobs", async () => {
  const actual = await vi.importActual<typeof import("@/shared/api/jobs")>(
    "@/shared/api/jobs"
  );
  return { ...actual, getSystemStatus: vi.fn() };
});

function entry(): HomeEntry {
  return {
    id: "asset-1",
    display_name: "Liver 01",
    original_filename: "liver.tif",
    notes: "Glucose infusion sample",
    metadata_summary: "1024x1024",
    width: 1024,
    height: 1024,
    pixel_size_nm: 5,
    experiment_id: "11111111-1111-1111-1111-111111111111",
    experiment_name: "Liver study",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    preprocess_stage: "DONE",
    preprocess_progress: 100,
    ngff_ready: true,
    can_open: true,
  };
}

function page(): HomeEntryPage {
  return { results: [entry()], total: 1, limit: 5, offset: 0, has_more: false };
}

function uploadedAsset(): AssetDetail {
  return {
    id: "asset-new",
    file_path: "",
    original_filename: "grid2.tif",
    display_name: "grid2",
    is_eval_set: false,
    width: 2048,
    height: 2048,
    channels: 1,
    bit_depth: 8,
    pixel_size_nm: 5,
    notes: "Fresh import",
    experiment_id: "11111111-1111-1111-1111-111111111111",
    experiment_name: "Liver study",
    dataset_ids: [],
    dataset_names: [],
    preprocess_stage: "ENCODING",
    preprocess_progress: 5,
    ngff_ready: false,
    is_workable: true,
    tags: [],
    created_at: "2026-02-02T00:00:00Z",
    updated_at: "2026-02-02T00:00:00Z",
  };
}

async function importOneFile() {
  const zone = await screen.findByTestId("import-drop-zone");
  fireEvent.drop(zone, {
    dataTransfer: {
      files: [new File([new Uint8Array(16)], "grid2.tif", { type: "image/tiff" })],
      types: ["Files"],
    },
  });
  await screen.findByTestId("import-chosen-file");
  await act(async () => {
    fireEvent.submit(screen.getByTestId("import-form"));
  });
  return screen.findByRole("link", { name: "grid2" });
}

describe("LibraryPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    window.localStorage.setItem("quantem-workflow-guide-dismissed-v1", "1");
    vi.mocked(getHomeEntryPage).mockResolvedValue(page());
    vi.mocked(getSystemStatus).mockResolvedValue({
      app_version: "0.1.2",
      cuda_available: false,
      supported_upload_formats: [".tif", ".tiff", ".png"],
    });
    vi.mocked(uploadAsset).mockResolvedValue(uploadedAsset());
    vi.mocked(getAsset).mockResolvedValue(uploadedAsset());
    vi.mocked(recoverDeferredUploadedAssetPipelines).mockResolvedValue({
      job_ids: [],
    });
    server.use(
      http.get("http://127.0.0.1:8000/api/experiments/", () =>
        HttpResponse.json([
          {
            id: "11111111-1111-1111-1111-111111111111",
            name: "Liver study",
            notes: "",
            asset_count: 1,
            ungrouped_asset_count: 1,
            datasets: [],
            created_at: null,
            updated_at: null,
          },
        ])
      )
    );
  });

  function renderPage() {
    return render(
      <MemoryRouter>
        <LibraryPage />
      </MemoryRouter>
    );
  }

  it("uses QuantEM as the only home title and moves version/compute/models into settings", async () => {
    const user = userEvent.setup();
    renderPage();

    expect(await screen.findByRole("heading", { name: "QuantEM" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Library" })).not.toBeInTheDocument();
    expect(screen.queryByText("Running on CPU")).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Models" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Settings" }));
    expect(screen.getByRole("dialog", { name: "Settings" })).toBeInTheDocument();
    expect(screen.getByText("v0.1.2")).toBeInTheDocument();
    expect(screen.getByText("CPU")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Models" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Models" })).not.toBeInTheDocument();
  });

  it("remembers dismissal from the header guide button", async () => {
    const user = userEvent.setup();
    window.localStorage.clear();
    renderPage();
    await user.click(await screen.findByRole("button", { name: "Hide guide" }));
    expect(window.localStorage.getItem("quantem-workflow-guide-dismissed-v1")).toBe("1");
  });

  it("can reopen the guide without forgetting that it was dismissed", async () => {
    const user = userEvent.setup();
    renderPage();

    await user.click(await screen.findByRole("button", { name: "How this works" }));
    expect(screen.getByRole("button", { name: "Hide guide" })).toBeInTheDocument();
    expect(window.localStorage.getItem("quantem-workflow-guide-dismissed-v1")).toBe("1");
  });

  it("keeps the file picker directly available under Import an image", async () => {
    renderPage();

    const input = await screen.findByLabelText(/image file/i);
    expect(input).toHaveAttribute("type", "file");
    expect(input).toHaveAttribute("multiple");
  });

  it("still accepts a file dropped anywhere on the page", async () => {
    renderPage();
    await screen.findByTestId("import-drop-zone");
    const pageRoot = document.querySelector(".min-h-screen") as HTMLElement;

    fireEvent.drop(pageRoot, {
      dataTransfer: {
        files: [new File(["x"], "dropped-on-page.tif", { type: "image/tiff" })],
        types: ["Files"],
      },
    });

    expect(await screen.findByText("dropped-on-page.tif")).toBeInTheDocument();
  });

  it("pins the new image immediately without showing an import message box", async () => {
    renderPage();
    await importOneFile();

    expect(await screen.findByRole("link", { name: "grid2" })).toBeInTheDocument();
    expect(screen.queryByTestId("import-confirmation")).not.toBeInTheDocument();
    expect(screen.queryByText(/It is ready/i)).not.toBeInTheDocument();
  });

  it("sends the selected ordering to both the summary and preview queries", async () => {
    const user = userEvent.setup();
    renderPage();
    await waitFor(() => expect(getHomeEntryPage).toHaveBeenCalled());

    await user.selectOptions(screen.getByLabelText("Sort field"), "display_name");
    await user.selectOptions(screen.getByLabelText("Sort direction"), "asc");

    await waitFor(() =>
      expect(getHomeEntryPage).toHaveBeenCalledWith(
        expect.objectContaining({ ordering: "display_name" })
      )
    );
  });

  it("does not offer an unassigned experiment filter", async () => {
    renderPage();
    await screen.findByRole("link", { name: "Liver study" });

    expect(
      screen.queryByRole("option", { name: "Not in an experiment" })
    ).not.toBeInTheDocument();
  });

  it("bulk selection is limited to image cards actually shown in previews", async () => {
    const user = userEvent.setup();
    renderPage();
    await screen.findByText("Liver 01");

    expect(screen.queryByLabelText("Select Liver 01")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Select images" }));
    await user.click(await screen.findByLabelText("Select Liver 01"));

    expect(await screen.findByTestId("library-selection-bar")).toHaveTextContent(
      "1 image selected"
    );
  });

  it("lists experiments with five-image previews and links the experiment name", async () => {
    renderPage();
    const experimentLink = await screen.findByRole("link", { name: "Liver study" });
    expect(experimentLink).toHaveAttribute(
      "href",
      "/experiments/11111111-1111-1111-1111-111111111111"
    );
    expect(await screen.findByText("Previewing 1 of 1 image")).toBeInTheDocument();
    expect(screen.getByText("Glucose infusion sample")).toBeInTheDocument();
    await waitFor(() =>
      expect(getHomeEntryPage).toHaveBeenCalledWith(
        expect.objectContaining({
          experiment: "11111111-1111-1111-1111-111111111111",
          limit: 5,
        })
      )
    );
  });
});
