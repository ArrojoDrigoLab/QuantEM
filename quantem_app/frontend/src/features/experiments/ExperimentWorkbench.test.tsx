import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Link, MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ExperimentWorkbench } from "./ExperimentWorkbench";
import { getWorkspace, saveSample, type WorkspaceImage } from "@/shared/api/workspace";
import { apiRequest } from "@/shared/api/core/http";
import type { Experiment } from "@/shared/types/common";

vi.mock("@/shared/api/workspace", () => ({ getWorkspace: vi.fn(), saveSample: vi.fn() }));
vi.mock("@/shared/api/core/http", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/shared/api/core/http")>(),
  apiRequest: vi.fn(), resolveApiUrl: (url: string) => url,
}));
vi.mock("@/shared/api/assets", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/shared/api/assets")>(),
  getSegmentationTypes: vi.fn().mockResolvedValue([]),
}));

const experiment: Experiment = {
  id: "experiment", name: "Nanotomy T1D", notes: "", datasets: [
    { id: "d1", experiment: "experiment", name: "Donor fields", notes: "", asset_count: 51, created_at: null, updated_at: null },
  ], asset_count: 51, ungrouped_asset_count: 0, created_at: null, updated_at: null,
};
const images: WorkspaceImage[] = Array.from({ length: 51 }, (_, i) => ({
  id: `image-${i}`, name: `Field ${String(i + 1).padStart(2, "0")}`, stage: "DONE", pixel_size_nm: 8,
  datasets: [{ id: "d1", name: "Donor fields" }], sample_id: "", group: "",
  segmentations: [{ id: `seg-${i}`, type_id: "mito", type: "mitochondria", name: "Mitochondria",
    measurement_mode: "objects", stage: "COMPLETED", review_complete: true, confirmed_count: 7, provenance: {}, short_name: "Mitos", coverage: "whole" }],
}));
const identity = { image: "Field 01", asset_id: "image-0", organelle: "mitochondria", sample_id: "Donor 1", group: "T1D", measurement_area: "Whole image", pixel_size_nm: 8 };
function mount() {
  return render(<MemoryRouter><ExperimentWorkbench experiment={experiment} refreshKey={0}><div>Cards</div></ExperimentWorkbench></MemoryRouter>);
}

beforeEach(() => {
  vi.clearAllMocks();
  window.sessionStorage.clear();
  vi.spyOn(window, "scrollTo").mockImplementation(() => undefined);
  vi.mocked(getWorkspace).mockResolvedValue({ images });
  vi.mocked(saveSample).mockResolvedValue({});
  vi.mocked(apiRequest).mockImplementation(async (url) => url.includes("/analysis-history/")
    ? { results: [], count: 0, offset: 0, limit: 25, next_offset: null } : []);
});

describe("Experiment workbench", () => {
  it("keeps selection in analysis and retains it across filters and revisits", async () => {
    const user = userEvent.setup();
    const view = mount();
    await screen.findByText("Field 01");
    expect(screen.getByRole("checkbox", {name: "Select Field 01"})).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Select all filtered images/ })).toBeEnabled();
    expect(screen.getByRole("columnheader", { name: "Segmentations" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Analyses" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Analysis" }));
    await user.click(screen.getByRole("button", { name: "New analysis" }));
    await user.click(screen.getByRole("button", { name: "Select all matching (51)" }));
    await user.selectOptions(screen.getByLabelText("Dataset"), "d1");
    await user.click(screen.getByRole("button", { name: "Select all matching (51)" }));
    expect(screen.getByRole("heading", { name: "Measure selected images (51)" })).toBeInTheDocument();
    await user.type(screen.getByLabelText("Find images"), "Field 51");
    expect(screen.getByRole("checkbox", { name: "Select Field 51" })).toBeChecked();
    view.unmount();
    mount();
    await screen.findByText("Field 01");
    await user.click(screen.getByRole("button", { name: "Analysis" }));
    await user.click(screen.getByRole("button", { name: "New analysis" }));
    expect(screen.getByRole("checkbox", { name: "Select Field 51" })).toBeChecked();
    expect(screen.getByRole("heading", { name: "Measure selected images (51)" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Clear selection" }));
    expect(screen.getByRole("heading", { name: "Measure selected images (0)" })).toBeInTheDocument();
  });

  it("requires a readiness check, queues the deduplicated selection and exports partial results", async () => {
    const user = userEvent.setup();
    vi.mocked(apiRequest).mockImplementation(async (url, options) => {
      if (url.includes("/analysis-history/")) return { results: [], count: 0, offset: 0, limit: 25, next_offset: null };
      if (url.includes("/analysis-batches/") && options?.method === "POST") {
        const data = JSON.parse(options.body as string);
        if (data.preview) return { items: [{ identity, ready: true, reason: "" }] };
        return { id: "batch", name: data.name, created_at: "2026-10-01", items: [
          { identity, status: "SUCCESS", reason: "", current: true, metrics: { n_objects: 7 } },
          { identity: { ...identity, image: "Field 02" }, status: "FAILED", reason: "Missing mask" },
        ] };
      }
      return [];
    });
    mount();
    await screen.findByText("Field 01");
    await user.click(screen.getByRole("button", { name: "Analysis" }));
    await user.click(screen.getByRole("button", { name: "New analysis" }));
    await user.click(screen.getByRole("button", { name: "Select all matching (51)" }));
    expect(screen.getByRole("heading", { name: "Measure selected images (51)" })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Mitochondria" })).toBeChecked();
    expect(screen.getByRole("button", { name: "Run ready images" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Check readiness" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Run ready images" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Run ready images" }));
    await screen.findByText("Missing mask");
    expect(screen.getByRole("button", { name: "Retry failed" })).toBeEnabled();
    expect(screen.getByRole("link", { name: "Export measurements ZIP" })).toHaveAttribute("href", "/api/analysis-batches/batch/export/");
    expect(screen.queryByRole("checkbox", { name: "Include object tables" })).not.toBeInTheDocument();
    const submissions = vi.mocked(apiRequest).mock.calls.filter(([, options]) => options?.method === "POST").map(([, options]) => JSON.parse(options!.body as string));
    expect(submissions.at(-1).asset_ids).toHaveLength(51);
    expect(submissions.at(-1).segmentation_type_ids.length).toBeGreaterThan(0);
    expect(submissions.at(-1).output_files["organelle_objects.csv"]).toBe(true);
    const resultRows = within(screen.getAllByRole("table").at(-1)!).getAllByRole("row");
    expect(within(resultRows[1]).getByText("7")).toBeInTheDocument();
  });

  it("records explicit biological metadata only for selected images", async () => {
    const user = userEvent.setup();
    mount();
    await screen.findByText("Field 01");
    await user.click(screen.getByRole("button", { name: "Analysis" }));
    await user.click(screen.getByRole("button", { name: "New analysis" }));
    await user.click(screen.getByRole("checkbox", { name: "Select Field 01" }));
    await user.click(screen.getByText("Sample groups", { exact: true }));
    await user.type(screen.getByLabelText("Sample ID"), "Donor 12");
    await user.type(screen.getByLabelText("Group"), "T1D");
    await user.click(screen.getByRole("button", { name: "Save for 1 images" }));
    await waitFor(() => expect(saveSample).toHaveBeenCalledWith("experiment", "image-0", "Donor 12", "T1D"));
    expect(saveSample).toHaveBeenCalledTimes(1);
  });

  it("shows a loading failure with a working retry", async () => {
    vi.mocked(getWorkspace).mockRejectedValue(new Error("Library offline"));
    mount();
    expect(await screen.findByRole("alert")).toHaveTextContent("Library offline");
    vi.mocked(getWorkspace).mockResolvedValue({ images });
    await userEvent.click(screen.getByRole("button", { name: "Refresh status" }));
    await screen.findByText("Field 01");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("reloads models on repeated tab and workspace navigation clicks and the shared refresh", async () => {
    const user = userEvent.setup();
    let modelName = "First saved model";
    vi.mocked(apiRequest).mockImplementation(async (url) => url.startsWith("/api/finetune/models/") ? [{
      id: "saved", name: modelName, base_model: "quantem:mito", segmentation_type: "mito",
      mode: "head", status: "SUCCESS", created_at: "2026-10-01", asset_ids: [],
    }] : []);
    render(<MemoryRouter initialEntries={["/experiments/experiment?tab=models"]}>
      <Link to="/experiments/experiment?tab=models">Workspace Models</Link>
      <ExperimentWorkbench experiment={experiment} refreshKey={0}><div>Cards</div></ExperimentWorkbench>
    </MemoryRouter>);
    await screen.findByText("First saved model");
    const models = screen.getByRole("region", { name: "Saved fine-tuned models" });
    expect(within(models).queryByRole("button", { name: "Refresh" })).not.toBeInTheDocument();

    modelName = "Updated from the tab";
    await user.click(screen.getByRole("button", { name: "Models" }));
    await screen.findByText(modelName);

    modelName = "Updated from the workspace link";
    await user.click(screen.getByRole("link", { name: "Workspace Models" }));
    await screen.findByText(modelName);

    modelName = "Updated from the refresh icon";
    await user.click(screen.getByRole("button", { name: "Refresh status" }));
    await screen.findByText(modelName);
  });

  it("pages the browsing table and preserves the dataset cards view", async () => {
    const user = userEvent.setup();
    mount();
    await screen.findByText("Field 01");
    await user.click(screen.getByRole("button", { name: "Next page" }));
    expect(screen.getByText("Page 2 of 2 · 51 images")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open Field 51" })).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("View"), "cards");
    expect(screen.getByText("Cards")).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });
});
