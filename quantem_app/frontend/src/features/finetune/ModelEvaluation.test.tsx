import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, expect, it, vi } from "vitest";
import { useModelApplicationChoice } from "@/features/experiments/modelApplicationChoice";
import { ModelEvaluation } from "./ModelEvaluation";
import { apiRequest } from "@/shared/api/core/http";
import { getFineTuneScope } from "@/shared/api/finetune";
import type { SegmentationType } from "@/shared/types/images";
import type { FineTuneScopeResponse, FineTuneAdapterSummary } from "@/shared/types/finetune";

vi.mock("@/shared/api/core/http", () => ({ apiRequest: vi.fn() }));
vi.mock("@/features/models/useModelCatalogue", () => ({ useModelCatalogue: () => ({ catalogue: null }) }));
vi.mock("@/shared/api/finetune", () => ({ getFineTuneScope: vi.fn() }));
vi.mock("@/features/experiments/BulkModelPanel", () => ({
  BulkModelPanel: ({ selectedIds }: { selectedIds: string[] }) => <p>Apply to {selectedIds.join(", ")}</p>,
}));
const type: SegmentationType = {
  id: "mito", internal_name: "quantem_internal_mito", short_name: "Mito",
  long_name: "Mitochondria", default_color: null, tags: [], created_at: "", updated_at: "",
};
beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getFineTuneScope).mockResolvedValue({ experiments: [{
    id: "experiment", name: "Selection", datasets: [], ungrouped_images: [
      { id: "image", name: "Field 1", annotation_count: 0, confirmed_areas: 0, done_rois: 0 },
      { id: "other", name: "Field 2", annotation_count: 0, confirmed_areas: 0, done_rois: 0 },
    ],
  }] });
  vi.mocked(apiRequest).mockImplementation(async (_url, options) => {
    if (_url === "/api/finetune/evaluations/eval/") return { id: "eval", status: "SUCCESS", message: "Finished", created_at: "2026-10-28", results: { mean_dice: 0.8, per_roi: [], notes: [] } };
    if (options?.method !== "POST") return [];
    const body = JSON.parse(options.body as string);
    return body.preview ? { threshold: 0.6, rois: [{ id: "roi", name: "Completed ROI", image_id: "image", training_overlap: true }] }
      : { id: "eval", status: "SUCCESS", message: "Finished", created_at: "2026-10-28", results: { mean_dice: 0.8, per_roi: [], notes: [] } };
  });
});

it("starts with visual evaluation requiring no annotations and preserves the selected images", async () => {
  render(<MemoryRouter><ModelEvaluation types={[type]} initialTypeId="mito" initialExperimentId="experiment" initialAssetIds={["image"]} adapters={[]} /></MemoryRouter>);
  expect(await screen.findByText("Apply to image")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Inspect Field 1" })).toBeInTheDocument();
  expect(screen.queryByRole("link", { name: "Inspect Field 2" })).not.toBeInTheDocument();
  expect(screen.queryByText("Evaluation ROIs")).not.toBeInTheDocument();
});

it("flags training overlap while allowing a chosen ROI evaluation", async () => {
  const user = userEvent.setup();
  render(<MemoryRouter><ModelEvaluation types={[type]} initialTypeId="mito" initialExperimentId="experiment" initialAssetIds={["image"]} adapters={[]} /></MemoryRouter>);
  await screen.findByText("Apply to image");
  await user.click(screen.getByRole("button", { name: "Score against ROIs" }));
  await user.click(screen.getByRole("button", { name: "Choose ROIs from 1 images" }));
  expect(await screen.findByText(/Image used in training; evaluation is allowed/)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Evaluate selected ROIs" }));
  await waitFor(() => expect(screen.getByText("Mean ROI Dice: 0.800")).toBeInTheDocument());
  const submission = vi.mocked(apiRequest).mock.calls.filter(([, options]) => options?.method === "POST").at(-1)!;
  expect(JSON.parse(submission[1]!.body as string)).toMatchObject({ asset_ids: ["image"], roi_ids: ["roi"], threshold: 0.6 });
});

it.each(["er", "nucleus", "ld"])("keeps %s context in the evaluation and application handoff", async (organelle) => {
  const user = userEvent.setup();
  const selectedType = { ...type, id: organelle, internal_name: `quantem_internal_${organelle}` };
  render(<MemoryRouter><ModelEvaluation types={[selectedType]} initialTypeId={organelle} initialExperimentId="experiment" initialAssetIds={["image"]} adapters={[]} initialBaseModel={`quantem:${organelle}`} /></MemoryRouter>);
  await screen.findByText("Apply to image");
  expect(useModelApplicationChoice.getState().choices.experiment.organelles).toEqual([organelle]);
  await user.click(screen.getByRole("button", { name: "Score against ROIs" }));
  await user.click(screen.getByRole("button", { name: "Choose ROIs from 1 images" }));
  const submission = vi.mocked(apiRequest).mock.calls.find(([, options]) => options?.method === "POST")!;
  expect(JSON.parse(submission[1]!.body as string)).toMatchObject({ segmentation_type: organelle, base_model: `quantem:${organelle}` });
});

it("uses a historical score's settings even after the current controls change", async () => {
  const historical = { id: "old", status: "SUCCESS", message: "Finished", created_at: "2026-10-28",
    settings: { experiment_id: "experiment", segmentation_type: "mito", asset_ids: ["image"], base_model: "omniem:mito", adapter_id: null, threshold: 0.73, rois: [] },
    results: { mean_dice: 0.91, per_roi: [], notes: [] } };
  vi.mocked(apiRequest).mockImplementation(async (url) => url.includes("/evaluations/old/") ? historical : url.includes("/evaluations/?") ? [historical] : []);
  const user = userEvent.setup();
  render(<MemoryRouter><ModelEvaluation types={[type]} initialTypeId="mito" initialExperimentId="experiment" initialAssetIds={["image"]} adapters={[]} /></MemoryRouter>);
  await user.click(screen.getByRole("button", { name: "Score against ROIs" }));
  await user.selectOptions(await screen.findByLabelText("Saved evaluations"), "old");
  await screen.findByText("Mean ROI Dice: 0.910");
  await user.selectOptions(screen.getByLabelText("Model"), "quantem");
  expect(screen.getByText(/Evaluated settings:.*threshold 0.73/)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Use evaluated settings for application" }));
  expect(useModelApplicationChoice.getState().choices.experiment).toMatchObject({ family: "omniem", organelles: ["mito"], adapter: "", threshold: 0.73 });
});

it("does not submit images from the previous organelle while its new scope is loading", async () => {
  let resolveScope!: (value: FineTuneScopeResponse) => void;
  const pending = new Promise<FineTuneScopeResponse>((resolve) => { resolveScope = resolve; });
  const er = { ...type, id: "er", internal_name: "quantem_internal_er", long_name: "ER" };
  const user = userEvent.setup();
  render(<MemoryRouter><ModelEvaluation types={[type, er]} initialTypeId="mito" initialExperimentId="experiment" adapters={[]} /></MemoryRouter>);
  await screen.findByText("Apply to image, other");
  await user.click(screen.getByRole("button", { name: "Score against ROIs" }));
  vi.mocked(getFineTuneScope).mockReturnValueOnce(pending);
  await user.selectOptions(screen.getByLabelText("Organelle"), "er");
  expect(screen.getByRole("button", { name: "Choose ROIs from 0 images" })).toBeDisabled();
  await act(async () => { resolveScope({ experiments: [] }); });
  expect(apiRequest).not.toHaveBeenCalledWith(expect.anything(), expect.objectContaining({ method: "POST" }));
});

it("uses adapter updates from the dialog and preserves ROI choices and threshold on reopening", async () => {
  const user = userEvent.setup();
  const props = { types: [type], initialTypeId: "mito", initialExperimentId: "experiment", initialAssetIds: ["image"], adapters: [] as FineTuneAdapterSummary[] };
  const view = render(<MemoryRouter><ModelEvaluation {...props} /></MemoryRouter>);
  await screen.findByText("Apply to image");
  await user.click(screen.getByRole("button", { name: "Score against ROIs" }));
  const newModel: FineTuneAdapterSummary = { id: "new-model", name: "Just trained", base_model: "omniem:mito", status: "SUCCESS", experiment: { id: "experiment", name: "Selection" }, created_at: "", asset_ids: ["image"], asset_count: 1 };
  view.rerender(<MemoryRouter><ModelEvaluation {...props} adapters={[newModel]} /></MemoryRouter>);
  expect(screen.getByRole("option", { name: "Just trained" })).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Choose ROIs from 1 images" }));
  const threshold = await screen.findByLabelText("Threshold");
  await user.clear(threshold); await user.type(threshold, "0.72");
  await user.click(screen.getByRole("checkbox"));
  view.unmount();
  render(<MemoryRouter><ModelEvaluation {...props} /></MemoryRouter>);
  await user.click(await screen.findByRole("button", { name: "Choose ROIs from 1 images" }));
  expect(await screen.findByLabelText("Threshold")).toHaveValue(0.72);
  expect(screen.getByRole("checkbox")).not.toBeChecked();
  expect(screen.getByRole("button", { name: "Evaluate selected ROIs" })).toBeDisabled();
});
