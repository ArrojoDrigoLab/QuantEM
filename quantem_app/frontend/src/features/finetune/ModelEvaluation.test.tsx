import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, expect, it, vi } from "vitest";
import { ModelEvaluation } from "./ModelEvaluation";
import { apiRequest } from "@/shared/api/core/http";
import { getFineTuneScope } from "@/shared/api/finetune";
import type { SegmentationType } from "@/shared/types/images";

vi.mock("@/shared/api/core/http", () => ({ apiRequest: vi.fn() }));
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
