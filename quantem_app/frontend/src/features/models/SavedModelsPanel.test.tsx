import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import { SavedModelsPanel } from "./SavedModelsPanel";
import { apiRequest } from "@/shared/api/core/http";
import { getSegmentationTypes } from "@/shared/api/assets";
import type { TrainingTrace } from "@/shared/types/finetune";

vi.mock("@/shared/api/core/http", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/shared/api/core/http")>(), apiRequest: vi.fn(),
}));
vi.mock("@/shared/api/assets", () => ({ getSegmentationTypes: vi.fn() }));
vi.mock("@/features/finetune/FineTuneDialog", () => ({
  FineTuneDialog: (props: { initialParentAdapterId?: string; initialSavedAdapterId?: string }) =>
    <div role="dialog">{props.initialParentAdapterId ? `Continue ${props.initialParentAdapterId}` : `Apply ${props.initialSavedAdapterId}`}</div>,
}));
const model = { id: "parent", name: "T1D mitochondria v1", base_model: "quantem:mito", segmentation_type: "mito",
  mode: "head", status: "SUCCESS", created_at: "2026-10-01", asset_ids: ["field"], experiment: { id: "experiment" } };
beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getSegmentationTypes).mockResolvedValue([]);
  vi.mocked(apiRequest).mockResolvedValue([model]);
});

it("shows the reserved-name explanation when a rename uses a deleted model's name", async () => {
  const message = "This name was used by a previous model and cannot be re-used. Make a new name.";
  vi.mocked(apiRequest).mockImplementation(async (_url, options) => {
    if (options?.method === "PATCH") throw new Error(JSON.stringify({ code: "model_name_reserved", detail: message }));
    return [model];
  });
  render(<SavedModelsPanel />);
  await userEvent.click(await screen.findByRole("button", { name: "Rename" }));
  await userEvent.clear(screen.getByLabelText("Model name"));
  await userEvent.type(screen.getByLabelText("Model name"), "Deleted name");
  await userEvent.click(screen.getByRole("button", { name: "Save name" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(message);
  expect(screen.getByLabelText("Model name")).toHaveValue("Deleted name");
});

it("opens continuation and reuse with the saved model identity", async () => {
  const user = userEvent.setup();
  const view = render(<SavedModelsPanel experimentId="experiment" />);
  await user.click(await screen.findByRole("button", { name: "Continue training" }));
  expect(screen.getByRole("dialog")).toHaveTextContent("Continue parent");
  view.unmount();
  render(<SavedModelsPanel experimentId="experiment" />);
  await user.click(await screen.findByRole("button", { name: "Apply to images" }));
  expect(screen.getByRole("dialog")).toHaveTextContent("Apply parent");
});

it("renames, archives and restores through the model lifecycle API", async () => {
  const user = userEvent.setup();
  render(<SavedModelsPanel />);
  await user.click(await screen.findByRole("button", { name: "Rename" }));
  await user.clear(screen.getByLabelText("Model name"));
  await user.type(screen.getByLabelText("Model name"), "New model name");
  await user.click(screen.getByRole("button", { name: "Save name" }));
  await waitFor(() => expect(apiRequest).toHaveBeenCalledWith("/api/finetune/models/parent/", {
    method: "PATCH", body: JSON.stringify({ name: "New model name" }),
  }));
  await user.click(screen.getByRole("button", { name: "Archive" }));
  await waitFor(() => expect(apiRequest).toHaveBeenCalledWith("/api/finetune/models/parent/", {
    method: "PATCH", body: JSON.stringify({ archived: true }),
  }));
  vi.mocked(apiRequest).mockResolvedValue([{ ...model, archived_at: "2026-10-02" }]);
  await user.click(screen.getByRole("checkbox", { name: "Show archived" }));
  await user.click(await screen.findByRole("button", { name: "Restore to list" }));
  await waitFor(() => expect(apiRequest).toHaveBeenCalledWith("/api/finetune/models/parent/", {
    method: "PATCH", body: JSON.stringify({ archived: false }),
  }));
});

it.each([
  { reason: "an active task", message: "This model is in use by a task. Wait or cancel the task first." },
  { reason: "another saved model", message: "This model was used to train another saved model and cannot be deleted. You can archive it instead." },
])("keeps the model and shows a red error when deletion is blocked by $reason", async ({ message }) => {
  const user = userEvent.setup();
  vi.mocked(apiRequest).mockImplementation(async (_url, options) => {
    if (options?.method === "DELETE") throw new Error(message);
    return [model];
  });
  render(<SavedModelsPanel />);
  await user.click(await screen.findByRole("button", { name: "Delete model" }));
  expect(screen.getByRole("dialog")).toHaveTextContent(`Delete the model “${model.name}”? This will permanently delete this model. Existing segmentations and analyses will remain available, but this model will no longer be available for new segmentations.`);
  expect(screen.getByRole("dialog")).not.toHaveTextContent(message);
  await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Delete model" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(message);
  expect(screen.getByRole("alert")).toHaveClass("text-red-700");
  expect(screen.getByText(model.name)).toBeInTheDocument();
});

it("reports a failed model listing without claiming that no models exist", async () => {
  vi.mocked(apiRequest).mockRejectedValue(new Error("Service unavailable"));
  render(<SavedModelsPanel />);
  expect(await screen.findByRole("alert")).toHaveTextContent("Service unavailable");
  expect(screen.queryByText(/No saved fine-tunes/)).not.toBeInTheDocument();
});

it("shows both saved selection and refit histories without opening an apply dialog", async () => {
  const user = userEvent.setup();
  const training: TrainingTrace = { steps: 30, n_patches: 4, patches_seen: 4,
    selected_step: 30, stopped_early: false,
    history: [{ step: 30, train_loss: 0.42, validation_loss: null, validation_dice: null }] };
  const selection = { ...training, steps: 50, stopped_early: true,
    history: [{ step: 30, train_loss: 0.45, validation_loss: 0.52, validation_dice: 0.71 }] };
  vi.mocked(apiRequest).mockResolvedValue([{ ...model,
    training_rounds: [{ round: 1, training, checkpoint_selection: { selection } }] }]);
  render(<SavedModelsPanel />);
  await user.click(await screen.findByText("Training history"));
  expect(screen.getByRole("table", { name: "Round 1 checkpoint selection" })).toHaveTextContent("0.5200");
  expect(screen.getByRole("table", { name: "Round 1 final refit" })).toHaveTextContent("0.4200");
  expect(screen.getByText(/Validation selected step 30/)).toHaveTextContent("stopped early");
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});

it("explains why a requested validation split was unavailable", async () => {
  vi.mocked(apiRequest).mockResolvedValue([{ ...model, training_rounds: [{
    round: 1, training: { steps: 20, n_patches: 1, patches_seen: 1, history: [] },
    checkpoint_selection: { selection: null, note: "Checkpoint selection needs at least two training images." },
  }] }]);
  render(<SavedModelsPanel />);
  expect(await screen.findByText(/Checkpoint selection needs/)).toBeInTheDocument();
});
