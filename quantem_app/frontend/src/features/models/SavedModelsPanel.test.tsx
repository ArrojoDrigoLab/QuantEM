import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import { SavedModelsPanel } from "./SavedModelsPanel";
import { apiRequest } from "@/shared/api/core/http";
import { getSegmentationTypes } from "@/shared/api/assets";

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

it("keeps a model visible when deletion is blocked by an active task", async () => {
  const user = userEvent.setup();
  vi.mocked(apiRequest).mockImplementation(async (_url, options) => {
    if (options?.method === "DELETE") throw new Error("This model is in use by a task.");
    return [model];
  });
  render(<SavedModelsPanel />);
  await user.click(await screen.findByRole("button", { name: "Delete weights…" }));
  await user.click(screen.getByRole("button", { name: "Delete weights" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("in use by a task");
  expect(screen.getByText(model.name)).toBeInTheDocument();
});

it("reports a failed model listing without claiming that no models exist", async () => {
  vi.mocked(apiRequest).mockRejectedValue(new Error("Service unavailable"));
  render(<SavedModelsPanel />);
  expect(await screen.findByRole("alert")).toHaveTextContent("Service unavailable");
  expect(screen.queryByText(/No saved fine-tunes/)).not.toBeInTheDocument();
});
