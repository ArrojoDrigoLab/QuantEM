import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import { apiRequest } from "@/shared/api/core/http";
import { SavedResultsPanel } from "./SavedResultsPanel";

vi.mock("@/shared/api/core/http", () => ({ apiRequest: vi.fn(), resolveApiUrl: (value: string) => value }));
vi.mock("@/shared/api/assets", () => ({ getAssetNgffThumbnailUrl: () => "/image.png" }));
const results = [
  { id: "new", name: "New model · 0.5", is_canonical: true, available: true, preview_url: "/new.png", created_at: "2026-10-27" },
  { id: "old", name: "Earlier model · 0.7", is_canonical: false, available: true, preview_url: "/old.png", created_at: "2026-10-26" },
];
beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(apiRequest).mockImplementation(async (url, options) => options?.method === "POST" ? { job_id: "job" } : url.includes("/jobs/") ? { status: "SUCCESS" } : { results });
});

it("compares either model with independent 30% overlays and marks only the confirmed model", async () => {
  const user = userEvent.setup();
  render(<SavedResultsPanel segmentationId="seg" assetId="image" revision="1" onQueued={vi.fn()} />);
  await user.click(await screen.findByText(/Model results/));
  await user.click(screen.getByLabelText("Side by side"));
  const left = within(screen.getByRole("group", { name: "Prediction view" }));
  const right = within(screen.getByRole("group", { name: "Compare with view" }));
  expect(left.getByRole("combobox")).toHaveValue("new");
  expect(right.getByRole("combobox")).toHaveValue("old");
  for (const side of [left, right]) {
    expect(side.getByRole("option", { name: /^✓ New model · 0.5 · Confirmed/ })).toBeInTheDocument();
    expect(side.getByRole("option", { name: /^Earlier model/ })).toHaveTextContent(/^Earlier/);
    expect(side.getByAltText(/prediction overlay/)).toHaveStyle({ opacity: "0.3" });
    expect(side.getByRole("slider")).toHaveValue("0.3");
  }
  expect(left.queryByRole("button", { name: "Confirm this model" })).not.toBeInTheDocument();
  expect(right.getByRole("button", { name: "Confirm this model" }).parentElement).toContainElement(right.getByRole("combobox"));
  fireEvent.change(left.getByRole("slider"), { target: { value: "0.65" } });
  expect(left.getByAltText(/prediction overlay/)).toHaveStyle({ opacity: "0.65" });
  expect(right.getByAltText(/prediction overlay/)).toHaveStyle({ opacity: "0.3" });
  await user.selectOptions(left.getByRole("combobox"), "old");
  await user.selectOptions(right.getByRole("combobox"), "new");
  expect(left.getByRole("button", { name: "Confirm this model" })).toBeInTheDocument();
  expect(right.queryByRole("button", { name: "Confirm this model" })).not.toBeInTheDocument();
  expect(left.getByAltText(/prediction overlay/)).toHaveAttribute("src", "/old.png");
  expect(right.getByAltText(/prediction overlay/)).toHaveAttribute("src", "/new.png");
  expect(vi.mocked(apiRequest).mock.calls.filter(([, options]) => options?.method === "POST")).toHaveLength(0);
});

it("does not change or delete results when the confirmation is opened or cancelled", async () => {
  const user = userEvent.setup();
  render(<SavedResultsPanel segmentationId="seg" assetId="image" revision="1" onQueued={vi.fn()} />);
  await user.click(await screen.findByText(/Model results/));
  await user.selectOptions(screen.getByLabelText("Prediction"), "old");
  await user.click(screen.getByRole("button", { name: "Confirm this model" }));
  const dialog = screen.getByRole("dialog");
  expect(within(dialog).getByText("Use this model's outputs as the default for analysis? Your manually annotated areas will still override model outputs in ROIs marked as 'Done'.")).toBeInTheDocument();
  expect(within(dialog).getByRole("checkbox", { name: "New model · 0.5" })).toBeChecked();
  expect(vi.mocked(apiRequest).mock.calls.filter(([, options]) => options?.method === "POST")).toHaveLength(0);
  await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(vi.mocked(apiRequest).mock.calls.filter(([, options]) => options?.method === "POST")).toHaveLength(0);
});

it.each([true, false])("queues selection only on final confirmation, with delete old outputs = %s", async (deleteOld) => {
  const user = userEvent.setup();
  render(<SavedResultsPanel segmentationId="seg" assetId="image" revision="1" onQueued={vi.fn()} />);
  await user.click(await screen.findByText(/Model results/));
  await user.selectOptions(screen.getByLabelText("Prediction"), "old");
  await user.click(screen.getByRole("button", { name: "Confirm this model" }));
  const dialog = screen.getByRole("dialog");
  expect(within(dialog).getByRole("checkbox", { name: "New model · 0.5" })).toBeChecked();
  if (!deleteOld) await user.click(within(dialog).getByRole("checkbox", { name: "New model · 0.5" }));
  await user.click(within(dialog).getByRole("button", { name: "Confirm" }));
  await waitFor(() => expect(apiRequest).toHaveBeenCalledWith("/api/segmentations/seg/results/", {
    method: "POST", body: JSON.stringify({ result_id: "old", delete_result_ids: deleteOld ? ["new"] : [] }),
  }));
  // Acceptance comes from the server after extraction, not from clicking Confirm.
  expect(screen.getByRole("option", { name: /^✓ New model/ })).toBeInTheDocument();
  expect(screen.getByRole("option", { name: /^Earlier model/ })).toBeInTheDocument();
});
