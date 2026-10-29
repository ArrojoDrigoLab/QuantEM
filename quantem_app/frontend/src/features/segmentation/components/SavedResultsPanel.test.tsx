import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import { apiRequest } from "@/shared/api/core/http";
import type { AssetDetail } from "@/shared/types";
import { SavedResultsPanel } from "./SavedResultsPanel";

vi.mock("@/shared/api/core/http", () => ({
  apiRequest: vi.fn(),
  resolveApiUrl: (value: string) => value,
}));
vi.mock("@/shared/api/assets", () => ({
  getAssetNgffUrl: () => "/image.zarr",
}));
vi.mock("@/viewer/viewportSync/useViewportSyncGroup", () => ({
  useViewportSyncGroup: () => ({ viewport: null, publishFromViewer: vi.fn() }),
}));
vi.mock("@/viewer/components/ImageViewer", () => ({
  ImageViewer: ({ overlays }: { overlays?: { bitmapOverlays?: { image: string; opacity: number }[] } }) => {
    const overlay = overlays?.bitmapOverlays?.[0];
    return (
      <div
        data-testid="comparison-viewer"
        data-overlay={overlay?.image ?? ""}
        data-opacity={overlay?.opacity ?? ""}
      />
    );
  },
}));

const image = {
  id: "image",
  display_name: "Cell",
  width: 512,
  height: 256,
  pixel_size_nm: 8,
} as AssetDetail;

const results = [
  {
    id: "new",
    name: "New model · 0.5",
    source_model: "quantem:mito",
    threshold: 0.5,
    is_canonical: true,
    available: true,
    preview_url: "/new.png",
    created_at: "2026-10-27T00:00:00Z",
  },
  {
    id: "old",
    name: "Earlier model · 0.7",
    source_model: "quantem:mito",
    threshold: 0.7,
    is_canonical: false,
    available: true,
    preview_url: "/old.png",
    created_at: "2026-10-26T00:00:00Z",
  },
];

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(apiRequest).mockImplementation(async (url, options) =>
    options?.method === "POST"
      ? { job_id: "job" }
      : url.includes("/jobs/")
        ? { status: "SUCCESS" }
        : { results }
  );
});

function renderPanel(overrides: Partial<Parameters<typeof SavedResultsPanel>[0]> = {}) {
  const props = {
    segmentationId: "seg",
    image,
    revision: "1",
    open: true,
    onClose: vi.fn(),
    onResultsChange: vi.fn(),
    onQueued: vi.fn(),
    ...overrides,
  };
  return { ...render(<SavedResultsPanel {...props} />), props };
}

it("loads retained runs while closed without rendering the old accordion", async () => {
  const onResultsChange = vi.fn();
  renderPanel({ open: false, onResultsChange });
  await waitFor(() => expect(onResultsChange).toHaveBeenCalledWith(results));
  expect(screen.queryByText("Model results")).not.toBeInTheDocument();
  expect(screen.queryByLabelText("Compare model runs")).not.toBeInTheDocument();
});

it("opens with the confirmed run left, newest alternative right, and one shared opacity", async () => {
  renderPanel();
  const comparison = await screen.findByLabelText("Compare model runs");
  const left = within(screen.getByRole("group", { name: "Confirmed model view" }));
  const right = within(screen.getByRole("group", { name: "Comparison model view" }));
  await waitFor(() => {
    expect(left.getByRole("combobox")).toHaveValue("new");
    expect(right.getByRole("combobox")).toHaveValue("old");
  });
  expect(within(left.getByRole("combobox")).getByRole("option", { name: /Earlier model/ })).toBeDisabled();
  expect(within(right.getByRole("combobox")).getByRole("option", { name: /New model/ })).toBeDisabled();
  expect(screen.queryByLabelText("Side by side")).not.toBeInTheDocument();
  expect(screen.queryByText(/Overview zoom/)).not.toBeInTheDocument();

  const viewers = within(comparison).getAllByTestId("comparison-viewer");
  expect(viewers[0]).toHaveAttribute("data-overlay", "/new.png");
  expect(viewers[1]).toHaveAttribute("data-overlay", "/old.png");
  fireEvent.change(screen.getByLabelText("Model result opacity"), {
    target: { value: "0.35" },
  });
  await waitFor(() => {
    expect(viewers[0]).toHaveAttribute("data-opacity", "0.35");
    expect(viewers[1]).toHaveAttribute("data-opacity", "0.35");
  });
});

it("queues a noncanonical choice only after the confirmation dialog", async () => {
  const user = userEvent.setup();
  renderPanel();
  const right = within(await screen.findByRole("group", { name: "Comparison model view" }));
  await waitFor(() => expect(right.getByRole("combobox")).toHaveValue("old"));
  await user.click(right.getByRole("button", { name: "Confirm this model" }));
  const dialog = screen.getByRole("dialog");
  expect(within(dialog).getByRole("checkbox", { name: "New model · 0.5" })).toBeChecked();
  expect(
    vi.mocked(apiRequest).mock.calls.filter(([, options]) => options?.method === "POST")
  ).toHaveLength(0);
  await user.click(within(dialog).getByRole("button", { name: "Confirm" }));
  await waitFor(() =>
    expect(apiRequest).toHaveBeenCalledWith("/api/segmentations/seg/results/", {
      method: "POST",
      body: JSON.stringify({ result_id: "old", delete_result_ids: ["new"] }),
    })
  );
});
