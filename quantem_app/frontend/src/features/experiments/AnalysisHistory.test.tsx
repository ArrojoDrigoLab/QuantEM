import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, expect, it, vi } from "vitest";
import { AnalysisHistory } from "./AnalysisHistory";
import { getAnalysisBatch, getAnalysisHistory, type AnalysisHistoryEntry } from "@/shared/api/analysisHistory";
import { getAnalysisRun, getAnalysisTable } from "@/shared/api/analysis";
import type { AnalysisRun } from "@/shared/types/analysis";

vi.mock("@/shared/api/analysisHistory", () => ({ getAnalysisHistory: vi.fn(), getAnalysisBatch: vi.fn() }));
vi.mock("@/shared/api/analysis", () => ({ getAnalysisRun: vi.fn(), getAnalysisTable: vi.fn(), getAnalysisExportUrl: (id: string, name: string) => `/api/analysis/${id}/export/${name}` }));
vi.mock("@/shared/api/core/http", () => ({ resolveApiUrl: (url: string) => url }));
vi.mock("@/features/analysis/components/AnalysisResults", () => ({ AnalysisResults: () => <div>Saved measurements</div> }));

const entry: AnalysisHistoryEntry = { id: "batch", kind: "batch", name: "Donor comparison", created_at: "2026-10-14T12:00:00Z", status: "SUCCESS", images: [{ id: "a", name: "Field A" }, { id: "b", name: "Field B" }], structures: ["Mitos"], measurement_count: 2, measurement_area: "Whole image", completed_image_ids: ["a", "b"] };
beforeEach(() => vi.clearAllMocks());

it("loads older pages on demand and preserves the image filter", async () => {
  vi.mocked(getAnalysisHistory).mockImplementation(async (_experiment, _image, offset = 0) => ({
    results: [{ ...entry, id: offset ? "older" : "recent", name: offset ? "Older analysis" : "Recent analysis" }],
    count: 26, limit: 25, offset, next_offset: offset ? null : 25,
  }));
  render(<MemoryRouter initialEntries={["/?image=a"]}><AnalysisHistory experimentId="exp" images={[]} /></MemoryRouter>);
  await screen.findByRole("button", { name: /Recent analysis/ });
  expect(screen.getByRole("button", { name: "Previous page" })).toBeDisabled();
  await userEvent.click(screen.getByRole("button", { name: "Next page" }));
  await screen.findByRole("button", { name: /Older analysis/ });
  expect(getAnalysisHistory).toHaveBeenLastCalledWith("exp", "a", 25);
  expect(screen.getByRole("button", { name: "Next page" })).toBeDisabled();
  expect(screen.queryByRole("button", { name: /Recent analysis/ })).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Previous page" }));
  await screen.findByRole("button", { name: /Recent analysis/ });
  expect(getAnalysisHistory).toHaveBeenLastCalledWith("exp", "a", 0);
});

it("filters history by image but opens every image and saved setting in the original batch", async () => {
  vi.mocked(getAnalysisHistory).mockResolvedValue({ results: [entry], count: 1, offset: 0, limit: 25, next_offset: null });
  vi.mocked(getAnalysisBatch).mockResolvedValue({ id: "batch", name: entry.name, created_at: entry.created_at, settings: { require_complete: true, reuse: false }, items: entry.images.map((image) => ({ identity: { image: image.name, asset_id: image.id, organelle: "mitochondria", sample_id: image.name, group: "", pixel_size_nm: 2.5, measurement_area: "Whole image" }, status: "SUCCESS", reason: "", params: { seed: 42 }, metrics: { n_objects: 7, area_fraction_mito: 0.00000012345 }, notes: ["No tissue mask was supplied."], run_id: `run-${image.id}` })) });
  render(<MemoryRouter initialEntries={["/experiments/exp?tab=analysis&image=a"]}><AnalysisHistory experimentId="exp" images={[]} /></MemoryRouter>);
  const row = await screen.findByRole("button", { name: /Donor comparison/ });
  expect(getAnalysisHistory).toHaveBeenCalledWith("exp", "a", 0);
  expect(getAnalysisBatch).not.toHaveBeenCalled();
  await userEvent.click(row);
  expect(await screen.findByRole("cell", { name: "Field B" })).toBeInTheDocument();
  expect(screen.getByRole("cell", { name: "Field A" })).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Download" })).toHaveAttribute("href", "/api/analysis-batches/batch/export/?objects=true");
  await userEvent.click(screen.getByText("Analysis settings"));
  expect(screen.getByText("Require whole-image coverage")).toBeVisible();
  expect(screen.getByText("Yes")).toBeVisible();
  expect(screen.getByRole("region", { name: "Notes" })).toHaveTextContent("Field B: No tissue mask was supplied.");

  vi.mocked(getAnalysisRun).mockResolvedValue({ id: "run-b", exports: ["objects.csv"], status: "SUCCESS" } as AnalysisRun);
  vi.mocked(getAnalysisTable).mockResolvedValue({ name: "objects.csv", columns: ["object_id", "area_um2"], rows: [["object-b", "2.75"]], offset: 0, limit: 100, has_more: false });
  const results = screen.getByRole("button", { name: "View results for Field B · mitochondria" });
  await userEvent.click(results);
  const dialog = await screen.findByRole("dialog", { name: "Field B · mitochondria" });
  expect(within(dialog).queryByText(/No tissue mask/)).not.toBeInTheDocument();
  expect(await within(dialog).findByRole("table", { name: "Objects spreadsheet" })).toHaveTextContent("object-b");
  expect(getAnalysisRun).toHaveBeenCalledWith("run-b");
  expect(within(dialog).getByRole("button", { name: "Close results" })).toHaveFocus();
  await userEvent.tab();
  expect(within(dialog).getByRole("tab", { name: "Spreadsheet" })).toHaveFocus();
  await userEvent.keyboard("{ArrowRight}");
  expect(within(dialog).getByRole("tab", { name: "Image averages" })).toHaveAttribute("aria-selected", "true");
  expect(within(dialog).getByRole("table", { name: "Image averages" })).toHaveTextContent("7");
  expect(within(dialog).getByRole("cell", { name: "0.00000012345" })).toBeVisible();
  expect(within(dialog).queryByRole("table", { name: "Objects spreadsheet" })).not.toBeInTheDocument();
  await userEvent.keyboard("{End}");
  expect(within(dialog).getByRole("tab", { name: "Settings" })).toHaveFocus();
  expect(within(dialog).getByText("Random seed")).toBeVisible();
  expect(within(dialog).getByText("42")).toBeVisible();
  await userEvent.tab();
  expect(within(dialog).getByRole("link", { name: "Download settings" })).toHaveFocus();
  await userEvent.tab();
  expect(within(dialog).getByRole("button", { name: "Close results" })).toHaveFocus();
  await userEvent.tab({ shift: true });
  expect(within(dialog).getByRole("link", { name: "Download settings" })).toHaveFocus();
  await userEvent.keyboard("{Escape}");
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(results).toHaveFocus();
});

it("opens older individual results and exposes their actual saved exports", async () => {
  vi.mocked(getAnalysisHistory).mockResolvedValue({ results: [{ ...entry, id: "run", kind: "run", images: [entry.images[0]] }], count: 1, offset: 0, limit: 25, next_offset: null });
  vi.mocked(getAnalysisRun).mockResolvedValue({ id: "run", params: { seed: 19 }, exports: ["objects.csv", "manifest.json"], status: "SUCCESS" } as unknown as AnalysisRun);
  render(<MemoryRouter><AnalysisHistory experimentId="exp" images={[]} /></MemoryRouter>);
  await userEvent.click(await screen.findByRole("button", { name: /Donor comparison/ }));
  expect(await screen.findByText("Saved measurements")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "objects.csv" })).toHaveAttribute("href", "/api/analysis/run/export/objects.csv");
});
