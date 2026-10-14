import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, expect, it, vi } from "vitest";
import { AnalysisHistory } from "./AnalysisHistory";
import { getAnalysisBatch, getAnalysisHistory, type AnalysisHistoryEntry } from "@/shared/api/analysisHistory";
import { getAnalysisRun } from "@/shared/api/analysis";
import type { AnalysisRun } from "@/shared/types/analysis";

vi.mock("@/shared/api/analysisHistory", () => ({ getAnalysisHistory: vi.fn(), getAnalysisBatch: vi.fn() }));
vi.mock("@/shared/api/analysis", () => ({ getAnalysisRun: vi.fn(), getAnalysisExportUrl: (id: string, name: string) => `/api/analysis/${id}/export/${name}` }));
vi.mock("@/shared/api/core/http", () => ({ resolveApiUrl: (url: string) => url }));
vi.mock("@/features/analysis/components/AnalysisResults", () => ({ AnalysisResults: () => <div>Saved measurements</div> }));

const entry: AnalysisHistoryEntry = { id: "batch", kind: "batch", name: "Donor comparison", created_at: "2026-10-14T12:00:00Z", status: "SUCCESS", images: [{ id: "a", name: "Field A" }, { id: "b", name: "Field B" }], structures: ["Mitos"], measurement_count: 2, measurement_area: "Whole image", completed_image_ids: ["a", "b"] };
beforeEach(() => vi.clearAllMocks());

it("filters history by image but opens every image and saved setting in the original batch", async () => {
  vi.mocked(getAnalysisHistory).mockResolvedValue([entry]);
  vi.mocked(getAnalysisBatch).mockResolvedValue({ id: "batch", name: entry.name, created_at: entry.created_at, settings: { require_complete: true, reuse: false }, items: entry.images.map((image) => ({ identity: { image: image.name, asset_id: image.id, organelle: "mitochondria", sample_id: image.name, group: "", pixel_size_nm: 2.5, measurement_area: "Whole image" }, status: "SUCCESS", reason: "", params: { seed: 42 }, metrics: { n_objects: 7 }, run_id: `run-${image.id}` })) });
  render(<MemoryRouter initialEntries={["/experiments/exp?tab=analysis&image=a"]}><AnalysisHistory experimentId="exp" images={[]} /></MemoryRouter>);
  const row = await screen.findByRole("button", { name: /Donor comparison/ });
  expect(getAnalysisHistory).toHaveBeenCalledWith("exp", "a");
  expect(getAnalysisBatch).not.toHaveBeenCalled();
  await userEvent.click(row);
  expect(await screen.findByRole("cell", { name: "Field B", exact: true })).toBeInTheDocument();
  expect(screen.getByRole("cell", { name: "Field A", exact: true })).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Download", exact: true })).toHaveAttribute("href", "/api/analysis-batches/batch/export/?objects=true");
  await userEvent.click(screen.getByText("Analysis settings"));
  expect(screen.getByText("Require completed review")).toBeVisible();
  expect(screen.getByText("Yes")).toBeVisible();
});

it("opens older individual results and exposes their actual saved exports", async () => {
  vi.mocked(getAnalysisHistory).mockResolvedValue([{ ...entry, id: "run", kind: "run", images: [entry.images[0]] }]);
  vi.mocked(getAnalysisRun).mockResolvedValue({ id: "run", params: { seed: 19 }, exports: ["objects.csv", "manifest.json"], status: "SUCCESS" } as unknown as AnalysisRun);
  render(<MemoryRouter><AnalysisHistory experimentId="exp" images={[]} /></MemoryRouter>);
  await userEvent.click(await screen.findByRole("button", { name: /Donor comparison/ }));
  expect(await screen.findByText("Saved measurements")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "objects.csv" })).toHaveAttribute("href", "/api/analysis/run/export/objects.csv");
});
