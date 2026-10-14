import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, expect, it, vi } from "vitest";
import { WorkspaceBar } from "./WorkspaceBar";
import { getAsset, getExperiment } from "@/shared/api/assets";
import { getNavigationImages, recordVisit, pinExperiment } from "@/shared/api/workspace";
import { getJobQueueStatus } from "@/shared/api/jobs";
import { rememberWorkspace } from "@/features/experiments/workspaceMemory";
import { RecentExperiments } from "@/features/experiments/RecentExperiments";
import type { Experiment } from "@/shared/types/common";
import type { AssetDetail } from "@/shared/types/images";

vi.mock("@/shared/api/assets", () => ({ getAsset: vi.fn(), getExperiment: vi.fn(), getAssetSegmentations: vi.fn() }));
vi.mock("@/shared/api/workspace", () => ({ getNavigationImages: vi.fn(), recordVisit: vi.fn(), pinExperiment: vi.fn() }));
vi.mock("@/shared/api/jobs", () => ({ getJobQueueStatus: vi.fn() }));
vi.mock("@/features/finetune/FineTuneDialog", () => ({
  FineTuneDialog: (props: { open: boolean; initialExperimentId?: string; initialAssetIds?: string[] }) => props.open
    ? <div role="dialog">{props.initialExperimentId} · {props.initialAssetIds?.join(",")}</div> : null,
}));
vi.mock("@/features/library/components/JobQueueSidebar", () => ({ JobQueueSidebar: () => null }));
vi.mock("@/features/experiments/CorrectionHistory", () => ({ CorrectionHistory: () => null }));
const experiment: Experiment = { id: "experiment", name: "Nanotomy T1D", datasets: [], notes: "",
  asset_count: 3, ungrouped_asset_count: 3, created_at: null, updated_at: null };

beforeEach(() => {
  vi.clearAllMocks();
  window.sessionStorage.clear();
  vi.mocked(getAsset).mockResolvedValue({ id: "image-2", display_name: "Field 2", experiment_id: "experiment" } as AssetDetail);
  vi.mocked(getExperiment).mockResolvedValue(experiment);
  vi.mocked(getNavigationImages).mockResolvedValue({ images: [{ id: "image-1", name: "Field 1" }, { id: "image-2", name: "Field 2" }, { id: "image-3", name: "Field 3" }] });
  vi.mocked(recordVisit).mockResolvedValue({});
  vi.mocked(getJobQueueStatus).mockResolvedValue({ running: [], queues: [], failed: [] } as unknown as Awaited<ReturnType<typeof getJobQueueStatus>>);
});

it("offers contextual fine-tuning and follows the filtered image order without leaving Analysis", async () => {
  rememberWorkspace("experiment.imageOrder", ["image-3", "image-2", "image-1"]);
  render(<MemoryRouter initialEntries={["/assets/image-2/analysis"]}><WorkspaceBar /></MemoryRouter>);
  await screen.findByRole("link", { name: "Nanotomy T1D" });
  expect(screen.getByRole("link", { name: "Previous image" })).toHaveAttribute("href", "/assets/image-3/analysis");
  expect(screen.getByRole("link", { name: "Next image" })).toHaveAttribute("href", "/assets/image-1/analysis");
  expect(screen.getByRole("link", { name: "Models" })).toHaveAttribute("href", "/experiments/experiment?tab=models");
  await userEvent.click(screen.getByRole("button", { name: "Fine-tune" }));
  expect(screen.getByRole("dialog")).toHaveTextContent("experiment · image-2");
  await waitFor(() => expect(recordVisit).toHaveBeenCalledWith("experiment", "/assets/image-2/analysis"));
});

it("puts pinned experiments before recent ones and resumes their last page", async () => {
  vi.mocked(pinExperiment).mockResolvedValue({ ...experiment, pinned: true });
  const changed = vi.fn();
  render(<MemoryRouter><RecentExperiments experiments={[
    { ...experiment, id: "recent", name: "Recent", last_opened_at: "2026-10-13", last_route: "/assets/recent/analysis" },
    { ...experiment, id: "pinned", name: "Pinned", pinned: true, last_opened_at: "2026-10-01" },
  ]} onChange={changed} /></MemoryRouter>);
  const links = screen.getAllByRole("link");
  expect(links.map((link) => link.textContent)).toEqual(["Pinned", "Recent"]);
  expect(links[1]).toHaveAttribute("href", "/assets/recent/analysis");
  await userEvent.click(screen.getByRole("button", { name: "Pin Recent" }));
  await waitFor(() => expect(pinExperiment).toHaveBeenCalledWith("recent", true));
  expect(changed).toHaveBeenCalled();
});
