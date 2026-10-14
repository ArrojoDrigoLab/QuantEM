import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import { getAnalysisTable, type AnalysisTablePage } from "@/shared/api/analysis";
import type { AnalysisRun } from "@/shared/types/analysis";
import { AnalysisTablePreview } from "./AnalysisTablePreview";

vi.mock("@/shared/api/analysis", () => ({
  getAnalysisTable: vi.fn(),
  getAnalysisExportUrl: (id: string, name: string) => `/api/analysis/${id}/export/${name}`,
}));

const run = { id: "run", exports: ["manifest.json", "image_summary.csv", "objects.csv", "composition.csv"] } as AnalysisRun;
const page: AnalysisTablePage = { name: "objects.csv", columns: ["object_id", "area_um2"], rows: [["9007199254740993", "0.00000012345"]], offset: 0, limit: 100, has_more: true };
beforeEach(() => vi.resetAllMocks());

it("keeps spreadsheet values intact, pages rows, and starts each table at its first row", async () => {
  vi.mocked(getAnalysisTable).mockImplementation(async (_id, name, offset = 0) => ({
    ...page, name, offset, rows: offset ? [["next-object", "1.2345678901234567"]] : page.rows,
    has_more: name === "objects.csv" && offset === 0,
  }));
  render(<AnalysisTablePreview run={run} />);
  const table = await screen.findByRole("table", { name: "Objects spreadsheet" });
  expect(within(table).getByRole("cell", { name: "9007199254740993" })).toBeVisible();
  expect(within(table).getByRole("cell", { name: "0.00000012345" })).toBeVisible();
  expect(screen.getByRole("button", { name: "Previous rows" })).toBeDisabled();
  await userEvent.click(screen.getByRole("button", { name: "Next rows" }));
  expect(await screen.findByRole("cell", { name: "next-object" })).toBeVisible();
  expect(getAnalysisTable).toHaveBeenLastCalledWith("run", "objects.csv", 100, expect.any(AbortSignal));
  expect(screen.getByText("Rows 101–101")).toBeVisible();
  expect(screen.getByRole("button", { name: "Next rows" })).toBeDisabled();
  await userEvent.selectOptions(screen.getByRole("combobox", { name: "Table" }), "composition.csv");
  expect(await screen.findByRole("table", { name: "Composition spreadsheet" })).toBeVisible();
  expect(getAnalysisTable).toHaveBeenLastCalledWith("run", "composition.csv", 0, expect.any(AbortSignal));
  expect(screen.getByRole("link", { name: "Download" })).toHaveAttribute("href", "/api/analysis/run/export/composition.csv");
  expect(screen.getByRole("button", { name: "Previous rows" })).toBeDisabled();
});

it("shows saved area measurements for whole-image segmentation results", async () => {
  vi.mocked(getAnalysisTable).mockResolvedValue({ ...page, name: "area.csv", columns: ["foreground_percent"], rows: [["17.25"]], has_more: false });
  render(<AnalysisTablePreview run={{ ...run, exports: ["area.csv", "manifest.json"] }} />);
  expect(await screen.findByRole("table", { name: "Area spreadsheet" })).toHaveTextContent("17.25");
  expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
});

it("shows empty tables without treating them as unavailable", async () => {
  vi.mocked(getAnalysisTable).mockResolvedValue({ ...page, rows: [], has_more: false });
  render(<AnalysisTablePreview run={run} />);
  expect(await screen.findByText("No rows in this table.")).toBeVisible();
  expect(screen.getByRole("columnheader", { name: "area_um2" })).toBeVisible();
  expect(screen.getByRole("button", { name: "Next rows" })).toBeDisabled();
});

it("does not request a spreadsheet when its export is unavailable", () => {
  render(<AnalysisTablePreview run={{ ...run, exports: ["manifest.json"] }} />);
  expect(screen.getByText("No spreadsheet is available for this image.")).toBeVisible();
  expect(getAnalysisTable).not.toHaveBeenCalled();
});

it("can retry a failed table request", async () => {
  vi.mocked(getAnalysisTable).mockRejectedValueOnce(new Error("The saved table could not be read.")).mockResolvedValueOnce(page);
  render(<AnalysisTablePreview run={run} />);
  expect(await screen.findByRole("alert")).toHaveTextContent("The saved table could not be read.");
  await userEvent.click(screen.getByRole("button", { name: "Retry" }));
  await waitFor(() => expect(screen.getByRole("table", { name: "Objects spreadsheet" })).toBeVisible());
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});
