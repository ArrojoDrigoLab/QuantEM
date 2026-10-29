import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { expect, it, vi } from "vitest";
import { apiRequest } from "@/shared/api/core/http";
import { BulkModelPanel } from "./BulkModelPanel";

vi.mock("@/shared/api/core/http", async (importOriginal) => ({ ...await importOriginal<typeof import("@/shared/api/core/http")>(), apiRequest: vi.fn() }));

it("checks a shared selection and queues ready images without annotations or unlock steps", async () => {
  const items = [{ asset_id: "a", image: "Field A", ready: true, reason: "", tiles: 100 }, { asset_id: "b", image: "Field B", ready: false, reason: "Preparing image" }];
  vi.mocked(apiRequest).mockImplementation(async (_url, options) => {
    if (_url === "/api/model-runs/batch/") return { id: "batch", created_at: "2026-10-27", items: items.map((row) => ({ ...row, status: row.ready ? "PENDING" : "SKIPPED" })) };
    if (!options?.method) return [];
    const data = JSON.parse(options.body as string);
    return data.preview ? { items } : { id: "batch", created_at: "2026-10-27", items: items.map((row) => ({ ...row, status: row.ready ? "PENDING" : "SKIPPED" })) };
  });
  const user = userEvent.setup();
  render(<MemoryRouter><BulkModelPanel experimentId="exp" selectedIds={["a", "b"]} onChanged={vi.fn()} /></MemoryRouter>);
  expect(screen.getByRole("button", { name: "Apply to ready images" })).toBeDisabled();
  await user.click(screen.getByRole("button", { name: "Check selected images" }));
  await screen.findByText("Preparing image");
  expect(screen.getByRole("button", { name: "Apply to ready images" })).toBeEnabled();
  await user.click(screen.getByRole("button", { name: "Apply to ready images" }));
  await waitFor(() => expect(screen.getByText("PENDING")).toBeInTheDocument());
  const submitted = vi.mocked(apiRequest).mock.calls.filter(([, options]) => options?.method === "POST").map(([, options]) => JSON.parse(options!.body as string));
  expect(submitted[1]).toEqual({ asset_ids: ["a", "b"], organelles: ["mito"], family: "omniem", source_models: { mito: "omniem:mito" }, threshold: null, preview: false });
  expect(screen.getByRole("button", { name: "Cancel remaining" })).toBeEnabled();
});

it("cancels queued and retrying jobs by removal, continues after failures, and cancels running jobs", async () => {
  const batch = { id: "cancel-batch", created_at: "2026-10-27", items: [
    { asset_id: "a", image: "A", job_id: "queued", status: "PENDING" },
    { asset_id: "b", image: "B", job_id: "retrying", status: "RETRY" },
    { asset_id: "c", image: "C", job_id: "running", status: "RUNNING" },
  ] };
  vi.mocked(apiRequest).mockReset();
  vi.mocked(apiRequest).mockImplementation(async (url, options) => {
    if (url === "/api/jobs/queued/" && options?.method === "DELETE") throw new Error("Removal failed");
    if (url === "/api/model-runs/cancel-batch/") return batch;
    if (url === "/api/experiments/cancel/model-runs/") return [batch];
    return [];
  });
  const user = userEvent.setup();
  render(<MemoryRouter><BulkModelPanel experimentId="cancel" selectedIds={["a", "b", "c"]} onChanged={vi.fn()} /></MemoryRouter>);
  await user.selectOptions(await screen.findByLabelText("Saved applications"), "cancel-batch");
  await user.click(screen.getByRole("button", { name: "Cancel remaining" }));
  await waitFor(() => expect(apiRequest).toHaveBeenCalledWith("/api/jobs/retrying/", { method: "DELETE" }));
  expect(apiRequest).toHaveBeenCalledWith("/api/jobs/running/cancel/", { method: "POST" });
  expect(await screen.findByRole("alert")).toHaveTextContent("Removal failed");
});

it("refreshes the image table on status changes, not on every progress update", async () => {
  const changed = vi.fn();
  let status = "RUNNING";
  let progress = 10;
  const batch = () => ({ id: "progress-batch", created_at: "2026-10-27", items: [{ asset_id: "a", image: "A", job_id: "job", status, progress }] });
  vi.mocked(apiRequest).mockImplementation(async (url) => url === "/api/model-runs/progress-batch/" ? batch()
    : url === "/api/experiments/progress/model-runs/" ? [batch()] : []);
  const user = userEvent.setup();
  render(<MemoryRouter><BulkModelPanel experimentId="progress" selectedIds={["a"]} onChanged={changed} /></MemoryRouter>);
  await user.selectOptions(await screen.findByLabelText("Saved applications"), "progress-batch");
  await waitFor(() => expect(changed).toHaveBeenCalledTimes(1));
  progress = 50;
  await act(async () => { window.dispatchEvent(new Event("focus")); });
  await screen.findByText("RUNNING · 50%");
  expect(changed).toHaveBeenCalledTimes(1);
  status = "SUCCESS";
  await act(async () => { window.dispatchEvent(new Event("focus")); });
  await waitFor(() => expect(changed).toHaveBeenCalledTimes(2));
});

it("restores the selected batch on reopen and does not carry it into another experiment", async () => {
  const batch = { id: "saved-batch", created_at: "2026-10-27", items: [{ asset_id: "a", image: "Old image", status: "SUCCESS" }] };
  vi.mocked(apiRequest).mockImplementation(async (url) => url === "/api/model-runs/saved-batch/" ? batch
    : url === "/api/experiments/first/model-runs/" ? [batch] : []);
  const user = userEvent.setup();
  const props = { experimentId: "first", selectedIds: ["a"], onChanged: vi.fn() };
  const view = render(<MemoryRouter><BulkModelPanel {...props} /></MemoryRouter>);
  await user.selectOptions(await screen.findByLabelText("Saved applications"), "saved-batch");
  view.unmount();
  const reopened = render(<MemoryRouter><BulkModelPanel {...props} /></MemoryRouter>);
  expect(await screen.findByRole("link", { name: "Old image" })).toBeInTheDocument();
  reopened.rerender(<MemoryRouter><BulkModelPanel {...props} experimentId="second" /></MemoryRouter>);
  expect(screen.queryByRole("link", { name: "Old image" })).not.toBeInTheDocument();
});

it("rejects out-of-range thresholds before previewing or queueing images", async () => {
  vi.mocked(apiRequest).mockResolvedValue([]);
  const user = userEvent.setup();
  render(<MemoryRouter><BulkModelPanel experimentId="invalid" selectedIds={["a"]} onChanged={vi.fn()} /></MemoryRouter>);
  await user.type(screen.getByLabelText("Threshold"), "1.5");
  expect(screen.getByRole("button", { name: "Check selected images" })).toBeDisabled();
  expect(screen.getByRole("alert")).toHaveTextContent("Threshold must be between 0 and 1.");
});
