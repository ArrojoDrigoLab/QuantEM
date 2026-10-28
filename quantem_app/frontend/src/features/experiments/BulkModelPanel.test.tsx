import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { expect, it, vi } from "vitest";
import { apiRequest } from "@/shared/api/core/http";
import { BulkModelPanel } from "./BulkModelPanel";

vi.mock("@/shared/api/core/http", () => ({ apiRequest: vi.fn() }));

it("checks a shared selection and queues ready images without annotations or unlock steps", async () => {
  const items = [{ asset_id: "a", image: "Field A", ready: true, reason: "", tiles: 100 }, { asset_id: "b", image: "Field B", ready: false, reason: "Preparing image" }];
  vi.mocked(apiRequest).mockImplementation(async (_url, options) => {
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
