import { useState } from "react";
import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { CorrectionHistory } from "./CorrectionHistory";
import { apiRequest } from "@/shared/api/core/http";
import { getAssetSegmentations } from "@/shared/api/assets";

vi.mock("@/shared/api/assets", () => ({ getAssetSegmentations: vi.fn().mockResolvedValue([
  { id: "seg", display_name: "Mitochondria" },
]) }));
vi.mock("@/shared/api/core/http", () => ({ apiRequest: vi.fn().mockResolvedValue([
  { id: "revision", name: "Before changes", created_at: "2026-10-14", object_count: 2 },
]) }));

function Example() {
  const [open, setOpen] = useState(false);
  return <><button onClick={() => setOpen(true)}>History</button>
    {open && <CorrectionHistory assetId="image" onClose={() => setOpen(false)} />}</>;
}

it("traps focus and closes only the top dialog, restoring focus at each level", async () => {
  render(<Example />);
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "History" }));
  const restore = await screen.findByRole("button", { name: "Restore…" });
  expect(screen.getByRole("button", { name: "Close" })).toHaveFocus();
  await user.tab({ shift: true });
  expect(restore).toHaveFocus();
  await user.tab();
  expect(screen.getByRole("button", { name: "Close" })).toHaveFocus();
  await user.click(restore);
  const confirmation = screen.getByRole("dialog", { name: "Restore correction revision?" });
  expect(within(confirmation).getByRole("button", { name: "Cancel" })).toHaveFocus();
  await user.tab({ shift: true });
  expect(within(confirmation).getByRole("button", { name: "Restore revision" })).toHaveFocus();
  await user.keyboard("{Escape}");
  expect(screen.queryByRole("dialog", { name: "Restore correction revision?" })).not.toBeInTheDocument();
  expect(screen.getByRole("dialog", { name: "Correction history" })).toBeInTheDocument();
  expect(restore).toHaveFocus();
  expect(document.body.style.overflow).toBe("hidden");
  await user.keyboard("{Escape}");
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "History" })).toHaveFocus();
  expect(document.body.style.overflow).not.toBe("hidden");
});

it("selects an available segmentation when the URL refers to a removed one", async () => {
  window.location.hash = "#?seg=removed";
  try {
    render(<CorrectionHistory assetId="image" onClose={() => undefined} />);
    await screen.findByRole("button", { name: "Restore…" });
    expect(screen.getByRole("combobox")).toHaveValue("seg");
    expect(apiRequest).not.toHaveBeenCalledWith("/api/segmentations/removed/revisions/");
  } finally {
    window.location.hash = "";
  }
});

it("ignores history selection from an image that is no longer open", async () => {
  let finishFirst!: (values: Awaited<ReturnType<typeof getAssetSegmentations>>) => void;
  vi.mocked(getAssetSegmentations).mockImplementationOnce(() => new Promise((resolve) => { finishFirst = resolve; }));
  vi.mocked(getAssetSegmentations).mockResolvedValueOnce([
    { id: "second", display_name: "Second image" } as Awaited<ReturnType<typeof getAssetSegmentations>>[number],
  ]);
  const view = render(<CorrectionHistory assetId="first" onClose={() => undefined} />);
  view.rerender(<CorrectionHistory assetId="second" onClose={() => undefined} />);
  await screen.findByRole("option", { name: "Second image" });
  await act(async () => { finishFirst([
    { id: "first", display_name: "First image" } as Awaited<ReturnType<typeof getAssetSegmentations>>[number],
  ]); });
  expect(screen.getByRole("combobox")).toHaveValue("second");
  expect(screen.queryByRole("option", { name: "First image" })).not.toBeInTheDocument();
});
