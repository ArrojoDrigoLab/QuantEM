import { useState } from "react";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { CorrectionHistory } from "./CorrectionHistory";

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
