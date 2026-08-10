import { useState } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";

describe("ConfirmDialog", () => {
  it("renders an accessible dialog and focuses the cancel button", () => {
    render(
      <ConfirmDialog
        isOpen
        title="Delete job"
        message="This cannot be undone."
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />
    );

    const dialog = screen.getByRole("dialog", { name: "Delete job" });
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(screen.getByRole("button", { name: "Cancel" })).toHaveFocus();
  });

  it("closes on Escape and restores focus", async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();
    function Harness() {
      const [isOpen, setIsOpen] = useState(true);

      return (
        <>
          <button type="button">Open confirm</button>
          <ConfirmDialog
            isOpen={isOpen}
            title="Delete job"
            message="This cannot be undone."
            onConfirm={vi.fn()}
            onCancel={() => {
              onCancel();
              setIsOpen(false);
            }}
          />
        </>
      );
    }

    render(<Harness />);

    const trigger = screen.getByRole("button", { name: "Open confirm" });
    trigger.focus();

    await user.keyboard("{Escape}");

    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(trigger).toHaveFocus();
  });
});
