import {
  render as testingRender,
  screen,
  type RenderOptions,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import { ImageCard } from "@/features/library/components/ImageCard";
import type { HomeEntry } from "@/shared/types/images";

function render(ui: ReactElement, options: RenderOptions = {}) {
  return testingRender(ui, { wrapper: MemoryRouter, ...options });
}

function makeEntry(overrides: Partial<HomeEntry> = {}): HomeEntry {
  return {
    id: "11111111-2222-3333-4444-555555555555",
    display_name: "Liver 01",
    original_filename: "liver01.tif",
    notes: "A representative liver image with a deliberately longer note.",
    metadata_summary: "1024x1024",
    width: 1024,
    height: 1024,
    pixel_size_nm: 4.2,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-02T00:00:00Z",
    preprocess_stage: "DONE",
    preprocess_progress: 100,
    ngff_ready: true,
    can_open: true,
    ...overrides,
  };
}

describe("ImageCard", () => {
  it("opens the viewer from either the title or thumbnail", async () => {
    const user = userEvent.setup();
    render(
      <Routes>
        <Route path="/" element={<ImageCard image={makeEntry()} />} />
        <Route
          path="/assets/:assetId/viewer"
          element={<p>Viewer route rendered</p>}
        />
      </Routes>
    );

    const links = screen.getAllByRole("link");
    expect(links).toHaveLength(2);
    expect(links[0]).toHaveAttribute(
      "href",
      "/assets/11111111-2222-3333-4444-555555555555/viewer"
    );
    await user.click(screen.getByRole("link", { name: "Open Liver 01" }));
    expect(screen.getByText("Viewer route rendered")).toBeInTheDocument();
  });

  it("shows notes, dimensions and only the scale value", () => {
    render(<ImageCard image={makeEntry()} />);

    const notes = screen.getByText(/representative liver image/i);
    expect(notes.className).toContain("line-clamp-2");
    expect(notes).toHaveAttribute(
      "title",
      "A representative liver image with a deliberately longer note."
    );
    expect(screen.getByText("1024 x 1024")).toBeInTheDocument();
    expect(screen.getByText("4.2 nm/px")).toBeInTheDocument();
    expect(screen.queryByText(/entered by hand|from file/i)).not.toBeInTheDocument();
    expect(screen.queryByText("liver01.tif")).not.toBeInTheDocument();
    expect(screen.queryByText("Ready")).not.toBeInTheDocument();
  });

  it("shows a processing spinner or a failure tooltip instead of Ready", () => {
    const { rerender } = render(
      <ImageCard
        image={makeEntry({ preprocess_stage: "ENCODING", ngff_ready: false })}
      />
    );
    expect(screen.getByRole("status", { name: /Processing Liver 01/i })).toBeInTheDocument();

    rerender(
      <ImageCard
        image={makeEntry({
          preprocess_stage: "FAILED",
          ngff_ready: false,
          preprocess_error: "Encoding failed.",
        })}
      />
    );
    expect(screen.getByRole("button", { name: /Import failed/i })).toBeInTheDocument();
    expect(screen.getByText("Encoding failed.")).toBeInTheDocument();
    expect(
      screen.getByText("Delete this image and try to re-upload it.")
    ).toBeInTheDocument();
  });

  it("does not show a perpetual spinner for queued or skipped terminal states", () => {
    const { rerender } = render(
      <ImageCard
        image={makeEntry({ preprocess_stage: "NONE", ngff_ready: false })}
      />
    );
    expect(screen.queryByRole("status")).not.toBeInTheDocument();

    rerender(
      <ImageCard
        image={makeEntry({ preprocess_stage: "SKIPPED", ngff_ready: false })}
      />
    );
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("uses an overflow menu for edit, export, and delete on experiment pages", async () => {
    const user = userEvent.setup();
    const onEdit = vi.fn();
    const onExport = vi.fn();
    const onDelete = vi.fn();
    render(
      <ImageCard
        image={makeEntry()}
        onEdit={onEdit}
        onExport={onExport}
        onDelete={onDelete}
        useActionMenu
      />
    );

    expect(screen.queryByRole("button", { name: /Delete Liver/i })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Options for Liver/i }));
    await user.click(screen.getByRole("menuitem", { name: "Edit" }));
    expect(onEdit).toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: /Options for Liver/i }));
    await user.click(screen.getByRole("menuitem", { name: "Export" }));
    expect(onExport).toHaveBeenCalled();
  });

  it("closes the overflow menu with Escape", async () => {
    const user = userEvent.setup();
    render(
      <ImageCard
        image={makeEntry()}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        useActionMenu
      />
    );

    await user.click(screen.getByRole("button", { name: /Options for Liver/i }));
    expect(screen.getByRole("menu")).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });
});
