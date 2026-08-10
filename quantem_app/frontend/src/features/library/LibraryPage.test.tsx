import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LibraryPage } from "@/features/library/LibraryPage";
import { getHomeEntryPage } from "@/shared/api/assets";
import { getSystemStatus } from "@/shared/api/jobs";
import type { HomeEntryPage } from "@/shared/types/images";

vi.mock("react-router-dom", async () => {
  const actual =
    await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return { ...actual, useNavigate: () => vi.fn() };
});

vi.mock("@/shared/api/assets", async () => {
  const actual =
    await vi.importActual<typeof import("@/shared/api/assets")>(
      "@/shared/api/assets"
    );
  return {
    ...actual,
    getHomeEntryPage: vi.fn(),
    getAsset: vi.fn(),
    deleteAsset: vi.fn(),
  };
});

vi.mock("@/shared/api/jobs", async () => {
  const actual =
    await vi.importActual<typeof import("@/shared/api/jobs")>("@/shared/api/jobs");
  return { ...actual, getSystemStatus: vi.fn(), getJobQueueStatus: vi.fn() };
});

const EMPTY_PAGE: HomeEntryPage = {
  results: [],
  total: 0,
  limit: 60,
  offset: 0,
  has_more: false,
};

function renderLibrary() {
  return render(
    <MemoryRouter>
      <LibraryPage />
    </MemoryRouter>
  );
}

describe("LibraryPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    vi.mocked(getHomeEntryPage).mockResolvedValue(EMPTY_PAGE);
    vi.mocked(getSystemStatus).mockResolvedValue({
      cuda_available: false,
      supported_upload_formats: [".tif", ".tiff", ".png"],
    });
  });

  it("states CPU-only as a fact, not an amber warning", async () => {
    renderLibrary();

    // The README calls CPU-only fully supported and every released model runs
    // on it; an amber "CUDA unavailable" badge on the first screen reads as a
    // broken install.
    const badge = await screen.findByText("Running on CPU");
    expect(badge.className).not.toContain("amber");
    expect(badge).toHaveAttribute("title", expect.stringContaining("Everything works on CPU"));
    expect(screen.queryByText(/cuda unavailable/i)).toBeNull();
  });

  it("badges a real GPU as good news", async () => {
    vi.mocked(getSystemStatus).mockResolvedValue({
      cuda_available: true,
      supported_upload_formats: [".tif"],
    });
    renderLibrary();

    expect(await screen.findByText("GPU: CUDA")).toBeInTheDocument();
  });

  it("shows the workflow guide on a first visit and remembers it was dismissed", async () => {
    const user = userEvent.setup();
    const { unmount } = renderLibrary();

    expect(await screen.findByText("How QuantEM works")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Got it" }));
    expect(screen.queryByText("How QuantEM works")).toBeNull();

    unmount();
    renderLibrary();
    await screen.findByRole("heading", { name: "Library" });
    expect(screen.queryByText("How QuantEM works")).toBeNull();
  });

  it("can reopen the guide from the header", async () => {
    window.localStorage.setItem("quantem-workflow-guide-dismissed-v1", "1");
    const user = userEvent.setup();
    renderLibrary();

    await user.click(await screen.findByRole("button", { name: "How this works" }));
    expect(screen.getByText("How QuantEM works")).toBeInTheDocument();
  });

  it("opens the import panel from the empty state instead of pointing at a collapsed one", async () => {
    window.localStorage.setItem("quantem-workflow-guide-dismissed-v1", "1");
    const user = userEvent.setup();
    renderLibrary();

    // The old copy said "Import an image above" while the panel was collapsed.
    const importButton = await screen.findByRole("button", { name: "Import an image" });
    expect(screen.queryByLabelText(/image file/i)).toBeNull();

    await user.click(importButton);

    await waitFor(() => {
      expect(screen.getByLabelText(/image file/i)).toBeInTheDocument();
    });
  });
});
