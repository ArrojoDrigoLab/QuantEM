import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SamBoxToolControls } from "@/features/sam/SamBoxToolControls";
import type { SamBoxTool } from "@/features/sam/useSamBoxTool";

function tool(
  bytesDone: number,
  percent: number | null
): SamBoxTool {
  return {
    isActive: true,
    setActive: vi.fn(),
    isSubmitting: false,
    pendingCount: 0,
    overlays: [],
    handleImagePress: vi.fn(),
    handleImageDrag: vi.fn(),
    handleImageRelease: vi.fn(),
    cancelDrag: vi.fn(),
    model: {
      model: "micro-SAM EM organelles (ViT-B)",
      installed: false,
      size_bytes: 1_000_000_000,
      download: {
        status: "RUNNING",
        bytes_done: bytesDone,
        bytes_total: 1_000_000_000,
        error: "",
        percent,
      },
    },
    modelReady: false,
    downloadModel: vi.fn(),
    lastTiming: null,
  };
}

describe("SamBoxToolControls", () => {
  it("updates download values without replacing the progress control", () => {
    const { rerender } = render(
      <SamBoxToolControls tool={tool(10_000_000, 1)} selected onToggle={vi.fn()} />
    );
    const progress = screen.getByRole("progressbar");
    expect(progress).toHaveClass("sam-box-tool-progress");
    expect(progress).toHaveAttribute("value", "10000000");

    rerender(
      <SamBoxToolControls tool={tool(650_000_000, 65)} selected onToggle={vi.fn()} />
    );

    expect(screen.getByRole("progressbar")).toBe(progress);
    expect(progress).toHaveAttribute("value", "650000000");
    expect(screen.getByText(/Downloading 650 MB of 1000 MB \(65%\)/)).toBeInTheDocument();
  });
});
