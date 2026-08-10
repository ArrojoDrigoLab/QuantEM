import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CompletedRoiBackgroundNotice } from "@/features/segmentation/components/CompletedRoiBackgroundNotice";
import { isBackgroundCountWarning } from "@/features/segmentation/screen/hooks/useCompletedRoiWorkflow";

describe("CompletedRoiBackgroundNotice", () => {
  it("names the number of candidates about to become background", () => {
    // The gap: the dialog stated the rule -- everything unconfirmed inside
    // becomes background -- while 88 candidates were about to be trained
    // against as negatives, and nothing counted them.
    render(
      <CompletedRoiBackgroundNotice pending={{ status: "ready", count: 88 }} />
    );

    expect(screen.getByText("88 unconfirmed candidates")).toBeInTheDocument();
    expect(screen.getByText(/negative examples/)).toBeInTheDocument();
  });

  it("uses the singular for one candidate", () => {
    render(
      <CompletedRoiBackgroundNotice pending={{ status: "ready", count: 1 }} />
    );

    expect(screen.getByText("1 unconfirmed candidate")).toBeInTheDocument();
    expect(screen.getByText(/sits inside this area/)).toBeInTheDocument();
  });

  it("says plainly when nothing is affected", () => {
    render(
      <CompletedRoiBackgroundNotice pending={{ status: "ready", count: 0 }} />
    );

    expect(screen.getByText("No unconfirmed candidates")).toBeInTheDocument();
  });

  it("admits it could not count rather than implying zero", () => {
    render(<CompletedRoiBackgroundNotice pending={{ status: "error", count: null }} />);

    expect(screen.getByText(/could not be counted/)).toBeInTheDocument();
    expect(screen.queryByText(/^0 /)).not.toBeInTheDocument();
  });

  it("shows a counting state instead of a stale number", () => {
    render(<CompletedRoiBackgroundNotice pending={{ status: "loading", count: null }} />);

    expect(screen.getByText(/Counting the candidates/)).toBeInTheDocument();
  });
});

describe("isBackgroundCountWarning", () => {
  it("warns only for a known non-zero count", () => {
    expect(isBackgroundCountWarning({ status: "ready", count: 88 })).toBe(true);
    expect(isBackgroundCountWarning({ status: "ready", count: 0 })).toBe(false);
    expect(isBackgroundCountWarning({ status: "loading", count: null })).toBe(false);
    expect(isBackgroundCountWarning({ status: "error", count: null })).toBe(false);
  });
});
