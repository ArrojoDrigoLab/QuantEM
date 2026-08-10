import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  RunHistory,
  type RunHistoryRow,
} from "@/features/analysis/components/RunHistory";
import { ApiRequestError } from "@/shared/api/core/http";

/**
 * `displayStatus` is what the badge renders, and it is not always `status`.
 *
 * The row is only written when the worker moves the run on, so a run mid-write
 * showed PENDING here beside a panel reading "writing export bundle".
 * `reconcileRunHistory` supplies it; this component just renders it, and the
 * default below keeps the two the same for every test that is not about the
 * difference.
 */
function makeRun(overrides: Partial<RunHistoryRow> = {}): RunHistoryRow {
  const status = overrides.status ?? "SUCCESS";
  return {
    id: "run-1",
    status,
    displayStatus: status,
    group: "fasted",
    created_at: "2026-02-01T10:00:00Z",
    started_at: "2026-02-01T10:00:01Z",
    finished_at: "2026-02-01T10:00:09Z",
    export_dir: "/data/analysis/run-1",
    error: "",
    calibrated: true,
    n_objects: 214,
    n_caveats: 0,
    ...overrides,
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("RunHistory", () => {
  it("says nothing has been analysed when the list is genuinely empty", () => {
    render(
      <RunHistory runs={[]} selectedRunId={null} onSelect={vi.fn()} loading={false} />
    );

    expect(screen.getByText("0 runs")).toBeInTheDocument();
    expect(
      screen.getByText("Nothing has been analysed for this segmentation yet.")
    ).toBeInTheDocument();
  });

  it("does not render a dead endpoint as an empty history", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    render(
      <RunHistory
        runs={[]}
        selectedRunId={null}
        onSelect={vi.fn()}
        loading={false}
        error={new ApiRequestError("<!DOCTYPE html><h1>Not Found</h1>", { status: 404 })}
      />
    );

    // "0 runs -- nothing has been analysed yet" in response to a 404 claims the
    // user's previous runs do not exist. It must not be reachable that way.
    expect(screen.queryByText("0 runs")).toBeNull();
    expect(
      screen.queryByText("Nothing has been analysed for this segmentation yet.")
    ).toBeNull();
    expect(
      screen.getByText("The run-history endpoint did not answer.")
    ).toBeInTheDocument();
    expect(screen.getByText("unavailable")).toBeInTheDocument();
  });

  it("distinguishes a non-404 failure from a missing route", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    render(
      <RunHistory
        runs={[]}
        selectedRunId={null}
        onSelect={vi.fn()}
        loading={false}
        error={new ApiRequestError(JSON.stringify({ error: "Database is locked." }), {
          status: 500,
        })}
      />
    );

    expect(screen.getByText("Run history could not be loaded.")).toBeInTheDocument();
    expect(screen.getByText("Database is locked.")).toBeInTheDocument();
  });

  it("lists runs with their caveat count and calibration state", () => {
    render(
      <RunHistory
        runs={[makeRun({ calibrated: false, n_caveats: 2 })]}
        selectedRunId="run-1"
        onSelect={vi.fn()}
        loading={false}
      />
    );

    expect(screen.getByText("1 runs")).toBeInTheDocument();
    expect(screen.getByText("uncalibrated")).toBeInTheDocument();
    expect(screen.getByText("2 caveats")).toBeInTheDocument();
  });

  /**
   * The badge and the panel beside it are one run. They used to be two claims:
   * the panel read the live job, the row read whatever the worker had last
   * written, and a run mid-write showed `PENDING` next to "writing export
   * bundle". The permanent-looking one was the wrong one.
   */
  it("renders the reconciled status, not the row the worker last wrote", () => {
    render(
      <RunHistory
        runs={[makeRun({ status: "PENDING", displayStatus: "RUNNING" })]}
        selectedRunId="run-1"
        onSelect={vi.fn()}
        loading={false}
      />
    );

    expect(screen.getByText("RUNNING")).toBeInTheDocument();
    expect(screen.queryByText("PENDING")).not.toBeInTheDocument();
  });

  it("names a cancellation rather than calling it a failure", () => {
    render(
      <RunHistory
        runs={[makeRun({ status: "FAILED", displayStatus: "CANCELLED" })]}
        selectedRunId="run-1"
        onSelect={vi.fn()}
        loading={false}
      />
    );

    // The same word the Adapt wizard uses for the same click.
    expect(screen.getByText("CANCELLED")).toBeInTheDocument();
    expect(screen.queryByText("FAILED")).not.toBeInTheDocument();
  });
});
