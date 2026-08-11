import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  LibrarySelectionBar,
  datasetsLostBy,
} from "@/features/library/components/grouping/LibrarySelectionBar";
import { assignAssetGrouping } from "@/shared/api/assets";
import type { Experiment } from "@/shared/types/common";
import type { HomeEntry } from "@/shared/types/images";

vi.mock("@/shared/api/assets", async () => {
  const actual =
    await vi.importActual<typeof import("@/shared/api/assets")>(
      "@/shared/api/assets"
    );
  return { ...actual, assignAssetGrouping: vi.fn() };
});

function entry(overrides: Partial<HomeEntry> = {}): HomeEntry {
  return {
    id: "asset-1",
    display_name: "Scan",
    original_filename: "scan.tif",
    metadata_summary: "1024x1024",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    preprocess_stage: "DONE",
    preprocess_progress: 1,
    can_open: true,
    ...overrides,
  };
}

const FASTED: Experiment = {
  id: "exp-1",
  name: "Fasted cohort",
  notes: "",
  datasets: [
    {
      id: "set-1",
      experiment: "exp-1",
      name: "Liver 24h",
      notes: "",
      asset_count: 1,
      created_at: null,
      updated_at: null,
    },
  ],
  asset_count: 1,
  ungrouped_asset_count: 0,
  created_at: null,
  updated_at: null,
};

const FED: Experiment = {
  id: "exp-2",
  name: "Fed cohort",
  notes: "",
  datasets: [],
  asset_count: 0,
  ungrouped_asset_count: 0,
  created_at: null,
  updated_at: null,
};

/**
 * The cost of a move, counted before it is paid.
 *
 * A dataset belongs to exactly one experiment, so moving images out of theirs
 * takes them out of its datasets. That is information the user did not ask to
 * lose, so it has to be on screen next to the button rather than in the reply.
 */
describe("datasetsLostBy", () => {
  const filed = entry({
    id: "filed",
    experiment_id: "exp-1",
    dataset_ids: ["set-1"],
    dataset_names: ["Liver 24h"],
  });
  const loose = entry({ id: "loose" });

  it("counts nothing when the experiment is being left alone", () => {
    expect(datasetsLostBy([filed, loose], { kind: "keep" })).toBe(0);
  });

  it("counts nothing when the images are staying where they are", () => {
    expect(
      datasetsLostBy([filed], { kind: "existing", id: "exp-1" })
    ).toBe(0);
  });

  it("counts the images that would leave a dataset on a move", () => {
    expect(
      datasetsLostBy([filed, loose], { kind: "existing", id: "exp-2" })
    ).toBe(1);
  });

  it("counts them for a brand new experiment too", () => {
    expect(datasetsLostBy([filed], { kind: "new", name: "Starved" })).toBe(1);
  });

  it("counts them when the experiment is being cleared", () => {
    expect(datasetsLostBy([filed], { kind: "none" })).toBe(1);
  });
});

describe("LibrarySelectionBar", () => {
  beforeEach(() => {
    vi.mocked(assignAssetGrouping).mockResolvedValue({
      assets_changed: 1,
      dataset_links_dropped: 0,
      assets_moved_out_of_datasets: 0,
      datasets_left: [],
      experiment: FASTED,
      datasets: [],
    });
  });

  function renderBar(selected: HomeEntry[] = [entry()]) {
    return render(
      <LibrarySelectionBar
        selected={selected}
        experiments={[FASTED, FED]}
        onApplied={vi.fn()}
        onClearSelection={vi.fn()}
        onSelectAllShown={vi.fn()}
        shownCount={selected.length}
      />
    );
  }

  /**
   * The default must be a no-op. A bar that opens on "No experiment" would
   * unassign a whole selection the first time someone pressed the only enabled
   * button on it.
   */
  it("does nothing until something is actually chosen", () => {
    renderBar();

    expect(screen.getByRole("button", { name: /Apply to/ })).toBeDisabled();
  });

  it("files a selection into an existing experiment", async () => {
    const user = userEvent.setup();
    renderBar([entry({ id: "a" }), entry({ id: "b" })]);

    await user.selectOptions(screen.getByLabelText("Experiment"), "exp-1");
    await user.click(screen.getByRole("button", { name: "Apply to 2 images" }));

    await waitFor(() =>
      expect(assignAssetGrouping).toHaveBeenCalledWith({
        asset_ids: ["a", "b"],
        experiment: "exp-1",
      })
    );
  });

  it("creates an experiment from a typed name", async () => {
    const user = userEvent.setup();
    renderBar();

    await user.selectOptions(screen.getByLabelText("Experiment"), "__new__");
    await user.type(
      screen.getByLabelText("New experiment…"),
      "Starved cohort"
    );
    await user.click(screen.getByRole("button", { name: /Apply to/ }));

    await waitFor(() =>
      expect(assignAssetGrouping).toHaveBeenCalledWith({
        asset_ids: ["asset-1"],
        experiment_name: "Starved cohort",
      })
    );
  });

  /** Omitting the key is what "leave the experiment alone" means to the server. */
  it("sends no experiment when only the dataset is being changed", async () => {
    const user = userEvent.setup();
    renderBar([
      entry({ id: "a", experiment_id: "exp-1", experiment_name: "Fasted cohort" }),
    ]);

    await user.selectOptions(screen.getByLabelText("Dataset"), "set-1");
    await user.click(screen.getByRole("button", { name: /Apply to/ }));

    await waitFor(() =>
      expect(assignAssetGrouping).toHaveBeenCalledWith({
        asset_ids: ["a"],
        datasets: ["set-1"],
      })
    );
  });

  it("warns before a move that would empty a dataset", async () => {
    const user = userEvent.setup();
    renderBar([
      entry({
        id: "filed",
        experiment_id: "exp-1",
        dataset_ids: ["set-1"],
        dataset_names: ["Liver 24h"],
      }),
    ]);

    await user.selectOptions(screen.getByLabelText("Experiment"), "exp-2");

    expect(
      screen.getByText(/in a dataset of another experiment/)
    ).toBeInTheDocument();
  });

  it("reports what the move actually cost once it lands", async () => {
    vi.mocked(assignAssetGrouping).mockResolvedValue({
      assets_changed: 1,
      dataset_links_dropped: 1,
      assets_moved_out_of_datasets: 1,
      datasets_left: ["Liver 24h"],
      experiment: FED,
      datasets: [],
    });
    const user = userEvent.setup();
    renderBar([
      entry({
        id: "filed",
        experiment_id: "exp-1",
        dataset_ids: ["set-1"],
        dataset_names: ["Liver 24h"],
      }),
    ]);

    await user.selectOptions(screen.getByLabelText("Experiment"), "exp-2");
    await user.click(screen.getByRole("button", { name: /Apply to/ }));

    expect(
      await screen.findByText(/1 image left Liver 24h/)
    ).toBeInTheDocument();
  });

  it("says what went wrong instead of failing silently", async () => {
    vi.mocked(assignAssetGrouping).mockRejectedValue(
      new Error(JSON.stringify({ detail: "That experiment is no longer in the library." }))
    );
    const user = userEvent.setup();
    renderBar();

    await user.selectOptions(screen.getByLabelText("Experiment"), "exp-1");
    await user.click(screen.getByRole("button", { name: /Apply to/ }));

    expect(
      await screen.findByText("That experiment is no longer in the library.")
    ).toBeInTheDocument();
  });
});
