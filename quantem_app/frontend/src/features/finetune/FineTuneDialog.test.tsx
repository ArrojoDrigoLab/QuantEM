/**
 * R13, as five things that must be true on screen.
 *
 * 1. the tree renders and the count totals the owner's example to **7**;
 * 2. a selection the server calls ineligible cannot be submitted, and says why;
 * 3. all three modes are offered and the server's `default_mode` is the one
 *    preselected;
 * 4. the bar states steps, rounds and an ETA — and "estimating" where the
 *    server has no honest estimate yet;
 * 5. success **offers** to run the new model and queues nothing until asked.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";
import { FineTuneDialog } from "@/features/finetune/FineTuneDialog";
import { server } from "@/test/msw/server";
import type {
  FineTunePreviewResponse,
  FineTuneProgress,
  FineTuneScopeResponse,
} from "@/shared/types/finetune";
import type { SegmentationType } from "@/shared/types/images";

const API = "http://127.0.0.1:8000";

const MITO: SegmentationType = {
  id: "type-mito",
  internal_name: "quantem_internal_mito",
  short_name: "Mitochondria",
  long_name: "Mitochondria",
  default_color: null,
  tags: [],
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

function image(id: string, name: string, confirmed: number, done: number) {
  return {
    id,
    name,
    confirmed_areas: confirmed,
    done_rois: done,
    annotation_count: confirmed + done,
  };
}

/** The owner's example: ten images, three annotated, seven annotations. */
const SCOPE: FineTuneScopeResponse = {
  experiments: [
    {
      id: "exp-fasted",
      name: "Fasted cohort",
      datasets: [
        {
          id: "ds-liver",
          name: "Liver 24h",
          image_count: 10,
          annotated_image_count: 3,
          annotation_count: 7,
          images: [
            image("img-1", "liver_01.tif", 3, 0),
            image("img-2", "liver_02.tif", 2, 1),
            image("img-3", "liver_03.tif", 0, 1),
          ],
        },
      ],
      ungrouped_images: [],
    },
    {
      id: "exp-fed",
      name: "Fed cohort",
      datasets: [],
      ungrouped_images: [image("img-fed", "fed_01.tif", 2, 0)],
    },
  ],
  unassigned_images: [],
};

function preview(
  overrides: Partial<FineTunePreviewResponse> = {}
): FineTunePreviewResponse {
  return {
    experiment: { id: "exp-fasted", name: "Fasted cohort" },
    asset_count: 10,
    annotation_count: 7,
    confirmed_areas: 5,
    done_rois: 2,
    tile_count: 12,
    per_image: [
      { asset_id: "img-1", name: "liver_01.tif", confirmed_areas: 3, done_rois: 0, tiles: 5 },
      { asset_id: "img-2", name: "liver_02.tif", confirmed_areas: 2, done_rois: 1, tiles: 5 },
      { asset_id: "img-3", name: "liver_03.tif", confirmed_areas: 0, done_rois: 1, tiles: 2 },
    ],
    default_mode: "holdout_1",
    eligible: true,
    blockers: [],
    ...overrides,
  };
}

function progressBody(overrides: Partial<FineTuneProgress> = {}): FineTuneProgress {
  return {
    status: "RUNNING",
    stage: "training",
    step: 240,
    total_steps: 600,
    round: 2,
    total_rounds: 5,
    percent: 38,
    eta_seconds: 145,
    message: "Training on 12 tiles",
    error: "",
    ...overrides,
  };
}

interface Stubs {
  previewBody?: Partial<FineTunePreviewResponse>;
  progress?: Partial<FineTuneProgress>;
  onApply?: (assetIds: string[]) => void;
}

function stubApi({ previewBody, progress, onApply }: Stubs = {}) {
  server.use(
    http.get(`${API}/api/finetune/scope/`, () => HttpResponse.json(SCOPE)),
    http.get(`${API}/api/finetune/adapters/`, () =>
      HttpResponse.json([
        {
          id: "ad-old",
          name: "First attempt",
          base_model: "quantem:mito",
          status: "SUCCESS",
          created_at: "2026-01-01T00:00:00Z",
          experiment: { id: "exp-fasted", name: "Fasted cohort" },
          asset_count: 4,
        },
      ])
    ),
    http.post(`${API}/api/finetune/preview/`, () =>
      HttpResponse.json(preview(previewBody))
    ),
    http.post(`${API}/api/finetune/runs/`, () =>
      HttpResponse.json({ adapter_id: "ad-new", job_id: "job-1" }, { status: 202 })
    ),
    http.get(`${API}/api/finetune/runs/:id/progress/`, () =>
      HttpResponse.json(progressBody(progress))
    ),
    http.get(`${API}/api/finetune/runs/:id/`, () =>
      HttpResponse.json({ id: "ad-new", caveats: [], cv_results: {} })
    ),
    http.post(`${API}/api/finetune/runs/:id/apply/`, async ({ request }) => {
      const body = (await request.json()) as { asset_ids: string[] };
      onApply?.(body.asset_ids);
      return HttpResponse.json(
        {
          queued: body.asset_ids.map((assetId) => ({
            asset_id: assetId,
            segmentation_id: `seg-${assetId}`,
            job_id: `job-${assetId}`,
          })),
        },
        { status: 202 }
      );
    })
  );
}

function openDialog(onClose: () => void = () => {}) {
  return render(
    <FineTuneDialog open onClose={onClose} segmentationType={MITO} />
  );
}

/** Tick the ten-image dataset and wait for the server's verdict to land. */
async function selectTheDataset(user: ReturnType<typeof userEvent.setup>) {
  const dataset = await screen.findByRole("checkbox", { name: "Liver 24h" });
  await user.click(dataset);
  await waitFor(() =>
    expect(screen.getByTestId("finetune-count")).toHaveTextContent(
      /cut into 12 tiles/
    )
  );
}

describe("FineTuneDialog", () => {
  it("opens from the library with the paginated organelle response", async () => {
    const user = userEvent.setup();
    stubApi();
    server.use(
      http.get(`${API}/api/segmentation-types/`, () =>
        HttpResponse.json({
          count: 1,
          next: null,
          previous: null,
          results: [MITO],
        })
      )
    );

    render(<FineTuneDialog open onClose={() => {}} />);

    const picker = screen.getByRole("combobox", { name: "Organelle" });
    await screen.findByRole("option", { name: "Mitochondria" });
    await user.selectOptions(picker, MITO.id);

    expect(await screen.findByTestId("scope-tree")).toBeInTheDocument();
  });

  it("renders the tree and totals the owner's ten-image dataset to 7", async () => {
    const user = userEvent.setup();
    stubApi();
    openDialog();

    const tree = await screen.findByTestId("scope-tree");
    expect(within(tree).getByText("Fasted cohort")).toBeInTheDocument();
    expect(within(tree).getByText("Liver 24h")).toBeInTheDocument();
    // Dataset-level by default: the images are behind the expander.
    expect(within(tree).queryByText("liver_01.tif")).not.toBeInTheDocument();

    await selectTheDataset(user);

    const count = screen.getByTestId("finetune-count");
    expect(count).toHaveTextContent("7 annotations across 10 images");
    // The tile count is present but subordinate -- it is not the headline.
    expect(count).toHaveTextContent("5 confirmed areas and 2 reviewed ROIs");
    expect(count).toHaveTextContent("cut into 12 tiles in Fasted cohort");
  });

  it("expands a dataset to sub-select individual images, and re-totals", async () => {
    const user = userEvent.setup();
    stubApi({ previewBody: { annotation_count: 5, asset_count: 2, tile_count: 8 } });
    openDialog();

    await user.click(
      await screen.findByRole("button", { name: "Expand Liver 24h" })
    );
    await user.click(screen.getByRole("checkbox", { name: "liver_01.tif" }));
    await user.click(screen.getByRole("checkbox", { name: "liver_02.tif" }));

    await waitFor(() =>
      expect(screen.getByTestId("finetune-count")).toHaveTextContent(
        "5 annotations across 2 images"
      )
    );
  });

  it("searches the tree", async () => {
    const user = userEvent.setup();
    stubApi();
    openDialog();
    await screen.findByTestId("scope-tree");

    await user.type(screen.getByPlaceholderText("Search datasets and images"), "fed");

    expect(screen.queryByText("Fasted cohort")).not.toBeInTheDocument();
    expect(screen.getByText("Fed cohort")).toBeInTheDocument();
  });

  it("refuses a selection the server calls ineligible, and prints the reason", async () => {
    const user = userEvent.setup();
    stubApi({
      previewBody: {
        eligible: false,
        blockers: [
          "This selection spans two experiments. Everything in one fine-tune has to come from the same experiment.",
        ],
      },
    });
    openDialog();

    await selectTheDataset(user);
    await user.type(screen.getByLabelText("Name"), "Fasted liver mitochondria");

    expect(screen.getByTestId("finetune-blockers")).toHaveTextContent(
      "spans two experiments"
    );
    expect(screen.getByRole("button", { name: "Fine-tune" })).toBeDisabled();
  });

  it("offers three modes and preselects the one the server chose", async () => {
    const user = userEvent.setup();
    stubApi();
    openDialog();
    await selectTheDataset(user);

    const useAll = screen.getByRole("radio", { name: /^Use all/ });
    const holdOut = screen.getByRole("radio", { name: /^Hold out one Keep one/ });
    const cv = screen.getByRole("radio", { name: /cross-validation benchmarking/ });
    expect([useAll, holdOut, cv]).toHaveLength(3);

    // `default_mode` is "holdout_1"; the dialog honours it rather than deciding
    // for itself from the tile count.
    expect(holdOut).toBeChecked();
    expect(useAll).not.toBeChecked();

    await user.click(cv);
    expect(cv).toBeChecked();
    expect(holdOut).not.toBeChecked();
  });

  it("keeps the mode the user chose when the selection changes", async () => {
    const user = userEvent.setup();
    stubApi();
    openDialog();
    await selectTheDataset(user);

    await user.click(screen.getByRole("radio", { name: /^Use all/ }));
    await user.click(screen.getByRole("checkbox", { name: "Fed cohort" }));

    await waitFor(() =>
      expect(screen.getByRole("radio", { name: /^Use all/ })).toBeChecked()
    );
  });

  it("explains the modes from a control a keyboard can reach", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    stubApi();
    openDialog(onClose);
    await screen.findByTestId("scope-tree");

    const help = screen.getByRole("button", {
      name: "About how the training data is used",
    });
    expect(help).toHaveAttribute("aria-expanded", "false");
    expect(
      screen.getByText(/Hold out one, with cross-validation benchmarking —/)
    ).not.toBeVisible();

    // Focus alone opens it: hover-only help does not exist for anyone arriving
    // by keyboard.
    help.focus();
    await waitFor(() => expect(help).toHaveAttribute("aria-expanded", "true"));
    expect(
      screen.getByText(/Hold out one, with cross-validation benchmarking —/)
    ).toBeVisible();
    expect(screen.getByText(/3 tiles or fewer/)).toBeVisible();

    // And it stays open on a click, rather than being toggled shut by the
    // hover the click implies.
    await user.click(help);
    expect(help).toHaveAttribute("aria-expanded", "true");

    // Escape closes the help without also closing the dialog behind it.
    await user.keyboard("{Escape}");
    expect(help).toHaveAttribute("aria-expanded", "false");
    expect(onClose).not.toHaveBeenCalled();
  });

  it("shows steps, rounds and an ETA once the run is under way", async () => {
    const user = userEvent.setup();
    stubApi();
    openDialog();
    await selectTheDataset(user);
    await user.type(screen.getByLabelText("Name"), "Fasted liver mitochondria");
    await user.click(screen.getByRole("button", { name: "Fine-tune" }));

    const bar = await screen.findByTestId("finetune-progress");
    await waitFor(() => expect(bar).toHaveTextContent("round 2 of 5"));
    expect(bar).toHaveTextContent("240 of 600 steps");
    expect(bar).toHaveTextContent("about 2 min left");
    // The server's percent, not one recomputed from 240/600.
    expect(bar).toHaveTextContent("38%");
    expect(screen.getByText("Training on 12 tiles")).toBeInTheDocument();
  });

  it("says it is still estimating rather than showing no time left", async () => {
    const user = userEvent.setup();
    stubApi({ progress: { eta_seconds: null, round: 1, total_rounds: 1 } });
    openDialog();
    await selectTheDataset(user);
    await user.type(screen.getByLabelText("Name"), "Fasted liver mitochondria");
    await user.click(screen.getByRole("button", { name: "Fine-tune" }));

    const bar = await screen.findByTestId("finetune-progress");
    await waitFor(() => expect(bar).toHaveTextContent("estimating time left"));
  });

  it("offers to run the new model on success, and queues nothing until asked", async () => {
    const user = userEvent.setup();
    const applied = vi.fn();
    stubApi({
      progress: { status: "SUCCESS", step: 600, percent: 100, eta_seconds: 0 },
      onApply: applied,
    });
    openDialog();
    await selectTheDataset(user);
    await user.type(screen.getByLabelText("Name"), "Fasted liver mitochondria");
    await user.click(screen.getByRole("button", { name: "Fine-tune" }));

    const success = await screen.findByTestId("finetune-success");
    expect(success).toHaveTextContent("is trained and saved");
    // The offer is on screen and nothing has been queued.
    expect(applied).not.toHaveBeenCalled();

    const images = within(screen.getByTestId("finetune-apply-images"));
    expect(images.getAllByRole("checkbox")).toHaveLength(3);
    // "Some or all": untick one and the button says so.
    await user.click(images.getByRole("checkbox", { name: "liver_03.tif" }));
    const run = screen.getByRole("button", { name: "Run on 2 images" });
    expect(applied).not.toHaveBeenCalled();

    await user.click(run);

    await waitFor(() => expect(applied).toHaveBeenCalledWith(["img-1", "img-2"]));
    expect(await screen.findByTestId("finetune-applied")).toHaveTextContent(
      "Queued on 2 images"
    );
  });

  it("says a failure changed nothing, and leaves a way out", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    stubApi({
      progress: {
        status: "FAILED",
        step: 130,
        percent: 21,
        error: "Training ran out of memory before it finished.",
      },
    });
    openDialog(onClose);
    await selectTheDataset(user);
    await user.type(screen.getByLabelText("Name"), "Fasted liver mitochondria");
    await user.click(screen.getByRole("button", { name: "Fine-tune" }));

    const failed = await screen.findByTestId("finetune-failed");
    expect(failed).toHaveTextContent("ran out of memory");

    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(onClose).toHaveBeenCalled();
  });

  it("names an existing fine-tune when it is picked for overwrite", async () => {
    const user = userEvent.setup();
    stubApi();
    openDialog();
    await screen.findByTestId("scope-tree");

    await user.selectOptions(
      await screen.findByLabelText("Overwrite an existing fine-tune"),
      "ad-old"
    );

    expect(screen.getByLabelText("Name")).toHaveValue("First attempt");
    expect(
      screen.getByText(/old weights stay in place until this run succeeds/)
    ).toBeInTheDocument();
  });

  it("warns before the server has to, when a new name is already taken", async () => {
    const user = userEvent.setup();
    stubApi();
    openDialog();
    await selectTheDataset(user);

    await user.type(screen.getByLabelText("Name"), "First attempt");

    expect(
      screen.getByText(/already called that. Pick it below to replace it/)
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Fine-tune" })).toBeDisabled();
  });
});
