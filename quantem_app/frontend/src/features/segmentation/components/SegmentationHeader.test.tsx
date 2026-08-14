import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SegmentationHeader } from "@/features/segmentation/components/SegmentationHeader";
import { ApiRequestError } from "@/shared/api/core/http";
import { server } from "@/test/msw/server";
import type {
  AssetDetail,
  ImageSegmentation,
  SegmentationCompletionPreview,
  SourceModelOption,
} from "@/shared/types";
import type { AppliedAdapterState } from "@/features/models/appliedAdapter";
import type { ModelCatalogue, ModelPack } from "@/shared/types/finetune";

/**
 * The completion preview the confirmation reads.
 *
 * `GET /api/segmentations/<id>/complete` is the only trustworthy source for the
 * count the dialog shows: `POST` compares it against a fresh read and refuses a
 * stale one, so a dialog that quoted `segment_counts` off the segmentation
 * payload would just fail at the last click.
 */
function makePreview(
  overrides: Partial<SegmentationCompletionPreview> = {}
): SegmentationCompletionPreview {
  return {
    segmentation_id: "seg-1",
    status_stage: "CANDIDATES_READY",
    is_complete: false,
    confirmed_count: 1,
    discard_count: 7,
    discard_by_label_state: { CANDIDATE: 2, INFERRED: 5, EXCLUDED: 0 },
    discard_by_source_model: { "quantem:mito": 7 },
    restorable: true,
    archive_max_objects: 20000,
    restorable_count: 0,
    ...overrides,
  };
}

function usePreview(overrides: Partial<SegmentationCompletionPreview> = {}) {
  server.use(
    http.get("http://127.0.0.1:8000/api/segmentations/:segId/complete", () =>
      HttpResponse.json(makePreview(overrides))
    )
  );
}

const QUANTEM_MITO: SourceModelOption = {
  value: "quantem:mito",
  label: "QuantEM",
  model_family: "quantem",
  is_default: true,
  count: 214,
};

const OMNIEM_MITO: SourceModelOption = {
  value: "omniem:mito",
  label: "OmniEM",
  model_family: "omniem",
  count: 0,
};

const MANUAL: SourceModelOption = {
  value: "manual",
  label: "Manual",
  model_family: "manual",
  count: 1,
};

function makeModelPack(
  family: "quantem" | "omniem",
  installed: boolean,
  downloadBytes: number
): ModelPack {
  return {
    id: `${family}:mito`,
    family,
    organelle: "mito",
    title: family === "quantem" ? "QuantEM — Mitochondria" : "OmniEM — Mitochondria",
    installed,
    download_bytes: downloadBytes,
    canonical_nm: 8,
    tile_size: 512,
    default_threshold: 0.5,
    decoder: "decoder",
    neck: "neck",
    adapt: "adapt",
    licence: "licence",
    notes: "",
    runnable: installed,
    reason: null,
  };
}

const MODEL_CATALOGUE: ModelCatalogue = {
  packs: [
    makeModelPack("quantem", false, 2_500_000_000),
    makeModelPack("omniem", true, 4_000_000_000),
  ],
  adapted: [],
  device: null,
};

function makeImage(overrides: Partial<AssetDetail> = {}): AssetDetail {
  return {
    id: "img-1",
    file_path: "",
    original_filename: "source.tif",
    display_name: "Image 1",
    is_eval_set: false,
    width: 1000,
    height: 1000,
    channels: 1,
    bit_depth: 8,
    preprocess_stage: "DONE",
    preprocess_progress: 100,
    tags: [],
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ngff_ready: true,
    ngff_url: "/ngff/img-1.zarr",
    ...overrides,
  };
}

function makeSegmentation(overrides: Partial<ImageSegmentation> = {}): ImageSegmentation {
  return {
    id: "seg-1",
    segmentation_type: {
      id: "type-1",
      internal_name: "quantem_internal_mito",
      short_name: "Mito",
      long_name: "Mitochondria",
      default_color: null,
      tags: [],
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    },
    status_stage: "CANDIDATES_READY",
    status_progress: 100,
    segment_counts: {
      CONFIRMED: 1,
      EXCLUDED: 0,
      INFERRED: 5,
      CANDIDATE: 2,
    },
    source_models: [QUANTEM_MITO, OMNIEM_MITO, MANUAL],
    config: {
      supports_instance_params: true,
      instance_params: null,
    },
    is_complete: false,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

/** Every required prop, so each test only states what it is actually about. */
function renderHeader(overrides: Partial<React.ComponentProps<typeof SegmentationHeader>> = {}) {
  const seg = overrides.currentSegmentation ?? makeSegmentation();
  return render(
    <SegmentationHeader
      image={makeImage()}
      currentSegmentation={seg}
      sourceModelOptions={seg?.source_models}
      activeSourceModel="quantem:mito"
      displayedSourceModel="quantem:mito"
      onBackToHome={vi.fn()}
      onBackToViewer={vi.fn()}
      onToggleSegmentationComplete={vi.fn()}
      {...overrides}
    />
  );
}

describe("SegmentationHeader", () => {
  // The header asks for a preview whenever the confirmation opens, and the
  // suite runs with `onUnhandledRequest: "error"`.
  beforeEach(() => {
    usePreview();
  });

  it("fires action callbacks", async () => {
    const user = userEvent.setup();
    const onBackToHome = vi.fn();
    const onMarkDone = vi.fn();

    renderHeader({
      onBackToHome,
      onToggleSegmentationComplete: onMarkDone,
    });

    await user.click(screen.getByRole("button", { name: "← Back to Library" }));
    // "Mark Image Done" can throw away a whole inference pass, so it goes
    // through a confirmation; see the dedicated tests below.
    await user.click(screen.getByRole("button", { name: "Mark Image Done" }));
    await user.click(await screen.findByRole("button", { name: "Mark done" }));
    expect(onBackToHome).toHaveBeenCalledTimes(1);
    expect(onMarkDone).toHaveBeenCalledTimes(1);
  });

  it("keeps model-run actions out of the header", () => {
    renderHeader();

    expect(
      screen.queryByRole("button", { name: "Run Full Segmentation" })
    ).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Run Active ROI" })).not.toBeInTheDocument();
  });

  it("keeps the organelle selector out of the header", () => {
    renderHeader();

    expect(
      screen.queryByRole("combobox", { name: "Segmentation type" })
    ).not.toBeInTheDocument();
  });

  /**
   * "Marking it done locks the segmentation" is a promise the dialog has always
   * made and the app did not keep: every mutation control stayed enabled. The
   * server now refuses mutations on a COMPLETED segmentation, so the screen has
   * to show the lock rather than let the user find it by being refused.
   */
  describe("locked segmentation", () => {
    const DONE = makeSegmentation({ status_stage: "COMPLETED", is_complete: true });

    it("says it is locked, and how to unlock it", () => {
      renderHeader({ currentSegmentation: DONE });

      // Not only a tooltip: unreachable by keyboard, and a disabled control
      // with nothing beside it reads as a bug.
      const notice = screen.getByRole("status");
      expect(notice).toHaveTextContent("This segmentation is locked.");
      expect(notice).toHaveTextContent("Unlock segmentation");
      expect(
        screen.getByRole("button", { name: "Unlock segmentation" })
      ).toBeEnabled();
    });

    it("does not show a lock notice while the segmentation is open", () => {
      renderHeader();

      expect(screen.queryByText(/This segmentation is locked/)).not.toBeInTheDocument();
    });
  });

  /**
   * `status_error` reached no screen at all.
   *
   * A user cancelled an ER re-run. The server marked the segmentation FAILED
   * and wrote the reason. The header went on rendering the ordinary count chip
   * -- "190 confirmed of 190 from QuantEM", the 190 being the *previous* run's
   * objects -- with no trace that the run they had just launched produced
   * nothing. Re-run at a corrected pixel size, have it fail, and the
   * wrongly-scaled objects still look finished.
   */
  describe("a run that failed", () => {
    const CANCELLED =
      "Cancelled before it finished, so it produced no result. Nothing was " +
      "saved; start it again when you are ready.";

    const FAILED = makeSegmentation({
      status_stage: "FAILED",
      status_error: CANCELLED,
      segment_counts: { CONFIRMED: 190, EXCLUDED: 0, INFERRED: 0, CANDIDATE: 0 },
      source_models: [{ ...QUANTEM_MITO, count: 190 }, OMNIEM_MITO],
    });

    it("says the run failed, in text and not only in a tooltip", () => {
      renderHeader({ currentSegmentation: FAILED });

      // A tooltip is unreachable by keyboard and invisible unless you hover the
      // right ten pixels, which is the same reason the locked and blocked
      // notices beside it are rendered as text.
      expect(
        screen.getByText(/The last run on this segmentation failed\./)
      ).toBeInTheDocument();
    });

    it("prints the server's reason, which is the only thing that names it", () => {
      renderHeader({ currentSegmentation: FAILED });

      // "Cancelled", "the worker died" and "removed from the queue" all arrive
      // as FAILED and want different responses.
      expect(screen.getByText(new RegExp("Cancelled before it finished"))).toBeInTheDocument();
    });

    it("says the objects on screen belong to an earlier run", () => {
      renderHeader({ currentSegmentation: FAILED });

      expect(
        screen.getByText(
          /the 190 shown here were already on this segmentation before it started/
        )
      ).toBeInTheDocument();
    });

    it("does not report the previous run's count as the finished answer", () => {
      renderHeader({ currentSegmentation: FAILED });

      const chip = screen.getByTestId("displayed-objects-provenance");
      expect(chip).not.toHaveTextContent("190 confirmed of 190 from QuantEM");
      expect(chip).toHaveTextContent("Last run failed");
      expect(chip.className).toContain("error");
    });

    it("says nothing about a failure on a segmentation that has not failed", () => {
      renderHeader();

      expect(
        screen.queryByText(/The last run on this segmentation failed/)
      ).not.toBeInTheDocument();
    });
  });

  /**
   * Paper-cut 1: retry failures accrued in Tasks & Queues while the header
   * showed an older error.
   *
   * A retrying job is not FAILED -- the stage is the queue's business and is
   * left alone -- but after every failed attempt the server now writes
   * "Attempt N of M failed; retrying automatically. <error>" onto
   * `status_error`, superseding whatever an earlier run left there. The header
   * used to render `status_error` only on stage FAILED, so the whole retry
   * cycle played out with the newest failure visible only in the queue screen.
   */
  describe("a run that is retrying", () => {
    const RETRYING = makeSegmentation({
      status_stage: "RUNNING_INFERENCE",
      status_progress: 40,
      status_error:
        "Attempt 2 of 3 failed; retrying automatically. " +
        "failed: RuntimeError: CUDA error: out of memory",
    });

    it("renders the newest attempt's failure verbatim, not an older error", () => {
      renderHeader({ currentSegmentation: RETRYING });

      const notice = screen.getByTestId("latest-attempt-notice");
      expect(notice).toHaveTextContent(
        "Attempt 2 of 3 failed; retrying automatically."
      );
      expect(notice).toHaveTextContent("CUDA error: out of memory");
    });

    it("does not dress a retry up as a concluded failure", () => {
      renderHeader({ currentSegmentation: RETRYING });

      // "The last run on this segmentation failed" is the terminal-failure
      // sentence; another attempt is coming, and the server's own wording
      // already says so.
      expect(
        screen.queryByText(/The last run on this segmentation failed/)
      ).not.toBeInTheDocument();
    });

    it("also surfaces the abandoned-run repair's sentence", () => {
      // `status_reconcile` writes its message with the stage moved back to
      // CANDIDATES_READY, not FAILED, so it reached no screen at all before.
      const abandoned = makeSegmentation({
        status_error:
          "The last run stopped before it finished (its worker is no longer " +
          "running, usually because the application was closed or restarted " +
          "mid-run). Nothing already saved was lost. Run it again when you " +
          "are ready.",
      });
      renderHeader({ currentSegmentation: abandoned });

      expect(screen.getByTestId("latest-attempt-notice")).toHaveTextContent(
        "The last run stopped before it finished"
      );
    });

    it("shows nothing when the newest attempt succeeded and cleared the note", () => {
      renderHeader();

      expect(screen.queryByTestId("latest-attempt-notice")).not.toBeInTheDocument();
    });
  });

  /**
   * The single worst thing the app did: one click on the most prominent button
   * on the screen deleted a whole inference pass, with no dialog, and "Unlock
   * segmentation" brought none of it back.
   *
   * The endpoint now defaults to keeping everything and refuses a discard whose
   * acknowledged count is stale, so the dialog's job is to (a) ask, (b) quote a
   * count read live from `GET .../complete` rather than from a segmentation
   * payload that can be a poll behind, and (c) keep the deletion a separate,
   * opt-in decision from locking.
   */
  describe("Mark Image Done", () => {
    it("does not destroy anything on the first click", async () => {
      const user = userEvent.setup();
      const onMarkDone = vi.fn();
      renderHeader({ onToggleSegmentationComplete: onMarkDone });

      await user.click(screen.getByRole("button", { name: "Mark Image Done" }));

      expect(await screen.findByRole("dialog")).toBeInTheDocument();
      expect(onMarkDone).not.toHaveBeenCalled();
    });

    it("counts what would go by asking the server, not the cached payload", async () => {
      // The segmentation prop says 7 unconfirmed; the server says 32 because a
      // run finished. The dialog must quote the server, or the POST it makes
      // gets refused for a stale acknowledged count.
      const user = userEvent.setup();
      usePreview({ discard_count: 32, confirmed_count: 3 });
      renderHeader();

      await user.click(screen.getByRole("button", { name: "Mark Image Done" }));

      expect(
        await screen.findByText(/holds 32 objects nobody has confirmed/)
      ).toBeInTheDocument();
      expect(screen.getByText(/3 objects you confirmed/)).toBeInTheDocument();
    });

    it("locks without deleting anything unless the discard is ticked", async () => {
      const user = userEvent.setup();
      const onMarkDone = vi.fn();
      usePreview({ discard_count: 7, confirmed_count: 1 });
      renderHeader({ onToggleSegmentationComplete: onMarkDone });

      await user.click(screen.getByRole("button", { name: "Mark Image Done" }));
      await user.click(await screen.findByRole("button", { name: "Mark done" }));

      expect(onMarkDone).toHaveBeenCalledWith(undefined);
    });

    it("sends the discard with the exact count the user was shown", async () => {
      const user = userEvent.setup();
      const onMarkDone = vi.fn();
      usePreview({ discard_count: 32, confirmed_count: 0 });
      renderHeader({ onToggleSegmentationComplete: onMarkDone });

      await user.click(screen.getByRole("button", { name: "Mark Image Done" }));
      await user.click(
        await screen.findByRole("checkbox", {
          name: /Also delete the 32 objects nobody confirmed/,
        })
      );
      await user.click(
        screen.getByRole("button", { name: "Mark done and delete 32 objects" })
      );

      expect(onMarkDone).toHaveBeenCalledWith({
        discardUnconfirmed: true,
        acknowledgedDiscardCount: 32,
      });
    });

    /**
     * `discardable_queryset` is "everything that is not CONFIRMED", so the
     * count covers candidates nobody has opened *and* objects somebody opened
     * and rejected. Same arithmetic, different loss: a rejection is the record
     * of a review, and `fetchGroundTruthProvenance` feeds EXCLUDED objects to
     * the fine-tuning wizard as negative examples.
     */
    it("separates rejections from candidates nobody reviewed", async () => {
      const user = userEvent.setup();
      usePreview({
        discard_count: 39,
        confirmed_count: 12,
        discard_by_label_state: { CANDIDATE: 20, INFERRED: 12, EXCLUDED: 7 },
      });
      renderHeader();

      await user.click(screen.getByRole("button", { name: "Mark Image Done" }));

      expect(
        await screen.findByText(
          /32 objects were never reviewed and 7 are ones you rejected/
        )
      ).toBeInTheDocument();
      expect(screen.getByText(/trains against rejections/)).toBeInTheDocument();
      expect(
        screen.getByRole("checkbox", {
          name: /Also delete all 39 objects: the 32 never reviewed and the 7 you rejected/,
        })
      ).toBeInTheDocument();
    });

    it("keeps the plain wording when nothing was rejected", async () => {
      const user = userEvent.setup();
      usePreview({
        discard_count: 32,
        confirmed_count: 0,
        discard_by_label_state: { CANDIDATE: 30, INFERRED: 2, EXCLUDED: 0 },
      });
      renderHeader();

      await user.click(screen.getByRole("button", { name: "Mark Image Done" }));

      expect(
        await screen.findByRole("checkbox", {
          name: /Also delete the 32 objects nobody confirmed/,
        })
      ).toBeInTheDocument();
      expect(
        screen.queryByText(/trains against rejections/)
      ).not.toBeInTheDocument();
    });

    it("names the models the doomed objects came from", async () => {
      // The reported case: 32 model-detected objects, zero confirmations.
      const user = userEvent.setup();
      usePreview({
        discard_count: 32,
        confirmed_count: 0,
        discard_by_source_model: { "quantem:mito": 32 },
      });
      renderHeader();

      await user.click(screen.getByRole("button", { name: "Mark Image Done" }));

      expect(await screen.findByText(/32 from QuantEM/)).toBeInTheDocument();
    });

    it("says the discard can be undone when the server says it can", async () => {
      const user = userEvent.setup();
      usePreview({ discard_count: 7, confirmed_count: 1, restorable: true });
      renderHeader();

      await user.click(screen.getByRole("button", { name: "Mark Image Done" }));
      await user.click(await screen.findByRole("checkbox"));

      expect(
        screen.getByText(/archived first/)
      ).toBeInTheDocument();
    });

    it("says plainly when a discard is too large to undo", async () => {
      const user = userEvent.setup();
      usePreview({
        discard_count: 40000,
        confirmed_count: 1,
        restorable: false,
        archive_max_objects: 20000,
      });
      renderHeader();

      await user.click(screen.getByRole("button", { name: "Mark Image Done" }));
      await user.click(await screen.findByRole("checkbox"));

      expect(
        screen.getByText(/cannot be archived/)
      ).toBeInTheDocument();
    });

    it("destroys nothing when the confirmation is cancelled", async () => {
      const user = userEvent.setup();
      const onMarkDone = vi.fn();
      usePreview({ discard_count: 7, confirmed_count: 1 });
      renderHeader({ onToggleSegmentationComplete: onMarkDone });

      await user.click(screen.getByRole("button", { name: "Mark Image Done" }));
      await user.click(await screen.findByRole("button", { name: "Cancel" }));

      expect(onMarkDone).not.toHaveBeenCalled();
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });

    it("offers no deletion when there is nothing unconfirmed", async () => {
      const user = userEvent.setup();
      usePreview({ discard_count: 0, confirmed_count: 4 });
      renderHeader();

      await user.click(screen.getByRole("button", { name: "Mark Image Done" }));

      expect(
        await screen.findByText(/nothing here is unconfirmed/i)
      ).toBeInTheDocument();
      expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Mark done" })).toBeInTheDocument();
    });

    it("keeps the dialog open and says why when the server refuses", async () => {
      // The 409 the endpoint returns when the count moved while the dialog was
      // open. Nothing was deleted, and the user has to be told that.
      const user = userEvent.setup();
      usePreview({ discard_count: 7, confirmed_count: 1 });
      const onMarkDone = vi.fn().mockRejectedValue(
        new ApiRequestError(
          JSON.stringify({
            detail:
              "This segmentation now holds 9 unconfirmed object(s), not 7. Nothing was deleted.",
          }),
          { status: 409 }
        )
      );
      renderHeader({ onToggleSegmentationComplete: onMarkDone });

      await user.click(screen.getByRole("button", { name: "Mark Image Done" }));
      await user.click(await screen.findByRole("checkbox"));
      await user.click(
        screen.getByRole("button", { name: "Mark done and delete 7 objects" })
      );

      expect(await screen.findByRole("alert")).toHaveTextContent(
        /now holds 9 unconfirmed/
      );
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });

    it("unlocks on the first click, because unlocking restores rather than destroys", async () => {
      const user = userEvent.setup();
      const onToggle = vi.fn();
      renderHeader({
        currentSegmentation: makeSegmentation({
          status_stage: "COMPLETED",
          is_complete: true,
        }),
        onToggleSegmentationComplete: onToggle,
      });

      await user.click(screen.getByRole("button", { name: "Unlock segmentation" }));

      expect(onToggle).toHaveBeenCalledTimes(1);
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });

    it("says so rather than guessing when the preview cannot be read", async () => {
      const user = userEvent.setup();
      server.use(
        http.get("http://127.0.0.1:8000/api/segmentations/:segId/complete", () =>
          HttpResponse.json({ detail: "nope" }, { status: 500 })
        )
      );
      renderHeader();

      await user.click(screen.getByRole("button", { name: "Mark Image Done" }));

      expect(
        await screen.findByText(/will still lock the segmentation/)
      ).toBeInTheDocument();
    });
  });

  /**
   * An adapter applied to a segmentation changes the threshold every run uses
   * and, in `head` mode, the weights. `apply_active_adapter` only honours it
   * when the run's source model is the one it was fitted on. Neither fact
   * appeared anywhere on the screen that starts the run.
   */
  describe("applied adapter", () => {
    const APPLIED: AppliedAdapterState = {
      adapter: {
        id: "adapted:a1",
        base: "quantem:mito",
        name: "mito @ liver_HFD2",
        created_at: "2026-01-01T00:00:00Z",
        calibrated_threshold: 0.45,
        heldout_dice: 0.9,
        split_mode: "image-disjoint",
        mode: "head",
        segmentation_id: "seg-1",
        applied_at: "2026-01-02T00:00:00Z",
      },
      active: true,
      selectedSourceModel: "quantem:mito",
      publishedThreshold: 0.5,
      trainedHead: true,
    };

    it("says the run will use the fine-tuned head, and at what threshold", () => {
      renderHeader({ appliedAdapter: APPLIED });

      expect(screen.getByText(/Adapted model: mito @ liver_HFD2/)).toBeInTheDocument();
      expect(
        screen.getByText(
          /Run model will use your fine-tuned head at threshold 0\.45, not the published 0\.50/
        )
      ).toBeInTheDocument();
    });

    it("marks the adapted option in the model picker", () => {
      renderHeader({ appliedAdapter: APPLIED });

      expect(screen.getByRole("combobox", { name: "Model" })).toHaveValue(
        "quantem:mito"
      );
      expect(screen.getByRole("option", { name: "QuantEM (adapted)" })).toBeInTheDocument();
    });

    it("warns when the selected model means the adapter will be skipped", () => {
      // The silent case: the run falls back to the released pack at its
      // published threshold and nothing on screen changes.
      renderHeader({
        activeSourceModel: "omniem:mito",
        appliedAdapter: {
          ...APPLIED,
          active: false,
          selectedSourceModel: "omniem:mito",
        },
      });

      expect(screen.getByText(/Adapter not in use/)).toBeInTheDocument();
      expect(
        screen.getByText(/will use the published model at threshold 0\.50/)
      ).toBeInTheDocument();
    });

    it("says nothing when no adapter is applied", () => {
      renderHeader();

      expect(screen.queryByText(/Adapted model:/)).not.toBeInTheDocument();
      expect(screen.queryByText(/Adapter not in use/)).not.toBeInTheDocument();
      const picker = screen.getByRole("combobox", { name: "Model" });
      expect(picker.textContent).not.toContain("(adapted)");
    });

    it("does not claim a head when only the threshold was calibrated", () => {
      renderHeader({
        appliedAdapter: {
          ...APPLIED,
          adapter: { ...APPLIED.adapter, mode: "threshold_only" },
          trainedHead: false,
        },
      });

      expect(
        screen.getByText(/will use your calibration at threshold 0\.45/)
      ).toBeInTheDocument();
      expect(screen.queryByText(/fine-tuned head/)).not.toBeInTheDocument();
    });
  });

  it("offers routes back to the library, experiment, and viewer", async () => {
    const user = userEvent.setup();
    const onBackToExperiment = vi.fn();
    const onBackToViewer = vi.fn();
    renderHeader({ onBackToExperiment, onBackToViewer });

    expect(
      screen.getByRole("button", { name: "← Back to Library" })
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Back to Experiment" }));
    await user.click(screen.getByRole("button", { name: "Back to Viewer" }));
    expect(onBackToExperiment).toHaveBeenCalledOnce();
    expect(onBackToViewer).toHaveBeenCalledOnce();
    expect(screen.queryByText("source.tif")).not.toBeInTheDocument();
  });

  it("names no model that this product does not ship", () => {
    renderHeader();

    // The regression this whole component was rewritten for: a hard-coded
    // "MitoNet v1_mini" fallback naming a model that produced nothing.
    expect(screen.queryByText(/mitonet/i)).toBeNull();
    expect(screen.queryByText(/v1_mini/)).toBeNull();
  });

  it("reports the source model the displayed objects actually came from", () => {
    renderHeader();

    const provenance = screen.getByTestId("displayed-objects-provenance");
    // The confirmed count is on the chip itself, not only in the tooltip: it
    // is the number the analysis measures, and a tooltip is invisible unless
    // you happen to hover it.
    expect(provenance).toHaveTextContent("1 confirmed · 214 from QuantEM");
    expect(provenance).toHaveAttribute(
      "title",
      expect.stringContaining("the number the analysis measures")
    );
  });

  /**
   * A completed run that found nothing told the user it had never run.
   *
   * Reported on a calibrated 5 nm/px image with `quantem:mito`: the run reached
   * `CANDIDATES_READY` with zero candidates and the header read "No objects from
   * QuantEM yet." Running again produces the same result; the pixel size is
   * what moves the answer
   * (0 / 19 / 120 / 233 objects over the same pixels at 5 nm, unset, 10 nm,
   * 20 nm). The server had diagnosed it and put it on `run_notice`; no screen
   * read the field.
   */
  describe("a run that found nothing", () => {
    const EMPTY_RUN = makeSegmentation({
      segment_counts: { CONFIRMED: 0, EXCLUDED: 0, INFERRED: 0, CANDIDATE: 0 },
      source_models: [{ ...QUANTEM_MITO, count: 0 }, OMNIEM_MITO],
      run_notice: {
        kind: "no_objects",
        source_model: "quantem:mito",
        message: "This run finished without finding any objects.",
        next_steps: [
          "Check the image's pixel size (5 nm/px). It decides what size the model thinks these organelles are, and a wrong value makes a working model find nothing -- check it before the threshold, because lowering the threshold on a wrongly-scaled run does not bring the objects back.",
          "Lower the detection threshold and run again.",
          "Check that the selected model is trained for this organelle.",
        ],
      },
    });

    it("stops the chip claiming the model was never run", () => {
      renderHeader({ currentSegmentation: EMPTY_RUN });

      const provenance = screen.getByTestId("displayed-objects-provenance");
      expect(provenance).toHaveTextContent("Ran and found no objects");
      expect(provenance).not.toHaveAttribute(
        "title",
        expect.stringContaining("Nothing has been run")
      );
    });

    it("puts the full advice in the tag's hover message, not a separate box", () => {
      renderHeader({ currentSegmentation: EMPTY_RUN });

      const tag = screen.getByTestId("displayed-objects-provenance");
      expect(tag).toHaveAttribute(
        "title",
        expect.stringContaining("This run finished without finding any objects.")
      );
      expect(tag).toHaveAttribute("title", expect.stringContaining("5 nm/px"));
      expect(tag).toHaveAttribute(
        "title",
        expect.stringContaining("Lower the detection threshold and run again.")
      );
      expect(
        screen.queryByText("This run finished without finding any objects.")
      ).not.toBeInTheDocument();
    });

    it("does not show the tag for manual segmentation", () => {
      renderHeader({
        currentSegmentation: EMPTY_RUN,
        activeSourceModel: "manual",
        displayedSourceModel: null,
      });
      expect(screen.getByTestId("displayed-objects-provenance")).not.toHaveTextContent(
        "Ran and found no objects"
      );
    });

    it("shows a proven empty run even though it has no object overlay", () => {
      renderHeader({ currentSegmentation: EMPTY_RUN, displayedSourceModel: null });
      expect(screen.getByTestId("displayed-objects-provenance")).toHaveTextContent(
        "Ran and found no objects"
      );
    });

    it("does not show the tag when the model run failed", () => {
      renderHeader({
        currentSegmentation: makeSegmentation({
          ...EMPTY_RUN,
          status_stage: "FAILED",
          status_error: "The model could not finish.",
        }),
      });
      expect(screen.getByTestId("displayed-objects-provenance")).not.toHaveTextContent(
        "Ran and found no objects"
      );
    });

    it("says nothing when the server sent no notice", () => {
      renderHeader({ displayedSourceModel: null });

      expect(
        screen.queryByText(/finished without finding any objects/)
      ).not.toBeInTheDocument();
      expect(screen.getByTestId("displayed-objects-provenance")).not.toHaveTextContent(
        "Ran and found no objects"
      );
    });
  });

  /**
   * The second empty run: a re-run over a proofread segmentation, which is
   * *expected* to add nothing (extraction drops candidates landing on
   * confirmed or excluded objects). Composing the chip line client-side said
   * "Ran and found no objects" over twelve confirmed objects — false — so the
   * chip leads with the server's own `run_notice.summary`.
   */
  describe("a re-run that added nothing to a proofread segmentation", () => {
    const PROOFREAD_RERUN = makeSegmentation({
      segment_counts: { CONFIRMED: 12, EXCLUDED: 2, INFERRED: 0, CANDIDATE: 0 },
      run_notice: {
        kind: "no_new_objects",
        source_model: "quantem:mito",
        summary: "Ran and added no new objects",
        message:
          "This run added no new objects. The 14 object(s) already labelled in this image are unchanged.",
        next_steps: [
          "Nothing changed: the 14 object(s) you have already labelled here are exactly as they were.",
          "Rejected model proposals are not added again. Confirmed outlines stay unchanged above any new model preview; accepting that preview later merges strong overlaps or removes the confirmed pixels from it.",
        ],
      },
    });

    it("leads the chip with the server's summary, not a neutral count", () => {
      renderHeader({ currentSegmentation: PROOFREAD_RERUN });

      const provenance = screen.getByTestId("displayed-objects-provenance");
      expect(provenance).toHaveTextContent("Ran and added no new objects");
      // Not the empty-run wording: 12 objects are confirmed here.
      expect(provenance).not.toHaveTextContent("Ran and found no objects");
    });

    it("keeps the notice body in the chip tooltip", () => {
      renderHeader({ currentSegmentation: PROOFREAD_RERUN });

      const chip = screen.getByTestId("displayed-objects-provenance");
      expect(chip).toHaveAttribute(
        "title",
        expect.stringContaining("This run added no new objects.")
      );
      expect(screen.queryByText(/expected to find nothing new/)).not.toBeInTheDocument();
    });
  });

  /**
   * The resolution tag can read "5 nm/px" over objects produced before that
   * number existed — the state `run_analysis`
   * blanks every physical unit on. This screen is where the user decides the
   * work is finished, so it says so here, not first in the finished bundle.
   */
  describe("objects that predate the calibration", () => {
    it("warns beside the pixel-size badge when the server says so", () => {
      renderHeader({
        image: makeImage({ pixel_size_nm: 5 }),
        currentSegmentation: makeSegmentation({
          objects_pixel_size: {
            produced_nm: [null],
            predates_calibration: true,
            unstamped_count: 0,
          },
        }),
      });

      const chip = screen.getByTestId("objects-pixel-size-warning");
      expect(chip).toHaveTextContent("Objects predate the pixel size");
      expect(chip).toHaveAttribute(
        "title",
        expect.stringContaining("produced while this image had no pixel size")
      );
      expect(chip).toHaveAttribute(
        "title",
        expect.stringContaining("in pixels")
      );
    });

    it("stays silent when the objects were made at the recorded scale", () => {
      renderHeader({
        currentSegmentation: makeSegmentation({
          objects_pixel_size: {
            produced_nm: [5],
            predates_calibration: false,
            unstamped_count: 0,
          },
        }),
      });

      expect(
        screen.queryByTestId("objects-pixel-size-warning")
      ).not.toBeInTheDocument();
    });

    it("stays silent on an older backend that does not send the field", () => {
      renderHeader();

      expect(
        screen.queryByTestId("objects-pixel-size-warning")
      ).not.toBeInTheDocument();
    });
  });

  /**
   * The way out of the state the chip warns about. The analysis bundle's own
   * caveat says the objects have to go first (`POST .../labels/clear`) and
   * that "no screen offers that yet"; this button is that screen. The dialog's
   * wording is held to what the endpoint actually does: it deletes by label
   * state, so hand-drawn objects — stored as CONFIRMED — are deleted too, and
   * only unreviewed model candidates survive.
   */
  describe("discard-and-re-run recovery for calibrated-after-the-fact", () => {
    const PREDATES = makeSegmentation({
      objects_pixel_size: {
        produced_nm: [null],
        predates_calibration: true,
        unstamped_count: 0,
      },
      segment_counts: { CONFIRMED: 3, EXCLUDED: 2, CANDIDATE: 2, INFERRED: 5 },
      segment_counts_by_source_model: {
        "quantem:mito": { CONFIRMED: 2, EXCLUDED: 1, CANDIDATE: 2, INFERRED: 5 },
        manual: { CONFIRMED: 1, EXCLUDED: 1, CANDIDATE: 0, INFERRED: 0 },
      },
    });

    it("offers the recovery only beside the warning chip", () => {
      const { unmount } = renderHeader({
        currentSegmentation: PREDATES,
        onClearMislabeledObjects: vi.fn(),
      });
      expect(screen.getByTestId("clear-rerun-button")).toBeInTheDocument();
      unmount();

      // No warning, no recovery to offer: deleting reviewed work is only the
      // right advice when the objects are the mis-scaled set.
      renderHeader({ onClearMislabeledObjects: vi.fn() });
      expect(screen.queryByTestId("clear-rerun-button")).not.toBeInTheDocument();
    });

    it("asks first, and names truthfully what the endpoint deletes — hand-drawn included", async () => {
      const user = userEvent.setup();
      const onClear = vi.fn().mockResolvedValue(undefined);
      renderHeader({
        currentSegmentation: PREDATES,
        onClearMislabeledObjects: onClear,
      });

      await user.click(screen.getByTestId("clear-rerun-button"));

      const dialog = screen.getByRole("dialog");
      expect(dialog).toHaveTextContent(
        "Delete these objects and re-run at the corrected pixel size?"
      );
      // The doomed set, by the endpoint's own criterion (label state).
      expect(dialog).toHaveTextContent("3 objects confirmed and 2 rejected");
      // The truth the convenient wording would hide: hand-drawn objects are
      // stored as CONFIRMED and labels/clear does not spare them. Said as a
      // categorical fact, not a count — `segment_counts_by_source_model`
      // reports every source's CONFIRMED as the all-bundles total, so a
      // per-source number here would be wrong.
      expect(dialog).toHaveTextContent("drawn by hand");
      expect(dialog).toHaveTextContent("not spared");
      expect(dialog).toHaveTextContent("Nothing is archived");
      // What survives: unreviewed model candidates only.
      expect(dialog).toHaveTextContent("7 objects nobody reviewed are kept");
      // And what happens next, naming the model that will run.
      expect(dialog).toHaveTextContent(
        "one full inference pass over the image is queued with QuantEM"
      );
      expect(
        screen.getByRole("button", { name: "Delete 5 objects and re-run" })
      ).toBeInTheDocument();
      // Asking is not doing.
      expect(onClear).not.toHaveBeenCalled();
    });

    it("clears and re-runs only after the confirmation", async () => {
      const user = userEvent.setup();
      const onClear = vi.fn().mockResolvedValue(undefined);
      renderHeader({
        currentSegmentation: PREDATES,
        onClearMislabeledObjects: onClear,
      });

      await user.click(screen.getByTestId("clear-rerun-button"));
      await user.click(
        screen.getByRole("button", { name: "Delete 5 objects and re-run" })
      );

      expect(onClear).toHaveBeenCalledTimes(1);
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });

    it("keeps the dialog open and prints the refusal verbatim when the server refuses", async () => {
      const user = userEvent.setup();
      const refusal =
        "This segmentation is locked (marked done). Unlock it to change objects.";
      const onClear = vi
        .fn()
        .mockRejectedValue(new ApiRequestError(refusal, { status: 409 }));
      renderHeader({
        currentSegmentation: PREDATES,
        onClearMislabeledObjects: onClear,
      });

      await user.click(screen.getByTestId("clear-rerun-button"));
      await user.click(
        screen.getByRole("button", { name: "Delete 5 objects and re-run" })
      );

      expect(await screen.findByRole("alert")).toHaveTextContent(refusal);
      // Still open: closing on a refusal replaces the explanation with silence.
      expect(screen.getByRole("dialog")).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Delete 5 objects and re-run" })
      ).toBeInTheDocument();
    });

    it("is disabled while the segmentation is locked, and says why", () => {
      renderHeader({
        currentSegmentation: makeSegmentation({
          ...PREDATES,
          status_stage: "COMPLETED",
          is_complete: true,
        }),
        onClearMislabeledObjects: vi.fn(),
      });

      const button = screen.getByTestId("clear-rerun-button");
      expect(button).toBeDisabled();
      expect(button).toHaveAttribute("title", expect.stringContaining("locked"));
    });
  });

  it("separates the model selected to run from the objects on screen", () => {
    // Selecting OmniEM, which has produced nothing here, must not read as
    // "these objects came from OmniEM".
    renderHeader({ activeSourceModel: "omniem:mito" });

    expect(screen.getByText("Model")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Model" })).toHaveValue(
      "omniem:mito"
    );
    expect(screen.getByTestId("displayed-objects-provenance")).toHaveTextContent(
      "No objects from OmniEM yet"
    );
  });

  it("flags an overlay still showing another model's output", () => {
    renderHeader({
      activeSourceModel: "quantem:mito",
      displayedSourceModel: "omniem:mito",
    });

    const provenance = screen.getByTestId("displayed-objects-provenance");
    expect(provenance.className).toContain("warning");
    expect(provenance).toHaveAttribute(
      "title",
      expect.stringContaining("overlay still shows output from OmniEM")
    );
  });

  it("treats the legacy None source model as manual segmentation", () => {
    renderHeader({ activeSourceModel: "none" });

    expect(screen.getByRole("combobox", { name: "Model" })).toHaveValue("manual");
    expect(screen.getByTestId("displayed-objects-provenance")).toHaveTextContent(
      "1 confirmed · 1 from Manual"
    );
  });

  it("renders manual segmentation and both released models", async () => {
    const user = userEvent.setup();
    const onSourceModelChange = vi.fn();
    renderHeader({ onSourceModelChange });

    const model = screen.getByRole("combobox", { name: "Model" });
    expect(screen.getByRole("option", { name: "QuantEM" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "OmniEM" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Manual segmentation" })).toBeInTheDocument();

    await user.selectOptions(model, "omniem:mito");
    expect(onSourceModelChange).toHaveBeenCalledWith("omniem:mito");

    await user.selectOptions(model, "manual");
    expect(onSourceModelChange).toHaveBeenCalledWith("manual");
  });

  it("shows only the yellow download status beside a model that needs it", () => {
    renderHeader({ modelCatalogue: MODEL_CATALOGUE });

    expect(screen.getByRole("option", { name: "QuantEM" })).toBeInTheDocument();
    expect(
      screen.getByRole("img", {
        name:
          "Model is not downloaded. Will automatically download (2.5GB) on first run",
      })
    ).toHaveAttribute(
      "title",
      "Model is not downloaded. Will automatically download (2.5GB) on first run"
    );
    expect(screen.getByRole("img")).toHaveAttribute("data-model-state", "download");
    expect(screen.getByRole("img")).toHaveClass("header-model-availability-download");
  });

  it("shows no ready checkmark after a model is downloaded", () => {
    renderHeader({
      activeSourceModel: "omniem:mito",
      modelCatalogue: MODEL_CATALOGUE,
    });

    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByRole("option", { name: "OmniEM" })).toBeInTheDocument();
  });

  it("shows no availability tag for manual segmentation", () => {
    renderHeader({ activeSourceModel: "manual", modelCatalogue: MODEL_CATALOGUE });

    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Manual segmentation" })).toBeInTheDocument();
  });

  it("does not show model-run controls for manual segmentation", () => {
    renderHeader({ activeSourceModel: "manual" });

    expect(
      screen.queryByRole("button", { name: "Run Full Segmentation" })
    ).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Run Active ROI" })).not.toBeInTheDocument();
  });

  it("does not show a pixel-size tag in the labeling header", () => {
    renderHeader({
      image: makeImage({
        pixel_size_nm: 5,
        renditions: [
          {
            id: "rend-1",
            type: "FULL",
            metadata: { source_metadata: { pixel_size_nm: 5 } },
          },
        ],
      }),
    });

    expect(screen.queryByText("5 nm/px")).not.toBeInTheDocument();
  });

  it("does not show an uncalibrated pixel-size tag in the labeling header", () => {
    renderHeader({ image: makeImage({ pixel_size_nm: null }) });

    expect(screen.queryByText("Pixel size not set")).not.toBeInTheDocument();
  });

  it("shows unlock action for completed segmentations", () => {
    renderHeader({
      currentSegmentation: makeSegmentation({
        status_stage: "COMPLETED",
        is_complete: true,
      }),
    });

    expect(screen.getByRole("button", { name: "Unlock segmentation" })).toBeInTheDocument();
  });
});
