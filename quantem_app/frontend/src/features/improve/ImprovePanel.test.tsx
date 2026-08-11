/**
 * The five things P13 exists to fix, one test each, plus the invariant.
 *
 * Every case here failed on the six-step wizard this panel replaces:
 *
 * 1. improving took six screens; it now takes one button, and the result is a
 *    sentence rather than a Dice;
 * 2. **improving twice was impossible** — `StepRun.tsx:101` only offered a
 *    start button when `(!job && !adapter) || outcome.concluded`, and a
 *    succeeded run is neither, while a `localStorage` pointer written on start
 *    and never cleared brought a reload straight back to the finished run;
 * 3. a run with no probability map behind it reached the queue and died there;
 * 4. head training on too small a checked area did the same, minutes later;
 * 5. the honest statistics were four steps deep and unavoidable; they are now
 *    one click away and optional.
 *
 * And I-1, made visible: the panel states, before and after, that nothing the
 * user kept, removed or drew is touched.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { ImprovePanel } from "@/features/improve/ImprovePanel";
import { server } from "@/test/msw/server";
import type {
  Adapter,
  AdaptCropsResponse,
  ModelCatalogue,
  ModelPack,
} from "@/shared/types/finetune";

const API = "http://127.0.0.1:8000";
const ASSET_ID = "asset-1";
const SEG = "seg-mito";

function pack(id: string, organelle: string, title: string): ModelPack {
  return {
    id,
    family: id.split(":")[0] as ModelPack["family"],
    organelle: organelle as ModelPack["organelle"],
    title,
    installed: true,
    download_bytes: 1,
    canonical_nm: 8,
    tile_size: 512,
    default_threshold: 0.5,
    decoder: "affinity_mws",
    neck: "naive_1x1",
    adapt: "last_n",
    licence: "see NOTICE",
    notes: "",
    runnable: true,
    reason: null,
    encoder_tier: "exported",
  };
}

const CATALOGUE: ModelCatalogue = {
  packs: [pack("quantem:mito", "mito", "QuantEM — Mitochondria")],
  adapted: [],
  device: { kind: "cpu", name: "CPU", cuda: false, mps: false },
};

function segmentation() {
  return {
    id: SEG,
    asset: ASSET_ID,
    segmentation_type: {
      id: "type-mito",
      internal_name: "quantem_internal_mito",
      short_name: "Mitochondria",
      long_name: "Mitochondria",
      default_color: null,
      tags: [],
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    },
    segment_counts: { CONFIRMED: 34, EXCLUDED: 6, INFERRED: 0, CANDIDATE: 511 },
    status_stage: "CANDIDATES_READY",
    status_progress: 100,
    config: { supports_instance_params: true, instance_params: null },
    is_complete: false,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
}

function crops(overrides: Partial<AdaptCropsResponse> = {}): AdaptCropsResponse {
  return {
    crops: [
      {
        id: "crop-1",
        name: "asset1_0",
        image_key: ASSET_ID,
        segmentation_id: SEG,
        width: 900,
        height: 900,
        n_objects: 12,
        annotated_px: 810000,
        has_probability: true,
        image_name: "Liver 01",
        is_this_image: true,
      },
      {
        id: "crop-2",
        name: "asset2_0",
        image_key: "asset-2",
        segmentation_id: "seg-other",
        width: 900,
        height: 900,
        n_objects: 9,
        annotated_px: 810000,
        has_probability: true,
        image_name: "Grid2 Cell11",
        is_this_image: false,
      },
    ],
    split_mode: "image-disjoint",
    n_images: 2,
    ready: true,
    blockers: [],
    warnings: [],
    has_probability: true,
    mode_blockers: { threshold_only: [], head: [] },
    head_size: {
      base_model: "quantem:mito",
      ok: true,
      largest_nm: 4500,
      required_nm: 1832,
      largest_px: 900,
      required_px: 229,
      n_areas: 2,
      reason: null,
    },
    train_crop_names: ["asset1_0"],
    heldout_crop_names: ["asset2_0"],
    modes: ["threshold_only", "head"],
    ...overrides,
  };
}

function adapter(overrides: Partial<Adapter> = {}): Adapter {
  return {
    id: "ad-1",
    base_model: "quantem:mito",
    name: "Liver 01",
    status: "SUCCESS",
    mode: "threshold_only",
    steps: 0,
    trainable_params: 0,
    segmentation_id: SEG,
    split_mode: "image-disjoint",
    train_crop_names: ["asset1_0"],
    heldout_crop_names: ["asset2_0"],
    sweep: {
      thresholds: [0.4, 0.5, 0.65],
      train_dice: [0.7, 0.85, 0.9],
      calibrated_threshold: 0.65,
      train_dice_at_calibrated: 0.9,
      train_dice_at_default: 0.85,
      heldout_dice_at_calibrated: 0.83,
      heldout_dice_at_default: 0.78,
      heldout_oracle: 0.86,
      improvement: 0.05,
      per_crop: { asset1_0: 0.9, asset2_0: 0.83 },
      train_crop_names: ["asset1_0"],
      heldout_crop_names: ["asset2_0"],
    },
    calibrated_threshold: 0.65,
    default_threshold: 0.5,
    heldout_dice: 0.83,
    verified_reload: false,
    train_seconds: null,
    applied_at: null,
    created_at: "2026-02-01T00:00:00Z",
    error: "",
    caveats: ["The threshold was fit on the training crops only."],
    ...overrides,
  };
}

interface HandlerOptions {
  crops?: AdaptCropsResponse;
  /** Adapter bodies, consumed in order; the last one repeats. */
  adapters?: Adapter[];
  latest?: { adapter: Adapter | null; job_id: string | null };
}

/** Records every write the panel makes, so "refused before queueing" is testable. */
interface Calls {
  starts: Array<Record<string, unknown>>;
  applies: string[];
  reruns: string[];
}

function installHandlers(options: HandlerOptions = {}): Calls {
  const calls: Calls = { starts: [], applies: [], reruns: [] };
  const adapters = options.adapters ?? [adapter()];
  let started = 0;
  // Applying is a server-side stamp the panel re-reads, so the fake server has
  // to remember it: a GET that still says "not applied" would hide the fact
  // that the panel refetches rather than trusting its own optimism.
  const appliedAt = new Map<string, string>();

  server.use(
    http.get(`${API}/api/models/`, () => HttpResponse.json(CATALOGUE)),
    http.get(`${API}/api/assets/${ASSET_ID}/`, () =>
      HttpResponse.json({ id: ASSET_ID, display_name: "Liver 01" })
    ),
    http.get(`${API}/api/assets/${ASSET_ID}/segmentations/`, () =>
      HttpResponse.json([segmentation()])
    ),
    http.get(`${API}/api/segmentations/:segId/adapt/crops/`, () =>
      HttpResponse.json(options.crops ?? crops())
    ),
    http.get(`${API}/api/segmentations/:segId/adapt/latest/`, () =>
      HttpResponse.json(options.latest ?? { adapter: null, job_id: null })
    ),
    http.get(`${API}/api/segmentations/:segId/completed-rois/`, () =>
      HttpResponse.json([])
    ),
    http.get(`${API}/api/adapters/:adapterId/`, ({ params }) => {
      const id = String(params.adapterId);
      const index = adapters.findIndex((a) => a.id === id);
      const body = index >= 0 ? adapters[index] : adapters[adapters.length - 1];
      return HttpResponse.json({
        ...body,
        applied_at: appliedAt.get(id) ?? body.applied_at,
      });
    }),
    http.post(`${API}/api/segmentations/:segId/adapt/`, async ({ request }) => {
      calls.starts.push((await request.json()) as Record<string, unknown>);
      const next = adapters[Math.min(started, adapters.length - 1)];
      started += 1;
      return HttpResponse.json(
        { job_id: `job-${started}`, adapter_id: next.id },
        { status: 202 }
      );
    }),
    http.post(`${API}/api/adapters/:adapterId/apply/`, ({ params }) => {
      const id = String(params.adapterId);
      calls.applies.push(id);
      appliedAt.set(id, "2026-02-01T01:00:00Z");
      const current = adapters.find((a) => a.id === id) ?? adapters[0];
      return HttpResponse.json({
        ...current,
        applied_at: "2026-02-01T01:00:00Z",
        rerun_advice: {
          include_level: 0.65,
          previous_include_level: 0.5,
          changes_objects: true,
          preserves_manual_work: true,
          summary: "",
          preservation: "",
        },
      });
    }),
    http.post(`${API}/api/segmentations/:segId/apply-full-image/`, ({ params }) => {
      calls.reruns.push(String(params.segId));
      return HttpResponse.json({ job_id: "job-rerun" }, { status: 202 });
    }),
    http.get(`${API}/api/jobs/:jobId/`, ({ params }) =>
      HttpResponse.json({
        id: String(params.jobId),
        type: "train_organelle_adapter",
        status: "SUCCESS",
        progress: 100,
        message: "adaptation complete",
        cancel_requested: false,
        created_at: "2026-02-01T00:00:00Z",
        updated_at: "2026-02-01T00:00:01Z",
        result_json: {},
      })
    )
  );
  return calls;
}

function renderPanel() {
  return render(
    <MemoryRouter initialEntries={[`/assets/${ASSET_ID}/improve?seg=${SEG}`]}>
      <Routes>
        <Route path="/assets/:assetId/improve" element={<ImprovePanel />} />
      </Routes>
    </MemoryRouter>
  );
}

describe("ImprovePanel", () => {
  it("says what it will learn from, names the other image, and promises nothing changes", async () => {
    installHandlers();
    renderPanel();

    await screen.findByText(
      "I'll look at the 2 areas you've marked as checked — 1 on this image and " +
        "1 on Grid2 Cell11 — and match my cut-off to what you kept in them."
    );
    // The invariant, stated before the button rather than only after — beside
    // both rungs, because both are model passes.
    expect(
      screen.getAllByText(/Nothing you've drawn or kept will change/)
    ).toHaveLength(2);
  });

  it("runs end to end from one button and reports in plain language", async () => {
    const user = userEvent.setup();
    const calls = installHandlers();
    renderPanel();

    await user.click(await screen.findByTestId("learn-from-my-fixes"));

    const result = await screen.findByTestId("improve-result");
    expect(calls.starts).toHaveLength(1);
    expect(calls.starts[0].mode).toBe("threshold_only");
    // The number a methods section needs, and the one it replaces.
    expect(
      within(result).getByText("New include level 0.65, where my default is 0.50.")
    ).toBeInTheDocument();
    expect(
      within(result).getByText(
        "At this level I agree with your marks better than my default did."
      )
    ).toBeInTheDocument();
    expect(
      within(result).getByText("I was including a little too much.")
    ).toBeInTheDocument();
    // I-4: the sample size travels in the same sentence as the claim.
    expect(
      within(result).getByText(
        "Checked against 1 checked area on a different image that I did not fit to."
      )
    ).toBeInTheDocument();
    // I-1, visible.
    expect(within(result).getByTestId("preservation-note").textContent).toContain(
      "kept, removed or drawn by hand"
    );
    // No jargon escaped the drawer.
    expect(result.textContent).not.toMatch(/Dice|threshold|adapter/i);
  });

  it("can be pressed a second time, with no browser storage to clear", async () => {
    // The defect: `StepRun.tsx:101` hid the start button for a *succeeded* run,
    // and `adaptRunStorage` brought a reload back to that same finished run.
    const user = userEvent.setup();
    const second = adapter({ id: "ad-2", calibrated_threshold: 0.55 });
    const calls = installHandlers({ adapters: [adapter(), second] });
    renderPanel();

    await user.click(await screen.findByTestId("learn-from-my-fixes"));
    await screen.findByTestId("improve-result");

    const again = await screen.findByTestId("learn-from-my-fixes");
    expect(again).toHaveTextContent("Learn from my fixes again");
    expect(again).toBeEnabled();
    await user.click(again);

    await waitFor(() => expect(calls.starts).toHaveLength(2));
    await waitFor(() =>
      expect(
        screen.getByText("New include level 0.55, where my default is 0.50.")
      ).toBeInTheDocument()
    );
  });

  it("reattaches to a finished run from the server, not from localStorage", async () => {
    installHandlers({ latest: { adapter: adapter(), job_id: null } });
    renderPanel();

    await screen.findByTestId("improve-result");
    // And that reattached run does not lock the button.
    expect(await screen.findByTestId("learn-from-my-fixes")).toBeEnabled();
  });

  it("refuses a run with no probability map before anything is queued", async () => {
    const user = userEvent.setup();
    const reason =
      "No probability map covers the checked area. Run the model on this image first.";
    const calls = installHandlers({
      crops: crops({
        has_probability: false,
        mode_blockers: { threshold_only: [reason], head: [] },
      }),
    });
    renderPanel();

    expect(await screen.findByTestId("threshold-refusal")).toHaveTextContent(reason);
    const button = screen.getByTestId("learn-from-my-fixes");
    expect(button).toBeDisabled();
    await user.click(button);
    expect(calls.starts).toHaveLength(0);

    // The rung that does not need a stored map is still offered.
    expect(screen.getByTestId("train-on-my-fixes")).toBeEnabled();
  });

  it("greys head training with both spans when the checked area is too small", async () => {
    const sentence = "Your checked area is 1.1 µm across; this needs about 1.9 µm.";
    const calls = installHandlers({
      crops: crops({
        mode_blockers: { threshold_only: [], head: [sentence] },
        head_size: {
          base_model: "quantem:mito",
          ok: false,
          largest_nm: 1100,
          required_nm: 1856,
          largest_px: 220,
          required_px: 232,
          n_areas: 1,
          reason: sentence,
        },
      }),
    });
    const user = userEvent.setup();
    renderPanel();

    expect(await screen.findByTestId("head-refusal")).toHaveTextContent(sentence);
    const head = screen.getByTestId("train-on-my-fixes");
    expect(head).toBeDisabled();
    await user.click(head);
    expect(calls.starts).toHaveLength(0);

    // The cheap rung is untouched by a geometry problem.
    expect(screen.getByTestId("learn-from-my-fixes")).toBeEnabled();
  });

  it("keeps every sentence of the statistics layer, one click away", async () => {
    const user = userEvent.setup();
    installHandlers();
    renderPanel();

    await user.click(await screen.findByTestId("learn-from-my-fixes"));
    await screen.findByTestId("improve-result");
    expect(screen.queryByTestId("about-the-numbers-panel")).not.toBeInTheDocument();

    await user.click(screen.getByTestId("about-the-numbers"));

    const drawer = await screen.findByTestId("about-the-numbers-panel");
    // Nothing from `StepResults` was dropped in the move.
    expect(within(drawer).getByText(/Oracle ceiling — not a target/)).toBeInTheDocument();
    expect(within(drawer).getByText("Held-out, chosen threshold")).toBeInTheDocument();
    expect(within(drawer).getAllByText("image-disjoint").length).toBeGreaterThan(0);
    expect(within(drawer).getByText("Threshold sweep")).toBeInTheDocument();
    expect(
      within(drawer).getByText("Read before quoting these numbers")
    ).toBeInTheDocument();
    // The drawer's own signed number, and the plain sentence above it, agree.
    expect(within(drawer).getByText("+0.050")).toBeInTheDocument();
  });

  it("offers the re-run only after the level is in use, and says what survives it", async () => {
    const user = userEvent.setup();
    const calls = installHandlers();
    renderPanel();

    await user.click(await screen.findByTestId("learn-from-my-fixes"));
    await screen.findByTestId("improve-result");
    // Applying is not the re-run: nothing is offered to re-run until it is.
    expect(screen.queryByTestId("find-objects-again")).not.toBeInTheDocument();

    await user.click(screen.getByTestId("use-include-level"));
    await waitFor(() => expect(calls.applies).toEqual(["ad-1"]));

    const rerun = await screen.findByTestId("find-objects-again");
    expect(
      screen.getAllByText(/Only my own guesses are replaced/).length
    ).toBeGreaterThan(0);
    await user.click(rerun);
    await waitFor(() => expect(calls.reruns).toEqual([SEG]));
  });
});
