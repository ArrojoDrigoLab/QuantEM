/**
 * The wizard's memory, and what it does with it.
 *
 * Everything here is a regression on one report: a completed head training was
 * unreachable after a reload. Step 4 promises "The run goes on the job queue,
 * so you can leave this screen and come back to it" and that was false --
 * wizard state was in memory, so a refresh disabled steps 2-6, left no adapter
 * list, no Results and no Apply, and the models screen listed the trained
 * adapter read-only. Alongside it: the base-model preselect ignored the
 * segmentation the route named, and the auto-name froze at whatever pack was
 * selected when it first ran.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { Link, MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it } from "vitest";
import { AdaptWizard } from "@/features/finetune/AdaptWizard";
import { loadAdaptRun, rememberAdaptRun } from "@/features/finetune/adaptRunStorage";
import { server } from "@/test/msw/server";
import type {
  AdaptedModelEntry,
  Adapter,
  AdaptCropsResponse,
  ModelCatalogue,
  ModelPack,
} from "@/shared/types/finetune";

const API = "http://127.0.0.1:8000";
const ASSET_ID = "asset-1";
const MITO_SEG = "seg-mito";
const ER_SEG = "seg-er";
const ADAPTER_ID = "ad-1";

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

// Ordered so the ER pack is NOT first: the preselect used to fall through to
// `choices[0]` before the segmentation list arrived, and that is the bug.
const PACKS = [
  pack("quantem:mito", "mito", "QuantEM — Mitochondria"),
  pack("quantem:er", "er", "QuantEM — Endoplasmic Reticulum"),
];

function segmentation(id: string, internal: string, long: string) {
  return {
    id,
    asset: ASSET_ID,
    segmentation_type: {
      id: `type-${id}`,
      internal_name: internal,
      short_name: long,
      long_name: long,
      default_color: null,
      tags: [],
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    },
    segment_counts: { CONFIRMED: 4, EXCLUDED: 0, INFERRED: 0, CANDIDATE: 0 },
    status_stage: "CANDIDATES_READY",
    status_progress: 100,
    config: { supports_instance_params: true, instance_params: null },
    is_complete: false,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
}

const CROPS: AdaptCropsResponse = {
  crops: [
    {
      id: "crop-1",
      name: "asset-1_0",
      image_key: ASSET_ID,
      segmentation_id: ER_SEG,
      width: 256,
      height: 256,
      n_objects: 12,
      annotated_px: 1000,
      has_probability: true,
    },
  ],
  split_mode: "within-image",
  n_images: 1,
  ready: true,
  blockers: [],
  warnings: [],
  train_crop_names: ["asset-1_0"],
  heldout_crop_names: [],
  modes: ["threshold_only", "head"],
};

function adapter(overrides: Partial<Adapter> = {}): Adapter {
  return {
    id: ADAPTER_ID,
    base_model: "quantem:er",
    name: "er @ Liver 01",
    status: "SUCCESS",
    mode: "head",
    steps: 300,
    trainable_params: 100,
    segmentation_id: ER_SEG,
    split_mode: "within-image",
    train_crop_names: ["asset-1_0"],
    heldout_crop_names: [],
    sweep: {
      thresholds: [0.4, 0.45, 0.5],
      train_dice: [0.8, 0.9, 0.85],
      calibrated_threshold: 0.45,
      train_dice_at_calibrated: 0.9,
      train_dice_at_default: 0.85,
      heldout_dice_at_calibrated: 0.83,
      heldout_dice_at_default: 0.82,
      heldout_oracle: 0.84,
      improvement: 0.01,
      per_crop: { "asset-1_0": 0.83 },
      train_crop_names: ["asset-1_0"],
      heldout_crop_names: [],
    },
    calibrated_threshold: 0.45,
    heldout_dice: 0.83,
    verified_reload: true,
    train_seconds: 60,
    applied_at: null,
    created_at: "2026-02-01T00:00:00Z",
    error: "",
    caveats: [],
    ...overrides,
  };
}

function catalogue(adapted: AdaptedModelEntry[] = []): ModelCatalogue {
  return {
    packs: PACKS,
    adapted,
    device: { kind: "cpu", name: "CPU", cuda: false, mps: false },
  };
}

/**
 * Base handlers. The catalogue answers immediately and the segmentation list
 * answers a tick later, which is the ordering that produced the wrong
 * preselect in the first place.
 */
function installHandlers(options: { catalogue?: ModelCatalogue; adapter?: Adapter } = {}) {
  server.use(
    http.get(`${API}/api/models/`, () =>
      HttpResponse.json(options.catalogue ?? catalogue())
    ),
    http.get(`${API}/api/assets/${ASSET_ID}/`, () =>
      HttpResponse.json({ id: ASSET_ID, display_name: "Liver 01" })
    ),
    http.get(`${API}/api/assets/${ASSET_ID}/segmentations/`, async () => {
      await new Promise((resolve) => setTimeout(resolve, 10));
      return HttpResponse.json([
        segmentation(MITO_SEG, "quantem_internal_mito", "Mitochondria"),
        segmentation(ER_SEG, "quantem_internal_er", "Endoplasmic Reticulum"),
      ]);
    }),
    http.get(`${API}/api/segmentations/:segId/adapt/crops/`, () =>
      HttpResponse.json(CROPS)
    ),
    http.get(`${API}/api/segmentations/:segId/segments/`, () => HttpResponse.json([])),
    http.get(`${API}/api/segmentations/:segId/completed-rois/`, () =>
      HttpResponse.json([])
    ),
    http.get(`${API}/api/adapters/${ADAPTER_ID}/`, () =>
      HttpResponse.json(options.adapter ?? adapter())
    )
  );
}

function renderWizard(search = "") {
  return render(
    <MemoryRouter initialEntries={[`/assets/${ASSET_ID}/adapt${search}`]}>
      <Routes>
        <Route path="/assets/:assetId/adapt" element={<AdaptWizard />} />
      </Routes>
    </MemoryRouter>
  );
}

/**
 * The base-model card the wizard has selected, by its cyan selected border.
 * Scoped to the pack cards -- the current step pill wears the same border.
 */
function selectedBaseModel(): string | null {
  const card = Array.from(
    document.querySelectorAll<HTMLButtonElement>("button.border-cyan-500")
  ).find((button) => button.textContent?.includes("QuantEM —"));
  return card?.textContent ?? null;
}

describe("AdaptWizard", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("preselects the base model for the segmentation named by ?seg=", async () => {
    // The catalogue answers before the segmentation list does. The preselect
    // used to run once against no organelle, fall through to choices[0]
    // (QuantEM — Mitochondria) and freeze there -- which is how an ER
    // adaptation ended up recorded as "mito @ <image>".
    installHandlers();
    renderWizard(`?seg=${ER_SEG}`);

    await waitFor(() => {
      expect(selectedBaseModel()).toContain("QuantEM — Endoplasmic Reticulum");
    });
  });

  /**
   * Paper-cut 5: `?seg=` was honoured only on a full page load. The settle
   * effect kept the first value for the life of the component, so SPA
   * navigation to this route with a different `?seg=` changed the address bar
   * and left the whole wizard — picker, base model, step ladder — on the old
   * segmentation.
   */
  it("follows a ?seg= deep link on SPA navigation, resetting to the new segmentation", async () => {
    const user = userEvent.setup();
    installHandlers();
    render(
      <MemoryRouter initialEntries={[`/assets/${ASSET_ID}/adapt?seg=${ER_SEG}`]}>
        <Routes>
          <Route
            path="/assets/:assetId/adapt"
            element={
              <>
                {/* Same route, different ?seg= — the wizard stays mounted. */}
                <Link to={`/assets/${ASSET_ID}/adapt?seg=${MITO_SEG}`}>
                  Deep link to Mitochondria
                </Link>
                <AdaptWizard />
              </>
            }
          />
        </Routes>
      </MemoryRouter>
    );

    const picker = await screen.findByLabelText("Segmentation");
    await waitFor(() => expect(picker).toHaveValue(ER_SEG));
    await waitFor(() => {
      expect(selectedBaseModel()).toContain("QuantEM — Endoplasmic Reticulum");
    });

    await user.click(
      screen.getByRole("link", { name: "Deep link to Mitochondria" })
    );

    await waitFor(() => expect(picker).toHaveValue(MITO_SEG));
    // The preselect follows the new organelle, exactly as the picker's own
    // handler makes it do.
    await waitFor(() => {
      expect(selectedBaseModel()).toContain("QuantEM — Mitochondria");
    });
  });

  it("keeps the auto-name in step with the base model", async () => {
    const user = userEvent.setup();
    installHandlers();
    renderWizard(`?seg=${ER_SEG}`);

    await waitFor(() => {
      expect(selectedBaseModel()).toContain("QuantEM — Endoplasmic Reticulum");
    });

    // Pick a different pack, then walk to the budget step and read the name.
    await user.click(screen.getByRole("button", { name: /QuantEM — Mitochondria/ }));
    await user.click(await screen.findByRole("button", { name: "Next" }));
    await user.click(await screen.findByRole("button", { name: "Next" }));

    expect(await screen.findByLabelText("Name")).toHaveValue("mito @ Liver 01");
  });

  it("stops rewriting the name once the user has typed one", async () => {
    const user = userEvent.setup();
    installHandlers();
    renderWizard(`?seg=${ER_SEG}`);

    await waitFor(() => {
      expect(selectedBaseModel()).toContain("QuantEM — Endoplasmic Reticulum");
    });
    await user.click(await screen.findByRole("button", { name: "Next" }));
    await user.click(await screen.findByRole("button", { name: "Next" }));

    const name = await screen.findByLabelText("Name");
    await user.clear(name);
    await user.type(name, "my run");

    // Go back and change the pack; the typed name must survive.
    await user.click(screen.getByRole("button", { name: "1. Base model" }));
    await user.click(await screen.findByRole("button", { name: /QuantEM — Mitochondria/ }));
    await user.click(await screen.findByRole("button", { name: "Next" }));
    await user.click(await screen.findByRole("button", { name: "Next" }));

    expect(await screen.findByLabelText("Name")).toHaveValue("my run");
  });

  it("reattaches to a run remembered from a previous session", async () => {
    // The whole point: a reload used to leave steps 2-6 disabled with no route
    // to Results or Apply, so a head training was simply lost.
    rememberAdaptRun(ER_SEG, { adapterId: ADAPTER_ID, jobId: null });
    installHandlers();
    renderWizard(`?seg=${ER_SEG}`);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "5. Results" })).toBeEnabled();
    });
    expect(screen.getByRole("button", { name: "6. Apply" })).toBeEnabled();
  });

  /**
   * The Run step used to leave the wizard on RUNNING forever when the job poll
   * missed the transition -- the reported case sat on "step 300/300 ETA ~0s"
   * with the job already at SUCCESS. Nothing but a job transition ever moved
   * it, so here there is no job at all and the adapter alone has to do it.
   */
  it("leaves the Run step when the adapter says the run finished", async () => {
    rememberAdaptRun(ER_SEG, { adapterId: ADAPTER_ID, jobId: null });
    installHandlers();
    renderWizard(`?seg=${ER_SEG}`);

    // The Results page, which the Run step never handed over to.
    expect(
      await screen.findByRole("heading", { name: "er @ Liver 01" })
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Start adaptation" })
    ).not.toBeInTheDocument();
  });

  /**
   * The dead end: reload during a head training, click back to "What to fit".
   *
   * The step showed "Calibrate the threshold" selected -- the default, not the
   * run -- while a head training was actually going, and the enabled "Go to
   * run" beside it pointed at the other kind of run entirely. Reported as "a
   * trap for exactly the person who reloads because they are unsure what is
   * happening".
   */
  it("shows the running adaptation's own mode when stepping back to What to fit", async () => {
    const user = userEvent.setup();
    rememberAdaptRun(ER_SEG, { adapterId: ADAPTER_ID, jobId: null });
    installHandlers({ adapter: adapter({ status: "RUNNING" }) });
    renderWizard(`?seg=${ER_SEG}`);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "4. Run" })).toBeEnabled();
    });
    await user.click(screen.getByRole("button", { name: "3. What to fit" }));

    const head = await screen.findByRole("button", { name: /Train the head/ });
    expect(head).toHaveAttribute("aria-pressed", "true");
    expect(
      screen.getByRole("button", { name: /Calibrate the threshold/ })
    ).toHaveAttribute("aria-pressed", "false");
    // And it says so in words, rather than leaving a live-looking control that
    // nothing would act on.
    expect(
      screen.getByText(/is running now, and it was started as/)
    ).toBeInTheDocument();
    expect(head).toBeDisabled();
  });

  it("keeps the steps already reached reachable when stepping back", async () => {
    // Reachability was `number <= step`, so going back to re-read step 3
    // revoked Run, Results and Apply on a run that was still going -- and the
    // only way to get them back was another reload.
    const user = userEvent.setup();
    rememberAdaptRun(ER_SEG, { adapterId: ADAPTER_ID, jobId: null });
    installHandlers();
    renderWizard(`?seg=${ER_SEG}`);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "5. Results" })).toBeEnabled();
    });
    await user.click(screen.getByRole("button", { name: "3. What to fit" }));

    for (const label of ["4. Run", "5. Results", "6. Apply"]) {
      expect(screen.getByRole("button", { name: label })).toBeEnabled();
    }
  });

  it("restores the run's base model and name, not a fresh guess", async () => {
    // The auto-name and the preselect are both allowed to keep changing their
    // minds until the user pins them; a reattached run has to pin them itself
    // or the budget step relabels someone else's training.
    const user = userEvent.setup();
    rememberAdaptRun(MITO_SEG, { adapterId: ADAPTER_ID, jobId: null });
    installHandlers({
      adapter: adapter({ segmentation_id: MITO_SEG, name: "my careful run" }),
    });
    // Deliberately arrive on the *mito* segmentation while the adapter was
    // fitted on quantem:er: the preselect would otherwise say mito.
    renderWizard(`?seg=${MITO_SEG}`);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "4. Run" })).toBeEnabled();
    });
    await user.click(screen.getByRole("button", { name: "3. What to fit" }));

    expect(await screen.findByLabelText("Name")).toHaveValue("my careful run");
    await user.click(screen.getByRole("button", { name: "1. Base model" }));
    await waitFor(() => {
      expect(selectedBaseModel()).toContain("QuantEM — Endoplasmic Reticulum");
    });
  });

  it("lists adaptations already run on this segmentation, from the server", async () => {
    // Survives a cleared browser store: the catalogue is the source of truth.
    installHandlers({
      catalogue: catalogue([
        {
          id: `adapted:${ADAPTER_ID}`,
          base: "quantem:er",
          name: "er @ Liver 01",
          created_at: "2026-02-01T00:00:00Z",
          calibrated_threshold: 0.45,
          heldout_dice: 0.83,
          split_mode: "within-image",
          mode: "head",
          segmentation_id: ER_SEG,
          applied_at: null,
        },
      ]),
    });
    renderWizard(`?seg=${ER_SEG}`);

    const panel = (
      await screen.findByRole("heading", {
        name: "Adaptations already run on this segmentation",
      })
    ).closest("div") as HTMLElement;
    expect(within(panel).getByText("er @ Liver 01")).toBeInTheDocument();

    await userEvent.setup().click(
      within(panel).getByRole("button", { name: "Open results" })
    );

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "6. Apply" })).toBeEnabled();
    });
  });

  /**
   * Cancel used to end the wizard for that segmentation.
   *
   * The job went CANCELLED, the `Adapter` stayed RUNNING forever, and step 4
   * rendered a progress bar with a CANCELLED badge and no control at all --
   * steps 5 and 6 gate on `status === "SUCCESS"`, so there was nowhere to go
   * and nothing to press. The backend now concludes the adapter as FAILED
   * carrying `CANCELLED_DETAIL`; this is the other half.
   */
  describe("a cancelled run", () => {
    const CANCELLED_DETAIL =
      "Cancelled before it finished, so it produced no result. Nothing was " +
      "saved; start it again when you are ready.";

    const cancelled = () =>
      adapter({ status: "FAILED", error: CANCELLED_DETAIL, train_seconds: null });

    it("reads as a cancellation, not a crash, and offers a way back", async () => {
      rememberAdaptRun(ER_SEG, { adapterId: ADAPTER_ID, jobId: null });
      installHandlers({ adapter: cancelled() });
      renderWizard(`?seg=${ER_SEG}`);

      expect(
        await screen.findByText("You cancelled this run.")
      ).toBeInTheDocument();
      expect(screen.queryByText("This run failed.")).not.toBeInTheDocument();
      expect(screen.getByText(CANCELLED_DETAIL)).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Start again" })
      ).toBeEnabled();
    });

    /**
     * The second dead end, behind the first. Step 3 goes read-only "because a
     * run exists", which is right while one is live or finished and wrong for
     * one that concluded with nothing: the mode could not be changed, so a
     * cancelled head run on a machine that cannot load the pack left "Start
     * again" permanently disabled with no way to drop to threshold calibration.
     */
    it("hands the mode controls back once the run has concluded", async () => {
      const user = userEvent.setup();
      rememberAdaptRun(ER_SEG, { adapterId: ADAPTER_ID, jobId: null });
      installHandlers({ adapter: cancelled() });
      renderWizard(`?seg=${ER_SEG}`);

      await screen.findByText("You cancelled this run.");
      await user.click(screen.getByRole("button", { name: "3. What to fit" }));

      expect(
        await screen.findByRole("heading", { name: "Choose what to fit" })
      ).toBeInTheDocument();
      expect(
        screen.queryByText(/These settings describe that run/)
      ).not.toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: /Calibrate the threshold/ })
      ).toBeEnabled();
    });

    it("starts a fresh run and forgets the dead one", async () => {
      const started: unknown[] = [];
      rememberAdaptRun(ER_SEG, { adapterId: ADAPTER_ID, jobId: null });
      installHandlers({ adapter: cancelled() });
      server.use(
        http.post(`${API}/api/segmentations/${ER_SEG}/adapt/`, async ({ request }) => {
          started.push(await request.json());
          return HttpResponse.json({ adapter_id: "ad-2", job_id: "job-2" });
        }),
        http.get(`${API}/api/adapters/ad-2/`, () =>
          HttpResponse.json(adapter({ id: "ad-2", status: "RUNNING" }))
        ),
        http.get(`${API}/api/jobs/job-2/`, () =>
          HttpResponse.json({
            id: "job-2",
            type: "train_organelle_adapter",
            status: "RUNNING",
            progress: 5,
            message: "preparing crops",
            created_at: "2026-02-01T00:00:00Z",
            updated_at: "2026-02-01T00:00:00Z",
            attempts: 1,
            max_attempts: 1,
            next_run_at: "2026-02-01T00:00:00Z",
            payload_json: {},
            cancel_requested: false,
            resource_class: "gpu",
            queue_name: "default",
            tags: [],
            priority: "NORMAL",
          })
        )
      );
      renderWizard(`?seg=${ER_SEG}`);

      await userEvent
        .setup()
        .click(await screen.findByRole("button", { name: "Start again" }));

      await waitFor(() => expect(started).toHaveLength(1));
      // The dead run's pointer was dropped, so a reload cannot reattach to it.
      // (It is replaced by the new one, hence `not.toBe(ADAPTER_ID)`.)
      expect(loadAdaptRun(ER_SEG)?.adapterId).not.toBe(ADAPTER_ID);
      await waitFor(() => {
        expect(screen.getByText("preparing crops")).toBeInTheDocument();
      });
    });
  });

  /**
   * The wizard could recover from a cancellation before it could cause one.
   *
   * Step 4 says the run takes "minutes to tens of minutes" and invites you to
   * leave the screen; the only Cancel in the application was in the Library's
   * queue sidebar, which this screen has no route to. Two endpoints, because a
   * queued job and a running one leave by different doors.
   */
  describe("cancelling a run in flight", () => {
    const JOB_ID = "job-live";

    function serveJob(status: "PENDING" | "RUNNING", cancelRequested = false) {
      return http.get(`${API}/api/jobs/${JOB_ID}/`, () =>
        HttpResponse.json({
          id: JOB_ID,
          type: "train_organelle_adapter",
          status,
          progress: 40,
          message: "training the head",
          created_at: "2026-02-01T00:00:00Z",
          updated_at: "2026-02-01T00:00:00Z",
          attempts: 1,
          max_attempts: 1,
          next_run_at: "2026-02-01T00:00:00Z",
          payload_json: {},
          cancel_requested: cancelRequested,
          resource_class: "gpu",
          queue_name: "default",
          tags: [],
          priority: "NORMAL",
        })
      );
    }

    it("cancels a running job through the cancel endpoint", async () => {
      const cancelled: string[] = [];
      const deleted: string[] = [];
      rememberAdaptRun(ER_SEG, { adapterId: ADAPTER_ID, jobId: JOB_ID });
      installHandlers({ adapter: adapter({ status: "RUNNING" }) });
      server.use(
        serveJob("RUNNING"),
        http.post(`${API}/api/jobs/${JOB_ID}/cancel/`, () => {
          cancelled.push(JOB_ID);
          return HttpResponse.json({ status: "cancel_requested" });
        }),
        http.delete(`${API}/api/jobs/${JOB_ID}/`, () => {
          deleted.push(JOB_ID);
          return new HttpResponse(null, { status: 204 });
        })
      );
      renderWizard(`?seg=${ER_SEG}`);

      await userEvent
        .setup()
        .click(await screen.findByRole("button", { name: "Cancel run" }));

      await waitFor(() => expect(cancelled).toEqual([JOB_ID]));
      // `POST /cancel/` 409s on anything that is not RUNNING, so the two paths
      // must not be swapped.
      expect(deleted).toEqual([]);
    });

    it("removes a job that has not started, which cancel would refuse", async () => {
      const cancelled: string[] = [];
      const deleted: string[] = [];
      rememberAdaptRun(ER_SEG, { adapterId: ADAPTER_ID, jobId: JOB_ID });
      installHandlers({ adapter: adapter({ status: "RUNNING" }) });
      server.use(
        serveJob("PENDING"),
        http.post(`${API}/api/jobs/${JOB_ID}/cancel/`, () => {
          cancelled.push(JOB_ID);
          return HttpResponse.json({ status: "cancel_requested" });
        }),
        http.delete(`${API}/api/jobs/${JOB_ID}/`, () => {
          deleted.push(JOB_ID);
          return new HttpResponse(null, { status: 204 });
        })
      );
      renderWizard(`?seg=${ER_SEG}`);

      await userEvent
        .setup()
        .click(await screen.findByRole("button", { name: "Cancel run" }));

      // DELETE is the *only* exit a queued job has.
      await waitFor(() => expect(deleted).toEqual([JOB_ID]));
      expect(cancelled).toEqual([]);
    });

    it("says a cancellation is already on its way, and stops offering it", async () => {
      rememberAdaptRun(ER_SEG, { adapterId: ADAPTER_ID, jobId: JOB_ID });
      installHandlers({ adapter: adapter({ status: "RUNNING" }) });
      server.use(serveJob("RUNNING", true));
      renderWizard(`?seg=${ER_SEG}`);

      expect(
        await screen.findByText(/Cancellation requested\./)
      ).toBeInTheDocument();
      expect(
        screen.queryByRole("button", { name: "Cancel run" })
      ).not.toBeInTheDocument();
    });

    it("offers nothing to cancel on a run that has already concluded", async () => {
      rememberAdaptRun(ER_SEG, { adapterId: ADAPTER_ID, jobId: null });
      installHandlers({ adapter: adapter({ status: "RUNNING" }) });
      renderWizard(`?seg=${ER_SEG}`);

      // Reattached with no job row: the progress bar is all the wizard has, and
      // a button that cannot act is worse than no button.
      await waitFor(() => {
        expect(screen.getByRole("button", { name: "4. Run" })).toBeEnabled();
      });
      expect(
        screen.queryByRole("button", { name: "Cancel run" })
      ).not.toBeInTheDocument();
    });
  });

  it("does not offer another segmentation's adaptations", async () => {
    installHandlers({
      catalogue: catalogue([
        {
          id: "adapted:other",
          base: "quantem:mito",
          name: "mito @ Liver 01",
          created_at: "2026-02-01T00:00:00Z",
          calibrated_threshold: 0.45,
          heldout_dice: 0.83,
          split_mode: "within-image",
          mode: "head",
          segmentation_id: MITO_SEG,
          applied_at: null,
        },
      ]),
    });
    renderWizard(`?seg=${ER_SEG}`);

    await waitFor(() => {
      expect(selectedBaseModel()).toContain("QuantEM — Endoplasmic Reticulum");
    });
    expect(
      screen.queryByRole("heading", {
        name: "Adaptations already run on this segmentation",
      })
    ).not.toBeInTheDocument();
  });
});
