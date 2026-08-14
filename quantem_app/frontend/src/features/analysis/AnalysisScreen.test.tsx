/**
 * The Analysis screen's account of a run, driven through the screen itself.
 *
 * Three reported disagreements, all on this page:
 *
 * 1. The run-history badge lagged its own message -- a run mid-write showed
 *    `PENDING` beside "writing export bundle", because the badge read the row
 *    the worker last wrote and the message read the live job.
 * 2. One cancel read `CANCELLED` / "You cancelled this run." in the Adapt
 *    wizard and `FAILED` / "This run failed." here.
 * 3. There was no Cancel control on this screen at all. The only one in the
 *    application is in the Library's queue sidebar, which is not reachable
 *    from Analysis.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { Link, MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { AnalysisScreen } from "@/features/analysis/AnalysisScreen";
import { server } from "@/test/msw/server";
import type { AnalysisRunStatus } from "@/shared/types/analysis";
import type { JobStatus } from "@/shared/types/common";

const API = "http://127.0.0.1:8000";
const ASSET_ID = "asset-1";
const SEG_ID = "seg-1";
const RUN_ID = "run-1";
const JOB_ID = "job-1";

const CANCELLED_DETAIL =
  "Cancelled before it finished, so it produced no result. Nothing was saved; " +
  "start it again when you are ready.";

function segmentation() {
  return {
    id: SEG_ID,
    asset: ASSET_ID,
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
    segment_counts: { CONFIRMED: 4, EXCLUDED: 0, INFERRED: 0, CANDIDATE: 0 },
    status_stage: "CANDIDATES_READY",
    status_progress: 100,
    config: { supports_instance_params: true, instance_params: null },
    is_complete: false,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
}

function run(status: AnalysisRunStatus, error = "") {
  return {
    id: RUN_ID,
    segmentation_id: SEG_ID,
    status,
    group: "",
    created_at: "2026-02-01T10:00:00Z",
    started_at: "2026-02-01T10:00:01Z",
    finished_at: null,
    params: {},
    pixel_size_nm: 5,
    calibrated: true,
    composition: null,
    objects: null,
    points: null,
    distances: null,
    monte_carlo: null,
    monte_carlo_self_check: null,
    caveats: [],
    export_dir: "",
    exports: [],
    error,
  };
}

function summary(status: AnalysisRunStatus, error = "") {
  return {
    id: RUN_ID,
    status,
    group: "",
    created_at: "2026-02-01T10:00:00Z",
    started_at: "2026-02-01T10:00:01Z",
    finished_at: null,
    export_dir: "",
    error,
    n_objects: null,
    calibrated: true,
    n_caveats: 0,
  };
}

function job(status: JobStatus, options: { message?: string; cancelRequested?: boolean } = {}) {
  return {
    id: JOB_ID,
    type: "run_analysis",
    priority: "NORMAL",
    status,
    progress: 80,
    message: options.message ?? "writing export bundle",
    created_at: "2026-02-01T10:00:00Z",
    updated_at: "2026-02-01T10:00:05Z",
    attempts: 1,
    max_attempts: 1,
    next_run_at: "2026-02-01T10:00:00Z",
    payload_json: {},
    cancel_requested: options.cancelRequested ?? false,
    resource_class: "cpu",
    queue_name: "default",
    tags: [],
  };
}

interface Scenario {
  runStatus: AnalysisRunStatus;
  runError?: string;
  jobStatus: JobStatus;
  jobCancelRequested?: boolean;
  cancelled?: string[];
  deleted?: string[];
}

function install(scenario: Scenario) {
  server.use(
    http.get(`${API}/api/assets/${ASSET_ID}/`, () =>
      HttpResponse.json({ id: ASSET_ID, display_name: "Liver 01", pixel_size_nm: 5 })
    ),
    http.get(`${API}/api/assets/${ASSET_ID}/segmentations/`, () =>
      HttpResponse.json([segmentation()])
    ),
    http.get(`${API}/api/segmentations/${SEG_ID}/analysis/`, () =>
      HttpResponse.json([summary(scenario.runStatus, scenario.runError)])
    ),
    http.post(`${API}/api/segmentations/${SEG_ID}/analysis/`, () =>
      HttpResponse.json({ job_id: JOB_ID, analysis_run_id: RUN_ID })
    ),
    http.get(`${API}/api/analysis/${RUN_ID}/`, () =>
      HttpResponse.json(run(scenario.runStatus, scenario.runError))
    ),
    http.get(`${API}/api/jobs/${JOB_ID}/`, () =>
      HttpResponse.json(
        job(scenario.jobStatus, { cancelRequested: scenario.jobCancelRequested })
      )
    ),
    http.post(`${API}/api/jobs/${JOB_ID}/cancel/`, () => {
      scenario.cancelled?.push(JOB_ID);
      return HttpResponse.json({ status: "cancel_requested" });
    }),
    http.delete(`${API}/api/jobs/${JOB_ID}/`, () => {
      scenario.deleted?.push(JOB_ID);
      return new HttpResponse(null, { status: 204 });
    })
  );
}

function renderScreen() {
  return render(
    <MemoryRouter initialEntries={[`/assets/${ASSET_ID}/analysis?seg=${SEG_ID}`]}>
      <Routes>
        <Route path="/assets/:assetId/analysis" element={<AnalysisScreen />} />
      </Routes>
    </MemoryRouter>
  );
}

/** Start a run through the form, which is the only way to get a job id. */
async function startRun(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole("button", { name: "Run Analysis" }));
}

describe("AnalysisScreen", () => {
  it("does not offer pixel-size editing on the Analysis page", async () => {
    install({ runStatus: "SUCCESS", jobStatus: "SUCCESS" });
    renderScreen();
    await screen.findByText("Quantitative analysis");
    expect(screen.queryByRole("button", { name: /Edit Pixel Size/i })).not.toBeInTheDocument();
  });

  /**
   * Paper-cut 5: the `?seg=` deep link was honoured only on a full page load.
   * The settle effect kept the first segmentation for the life of the
   * component, so an in-app link to this screen with a different `?seg=` —
   * or back/forward across two analysis URLs — changed the address bar and
   * nothing on the screen.
   */
  it("follows a ?seg= deep link on SPA navigation, not only on full page load", async () => {
    const SEG2_ID = "seg-2";
    install({ runStatus: "SUCCESS", jobStatus: "SUCCESS" });
    server.use(
      http.get(`${API}/api/assets/${ASSET_ID}/segmentations/`, () =>
        HttpResponse.json([
          segmentation(),
          {
            ...segmentation(),
            id: SEG2_ID,
            segmentation_type: {
              ...segmentation().segmentation_type,
              id: "type-2",
              internal_name: "quantem_internal_er",
              short_name: "ER",
              long_name: "Endoplasmic Reticulum",
            },
          },
        ])
      ),
      http.get(`${API}/api/segmentations/${SEG2_ID}/analysis/`, () =>
        HttpResponse.json([])
      )
    );
    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={[`/assets/${ASSET_ID}/analysis?seg=${SEG_ID}`]}>
        <Routes>
          <Route
            path="/assets/:assetId/analysis"
            element={
              <>
                {/* Same route, different ?seg= — the component stays mounted,
                    which is exactly the case that used to be ignored. */}
                <Link to={`/assets/${ASSET_ID}/analysis?seg=${SEG2_ID}`}>
                  Deep link to ER
                </Link>
                <AnalysisScreen />
              </>
            }
          />
        </Routes>
      </MemoryRouter>
    );

    const picker = await screen.findByLabelText("Segmentation");
    await waitFor(() => expect(picker).toHaveValue(SEG_ID));

    await user.click(screen.getByRole("link", { name: "Deep link to ER" }));

    await waitFor(() => expect(picker).toHaveValue(SEG2_ID));
  });

  it("shows the live job status beside the job's own message", async () => {
    // The reported pair: PENDING next to "writing export bundle".
    install({ runStatus: "PENDING", jobStatus: "RUNNING" });
    const user = userEvent.setup();
    renderScreen();
    await startRun(user);

    expect(await screen.findByText("writing export bundle")).toBeInTheDocument();
    const panel = screen.getByRole("heading", { name: "Running" }).parentElement!
      .parentElement!;
    expect(within(panel).getByText("RUNNING")).toBeInTheDocument();
    expect(within(panel).queryByText("PENDING")).not.toBeInTheDocument();
  });

  it("agrees with itself in the history list", async () => {
    install({ runStatus: "PENDING", jobStatus: "RUNNING" });
    const user = userEvent.setup();
    renderScreen();
    await startRun(user);

    await screen.findByText("writing export bundle");
    const history = screen
      .getByRole("heading", { name: "Run history" })
      .closest("div")!.parentElement!;
    await waitFor(() => {
      expect(within(history).getByText("RUNNING")).toBeInTheDocument();
    });
    expect(within(history).queryByText("PENDING")).not.toBeInTheDocument();
  });

  it("calls a cancellation a cancellation, in the wizard's words", async () => {
    install({
      runStatus: "FAILED",
      runError: CANCELLED_DETAIL,
      jobStatus: "CANCELLED",
    });
    const user = userEvent.setup();
    renderScreen();
    await startRun(user);

    // Word for word what StepRun says about the same click.
    expect(
      await screen.findByRole("heading", { name: "You cancelled this run." })
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "This run failed" })
    ).not.toBeInTheDocument();
    expect(screen.getByText(CANCELLED_DETAIL)).toBeInTheDocument();
    expect(screen.getAllByText("CANCELLED").length).toBeGreaterThan(0);
  });

  it("offers a Cancel that stops a running job", async () => {
    const cancelled: string[] = [];
    const deleted: string[] = [];
    install({ runStatus: "PENDING", jobStatus: "RUNNING", cancelled, deleted });
    const user = userEvent.setup();
    renderScreen();
    await startRun(user);

    await user.click(await screen.findByRole("button", { name: "Cancel run" }));

    await waitFor(() => expect(cancelled).toEqual([JOB_ID]));
    // `POST /cancel/` 409s on anything that is not RUNNING; the two exits must
    // not be swapped.
    expect(deleted).toEqual([]);
  });

  it("removes a queued job, which cancel would refuse", async () => {
    const cancelled: string[] = [];
    const deleted: string[] = [];
    install({ runStatus: "PENDING", jobStatus: "PENDING", cancelled, deleted });
    const user = userEvent.setup();
    renderScreen();
    await startRun(user);

    await user.click(await screen.findByRole("button", { name: "Cancel run" }));

    await waitFor(() => expect(deleted).toEqual([JOB_ID]));
    expect(cancelled).toEqual([]);
  });

  it("stops offering Cancel once one has been requested", async () => {
    install({
      runStatus: "PENDING",
      jobStatus: "RUNNING",
      jobCancelRequested: true,
    });
    const user = userEvent.setup();
    renderScreen();
    await startRun(user);

    expect(
      await screen.findByText(/Cancellation requested\./)
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Cancel run" })
    ).not.toBeInTheDocument();
  });

  it("offers no Cancel for a run this screen did not start", async () => {
    // Picked out of the history, so there is no job id in hand and nothing
    // this screen could cancel. A button that cannot act is worse than none.
    install({ runStatus: "PENDING", jobStatus: "RUNNING" });
    const user = userEvent.setup();
    renderScreen();

    const history = screen
      .getByRole("heading", { name: "Run history" })
      .closest("div")!.parentElement!;
    await user.click(await within(history).findByText(/2026/));

    // The row and the panel both read PENDING: with no job in hand there is
    // nothing more current to report, and nothing to stop.
    await waitFor(() => {
      expect(screen.getAllByText("PENDING").length).toBeGreaterThan(0);
    });
    expect(
      screen.queryByRole("button", { name: "Cancel run" })
    ).not.toBeInTheDocument();
  });
});
