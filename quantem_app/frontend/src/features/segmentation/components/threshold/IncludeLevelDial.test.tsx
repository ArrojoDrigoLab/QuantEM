import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { delay, http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";

import { server } from "@/test/msw/server";

import { IncludeLevelDial, RunModelButton } from "./IncludeLevelDial";
import { useIncludeLevelControls, type IncludeLevelOptions } from "./useIncludeLevelControls";
import type { IncludeLevelState } from "./api";
import { useThresholdPreviewStore } from "./useThresholdPreviewStore";

const SEG_ID = "11111111-1111-4111-8111-111111111111";
const BASE = "http://127.0.0.1:8000";
const DIAL_URL = `${BASE}/api/segmentations/${SEG_ID}/include-level`;
const CONFIRM_URL = `${BASE}/api/segmentations/${SEG_ID}/confirm-model-output`;

function state(overrides: Partial<IncludeLevelState> = {}): IncludeLevelState {
  return {
    include_level: null,
    default_include_level: 0.5,
    minimum: 0,
    maximum: 1,
    run_version: 1,
    measurement_mode: "objects",
    object_count: 12,
    candidate_count: 0,
    confirmable_candidate_count: 0,
    manual_roi_candidate_count: 0,
    manual_roi_count: 0,
    confirmed_model_count: 0,
    can_move: true,
    detail: "",
    ...overrides,
  };
}

function serveState(value: IncludeLevelState) {
  server.use(http.get(DIAL_URL, () => HttpResponse.json(value)));
}

function DialHarness(props: Partial<IncludeLevelOptions>) {
  const controls = useIncludeLevelControls({ segmentationId: SEG_ID, ...props });
  return <>
    <header><RunModelButton controls={controls} /></header>
    <aside><IncludeLevelDial controls={controls} /></aside>
  </>;
}

function renderDial(props: Partial<IncludeLevelOptions> = {}) {
  return render(<DialHarness {...props} />);
}

afterEach(() => {
  useThresholdPreviewStore.getState().clear();
});

describe("IncludeLevelDial", () => {
  it("waits for a slow threshold read before refreshing after a stage change", async () => {
    let release!: () => void;
    const pending = new Promise<void>((resolve) => { release = resolve; });
    let calls = 0;
    let active = 0;
    let maxActive = 0;
    server.use(http.get(DIAL_URL, async () => {
      calls += 1; active += 1; maxActive = Math.max(maxActive, active);
      const first = calls === 1;
      if (first) await pending;
      active -= 1;
      return HttpResponse.json(state({ include_level: first ? 0.2 : 0.7 }));
    }));
    const view = renderDial({ sourceModel: "quantem:mito" });
    await waitFor(() => expect(calls).toBe(1));
    view.rerender(<DialHarness sourceModel="quantem:mito" statusStage="THRESHOLD_READY" />);
    expect(calls).toBe(1);
    await act(async () => { release(); });
    await waitFor(() => expect(screen.getByTestId("include-level-value")).toHaveTextContent("0.70"));
    expect(calls).toBe(2);
    expect(maxActive).toBe(1);
  });

  it("uses Threshold as the compact title and keeps the explanation on hover", async () => {
    serveState(state());

    renderDial();

    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "Threshold" })).toBeInTheDocument()
    );
    expect(screen.getByRole("img")).toHaveAccessibleName("Probability threshold for objects. The default threshold for this model is 0.50.");
    expect(screen.getByRole("slider")).toHaveAccessibleName("Threshold");
  });

  it("places only the slider and action below the title in the normal state", async () => {
    serveState(state({ include_level: 0.32 }));

    const { container } = renderDial();
    await screen.findByTestId("include-level-value");

    expect(screen.getByTestId("include-level-value")).toHaveTextContent("0.32");
    expect(screen.queryByTestId("include-level-apply")).not.toBeInTheDocument();
    expect(container.textContent).not.toMatch(/model does not run|objects at this/i);
  });

  it("shows the fine-tune's calculated threshold and identifies its adapter", async () => {
    const queries: string[] = [];
    server.use(
      http.get(DIAL_URL, ({ request }) => {
        queries.push(new URL(request.url).search);
        return HttpResponse.json(state({ default_include_level: 0.25 }));
      })
    );

    renderDial({ sourceModel: "omniem:mito", adapterId: "adapter-25" });

    await waitFor(() =>
      expect(screen.getByTestId("include-level-value")).toHaveTextContent("0.25")
    );
    expect(queries.some((query) => query.includes("adapter_id=adapter-25"))).toBe(true);
  });

  it("queues nothing while the slider is being dragged", async () => {
    serveState(state({ include_level: 0.5 }));
    const posted = vi.fn();
    server.use(
      http.post(DIAL_URL, () => {
        posted();
        return HttpResponse.json({ job_id: "job-1", include_level: 0.2 });
      })
    );

    renderDial();
    const slider = await screen.findByRole("slider");
    for (const value of [0.45, 0.4, 0.35, 0.3, 0.25, 0.2]) {
      fireEvent.change(slider, { target: { value: String(value) } });
    }

    expect(screen.getByTestId("include-level-value")).toHaveTextContent("0.50");
    expect(screen.getByTestId("include-level-new-value")).toHaveTextContent("0.20");
    expect(screen.getByRole("button", { name: "Recalculate at threshold 0.20" })).toBeEnabled();
    expect(posted).not.toHaveBeenCalled();
  });

  it("queues one job when the selected threshold is applied", async () => {
    serveState(state({ include_level: 0.5 }));
    const bodies: unknown[] = [];
    server.use(
      http.post(DIAL_URL, async ({ request }) => {
        bodies.push(await request.json());
        return HttpResponse.json({ job_id: "job-1", include_level: 0.2 });
      }),
      http.get(`${BASE}/api/jobs/job-1/`, () =>
        HttpResponse.json({ id: "job-1", status: "PENDING" })
      )
    );

    renderDial();
    const slider = await screen.findByRole("slider");
    fireEvent.change(slider, { target: { value: "0.2" } });
    await userEvent.click(screen.getByTestId("include-level-apply"));

    await waitFor(() => expect(bodies).toEqual([{ include_level: 0.2 }]));
  });

  it("refreshes the screen once re-extraction succeeds", async () => {
    let currentThreshold = 0.5;
    const onReextracted = vi.fn();
    server.use(
      http.get(DIAL_URL, () => HttpResponse.json(state({ include_level: currentThreshold }))),
      http.post(DIAL_URL, () => {
        currentThreshold = 0.48;
        return HttpResponse.json({ job_id: "job-1", include_level: 0.48 });
      }),
      http.get(`${BASE}/api/jobs/job-1/`, () =>
        HttpResponse.json({ id: "job-1", status: "SUCCESS" })
      )
    );

    renderDial({ onReextracted });
    fireEvent.change(await screen.findByRole("slider"), { target: { value: "0.48" } });
    useThresholdPreviewStore.getState().setOverlay({
      probData: new Uint8ClampedArray(4),
      width: 1,
      height: 1,
      bounds: [0, 0, 1, 1],
      color: [239, 68, 68],
      sourceModel: "quantem:mito",
    });
    await userEvent.click(screen.getByTestId("include-level-apply"));

    await waitFor(() => expect(onReextracted).toHaveBeenCalled());
    await waitFor(() =>
      expect(useThresholdPreviewStore.getState().overlay).toBeNull()
    );
    expect(screen.getByTestId("include-level-value")).toHaveTextContent("0.48");
    expect(screen.queryByTestId("include-level-new-value")).not.toBeInTheDocument();
    expect(screen.queryByTestId("include-level-apply")).not.toBeInTheDocument();
  });

  it("shows Run model only in the header and keeps the threshold explanation on hover", async () => {
    serveState(
      state({
        can_move: false,
        detail: "No stored probability map is available for this image.",
        error_code: "probability_map_missing",
      })
    );

    renderDial();

    const help = await screen.findByRole("img");
    expect(help).toHaveAccessibleName(
      "Probability threshold for objects. Run the model first to view results and adjust this value."
    );
    expect(screen.queryByText(/No stored probability map/)).not.toBeInTheDocument();
    expect(within(screen.getByRole("banner")).getByRole("button", { name: "Run model" })).toBeEnabled();
    expect(within(screen.getByRole("complementary")).queryByRole("button", { name: "Run model" })).not.toBeInTheDocument();
  });

  it("allows the first saved result to be applied at the model default", async () => {
    serveState(state({ include_level: null, default_include_level: 0.41 }));
    const bodies: unknown[] = [];
    server.use(
      http.post(DIAL_URL, async ({ request }) => {
        bodies.push(await request.json());
        return HttpResponse.json({ job_id: "job-default", include_level: 0.41 });
      }),
      http.get(`${BASE}/api/jobs/job-default/`, () =>
        HttpResponse.json({ id: "job-default", status: "PENDING" })
      )
    );

    renderDial();
    const previewButton = await screen.findByRole("button", {
      name: "Preview / Process",
    });
    await userEvent.click(previewButton);

    await waitFor(() => expect(bodies).toEqual([{ include_level: 0.41 }]));
  });

  it("renames Preview / Process to Confirm and confirms the selected model across the image", async () => {
    let current = state({
      include_level: 0.35,
      candidate_count: 4,
      confirmable_candidate_count: 2,
      manual_roi_candidate_count: 2,
      manual_roi_count: 1,
    });
    const confirmBodies: unknown[] = [];
    const onReextracted = vi.fn();
    const onConfirmStarted = vi.fn();
    const onConfirmCommitted = vi.fn();
    server.use(
      http.get(DIAL_URL, () => HttpResponse.json(current)),
      http.post(CONFIRM_URL, async ({ request }) => {
        confirmBodies.push(await request.json());
        current = state({
          include_level: 0.35,
          candidate_count: 2,
          confirmable_candidate_count: 0,
          manual_roi_candidate_count: 2,
          manual_roi_count: 1,
          confirmed_model_count: 2,
        });
        return HttpResponse.json({
          segmentation_id: SEG_ID,
          source_model: "quantem:mito",
          confirmed_count: 2,
          skipped_manual_roi_count: 2,
          manual_roi_count: 1,
          remaining_candidate_count: 2,
          overlay: {
            desired_revision: 7,
            applied_revision: 6,
            sync_applied: false,
            rebuild_mode: "async_full",
            source_model: "quantem:mito",
          },
        });
      })
    );

    renderDial({
      sourceModel: "quantem:mito",
      onReextracted,
      onConfirmStarted,
      onConfirmCommitted,
    });

    expect(await screen.findByRole("button", { name: "Confirm" })).toBeEnabled();
    expect(screen.getByText(/2 candidates inside manually annotated ROIs/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Confirm" }));

    await waitFor(() =>
      expect(confirmBodies).toEqual([{ source_model: "quantem:mito" }])
    );
    expect(onConfirmStarted).toHaveBeenCalledTimes(1);
    expect(onConfirmCommitted).toHaveBeenCalledWith(
      expect.objectContaining({ desired_revision: 7, source_model: "quantem:mito" })
    );
    await waitFor(() => expect(onReextracted).toHaveBeenCalled());
    expect(
      await screen.findByText(/Confirmed 2 model objects for analysis/i)
    ).toBeInTheDocument();
    expect(screen.queryByTestId("include-level-apply")).not.toBeInTheDocument();
  });

  it("downloads an organelle model before running it when none is installed", async () => {
    serveState(
      state({
        can_move: false,
        error_code: "probability_map_missing",
        detail: "No stored result is available.",
      })
    );
    let catalogueCalls = 0;
    const installCalls = vi.fn();
    const runBodies: unknown[] = [];
    server.use(
      http.get(`${BASE}/api/models/`, () => {
        catalogueCalls += 1;
        const installed = catalogueCalls > 1;
        return HttpResponse.json({
          packs: [
            {
              id: "quantem:mito",
              family: "quantem",
              organelle: "mito",
              title: "QuantEM Mitochondria",
              installed,
              download_bytes: 10,
              canonical_nm: 8,
              tile_size: 512,
              default_threshold: 0.5,
              decoder: "affinity",
              neck: "naive",
              adapt: "last4",
              licence: "MIT",
              notes: "",
              runnable: installed,
              reason: installed ? null : "Not installed yet.",
            },
          ],
          adapted: [],
          device: null,
        });
      }),
      http.post(/\/api\/models\/quantem(?::|%3A)mito\/install\/$/, () => {
        installCalls();
        return HttpResponse.json({ job_id: "install-job" });
      }),
      http.get(`${BASE}/api/jobs/install-job/`, () =>
        HttpResponse.json({ id: "install-job", status: "SUCCESS" })
      ),
      http.post(
        `${BASE}/api/segmentations/${SEG_ID}/apply-full-image/`,
        async ({ request }) => {
          runBodies.push(await request.json());
          return HttpResponse.json({ job_id: "run-job" });
        }
      ),
      http.get(`${BASE}/api/jobs/run-job/`, () =>
        HttpResponse.json({ id: "run-job", status: "PENDING" })
      )
    );
    const onSourceModelChange = vi.fn();

    renderDial({
      sourceModel: "quantem:mito",
      segmentationInternalName: "quantem_internal_mito",
      onSourceModelChange,
    });
    await userEvent.click(await screen.findByRole("button", { name: "Run model" }));

    await waitFor(() => expect(installCalls).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(runBodies).toEqual([{ source_model: "quantem:mito" }]));
    expect(onSourceModelChange).toHaveBeenCalledWith("quantem:mito");
  });

  it("hides recalculation when the slider returns to the saved threshold", async () => {
    serveState(state({ include_level: 0.65 }));
    renderDial();
    const slider = await screen.findByRole("slider");
    fireEvent.change(slider, { target: { value: "0.4" } });
    expect(screen.getByRole("button", { name: "Recalculate at threshold 0.40" })).toBeEnabled();
    fireEvent.change(slider, { target: { value: "0.65" } });
    expect(screen.queryByTestId("include-level-apply")).not.toBeInTheDocument();
    expect(screen.queryByTestId("include-level-new-value")).not.toBeInTheDocument();
  });

  it("does not offer another model run for a successful empty result", async () => {
    serveState(state({ include_level: 0.5, object_count: 0 }));
    renderDial();
    expect(await screen.findByRole("slider")).toBeEnabled();
    expect(screen.queryByRole("button", { name: "Run model" })).not.toBeInTheDocument();
  });

  it("updates availability and the default threshold when switching fine-tuned models", async () => {
    server.use(http.get(DIAL_URL, ({ request }) => {
      const hasResult = new URL(request.url).searchParams.get("adapter_id") === "adapter-a";
      return HttpResponse.json(state({ can_move: hasResult, default_include_level: hasResult ? 0.65 : 0.35 }));
    }));
    const { rerender } = renderDial({ sourceModel: "quantem:mito", adapterId: "adapter-a" });
    await waitFor(() => expect(screen.getByRole("slider")).toBeEnabled());
    expect(screen.getByRole("img")).toHaveAccessibleName("Probability threshold for objects. The default threshold for this model is 0.65.");
    expect(screen.queryByRole("button", { name: "Run model" })).not.toBeInTheDocument();
    rerender(<DialHarness sourceModel="quantem:mito" adapterId="adapter-b" />);
    expect(await screen.findByRole("button", { name: "Run model" })).toBeEnabled();
    expect(screen.getByRole("slider")).toBeDisabled();
    expect(screen.getByRole("img")).toHaveAccessibleName("Probability threshold for objects. Run the model first to view results and adjust this value.");
  });

  it("keeps a tested ROI in the preview requests", async () => {
    const roiId = "22222222-2222-4222-8222-222222222222";
    const getQueries: string[] = [];
    const applyBodies: unknown[] = [];
    server.use(
      http.get(DIAL_URL, ({ request }) => {
        getQueries.push(new URL(request.url).search);
        return HttpResponse.json(state({ include_level: 0.5 }));
      }),
      http.post(DIAL_URL, async ({ request }) => {
        applyBodies.push(await request.json());
        return HttpResponse.json({ job_id: "roi-apply", include_level: 0.3 });
      }),
      http.get(`${BASE}/api/jobs/roi-apply/`, () =>
        HttpResponse.json({ id: "roi-apply", status: "PENDING" })
      )
    );

    renderDial({ sourceModel: "quantem:mito", roiId });
    const slider = await screen.findByRole("slider");
    fireEvent.change(slider, { target: { value: "0.3" } });
    await userEvent.click(screen.getByTestId("include-level-apply"));

    await waitFor(() =>
      expect(getQueries.some((query) => query.includes(`roi_id=${roiId}`))).toBe(true)
    );
    await waitFor(() =>
      expect(applyBodies).toEqual([
        { include_level: 0.3, source_model: "quantem:mito", roi_id: roiId },
      ])
    );
  });

  it("runs the selected fine-tuned model and removes Run model when its result arrives", async () => {
    let hasResult = false;
    const runBodies: unknown[] = [];
    server.use(
      http.get(DIAL_URL, () => HttpResponse.json(state({ can_move: hasResult, default_include_level: 0.65 }))),
      http.get(`${BASE}/api/models/`, () => HttpResponse.json({
        packs: [{ id: "quantem:mito", organelle: "mito", installed: true, runnable: true }], adapted: [], device: null,
      })),
      http.post(`${BASE}/api/segmentations/${SEG_ID}/apply-full-image/`, async ({ request }) => {
        runBodies.push(await request.json());
        hasResult = true;
        return HttpResponse.json({ job_id: "run-adapted" });
      }),
      http.get(`${BASE}/api/jobs/run-adapted/`, () => HttpResponse.json({ id: "run-adapted", status: "SUCCESS" }))
    );
    renderDial({ sourceModel: "quantem:mito", adapterId: "adapter-a" });
    await userEvent.click(await screen.findByRole("button", { name: "Run model" }));
    await waitFor(() => expect(screen.getByRole("slider")).toBeEnabled());
    expect(runBodies).toEqual([{ source_model: "quantem:mito", adapter_id: "adapter-a" }]);
    expect(screen.queryByRole("button", { name: /Run model|Running model/ })).not.toBeInTheDocument();
    expect(screen.getByRole("img")).toHaveAccessibleName("Probability threshold for objects. The default threshold for this model is 0.65.");
  });

  it("ignores a delayed response for a previously selected model", async () => {
    let firstRequested = false;
    let firstReturned = false;
    server.use(http.get(DIAL_URL, async ({ request }) => {
      if (new URL(request.url).searchParams.get("adapter_id") === "adapter-a") {
        firstRequested = true;
        await delay(150);
        firstReturned = true;
        return HttpResponse.json(state({ can_move: true, default_include_level: 0.65 }));
      }
      return HttpResponse.json(state({ can_move: false, default_include_level: 0.35 }));
    }));
    const { rerender } = renderDial({ sourceModel: "quantem:mito", adapterId: "adapter-a" });
    await waitFor(() => expect(firstRequested).toBe(true));
    rerender(<DialHarness sourceModel="quantem:mito" adapterId="adapter-b" />);
    await screen.findByRole("button", { name: "Run model" });
    await waitFor(() => expect(firstReturned).toBe(true));
    expect(screen.getByRole("slider")).toBeDisabled();
    expect(screen.getByTestId("include-level-value")).toHaveTextContent("0.35");
  });
});

it("keeps a re-extraction pending through RETRY and refreshes only after success", async () => {
  let status = "RETRY";
  let observed = false;
  const refreshed = vi.fn();
  serveState(state({ include_level: 0.5 }));
  server.use(
    http.post(DIAL_URL, () => HttpResponse.json({ job_id: "retry-job", include_level: 0.2 })),
    http.get(`${BASE}/api/jobs/retry-job/`, () => { observed = true; return HttpResponse.json({ id: "retry-job", status }); }),
  );
  renderDial({ onReextracted: refreshed });
  fireEvent.change(await screen.findByRole("slider"), { target: { value: "0.2" } });
  await userEvent.click(screen.getByTestId("include-level-apply"));
  await waitFor(() => expect(observed).toBe(true));
  expect(screen.getByRole("slider")).toBeDisabled();
  expect(refreshed).not.toHaveBeenCalled();
  status = "SUCCESS";
  await waitFor(() => expect(refreshed).toHaveBeenCalledOnce(), { timeout: 2500 });
});
