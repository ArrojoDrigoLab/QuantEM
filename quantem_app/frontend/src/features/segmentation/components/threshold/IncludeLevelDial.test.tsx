import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ComponentProps } from "react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";

import { server } from "@/test/msw/server";

import { IncludeLevelDial } from "./IncludeLevelDial";
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

function renderDial(props: Partial<ComponentProps<typeof IncludeLevelDial>> = {}) {
  return render(<IncludeLevelDial segmentationId={SEG_ID} {...props} />);
}

afterEach(() => {
  useThresholdPreviewStore.getState().clear();
});

describe("IncludeLevelDial", () => {
  it("uses Threshold as the compact title and keeps the explanation on hover", async () => {
    serveState(state());

    renderDial();

    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "Threshold" })).toBeInTheDocument()
    );
    expect(screen.getByRole("img")).toHaveAccessibleName(/saved model result/i);
    expect(screen.getByRole("slider")).toHaveAccessibleName("Threshold");
  });

  it("places only the slider and action below the title in the normal state", async () => {
    serveState(state({ include_level: 0.32 }));

    const { container } = renderDial();
    await screen.findByTestId("include-level-value");

    expect(screen.getByTestId("include-level-value")).toHaveTextContent("0.32");
    expect(screen.getByRole("button", { name: "No candidates" })).toBeDisabled();
    expect(container.textContent).not.toMatch(/model does not run|objects at this/i);
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

    expect(screen.getByTestId("include-level-value")).toHaveTextContent("0.20");
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
    serveState(state({ include_level: 0.5 }));
    const onReextracted = vi.fn();
    server.use(
      http.post(DIAL_URL, () =>
        HttpResponse.json({ job_id: "job-1", include_level: 0.48 })
      ),
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
  });

  it("moves the unavailable explanation into a tooltip and shows Run model", async () => {
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
      "No stored result is kept for this image, so the include level cannot be moved without running the model again. Running it once saves one, and the level can be moved freely from then on. Use Run model below, then adjust the threshold."
    );
    expect(screen.queryByText(/No stored probability map/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run model" })).toBeEnabled();
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
    expect(previewButton).toHaveAttribute(
      "title",
      "Turn this model and threshold setting into objects. Normalizes the shapes and applies size thresholds."
    );
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
    expect(screen.getByRole("button", { name: "Confirmed" })).toBeDisabled();
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
});
