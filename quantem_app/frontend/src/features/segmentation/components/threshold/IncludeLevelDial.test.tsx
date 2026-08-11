import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ComponentProps } from "react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";

import { server } from "@/test/msw/server";

import { IncludeLevelDial } from "./IncludeLevelDial";
import type { IncludeLevelState } from "./api";

const SEG_ID = "11111111-1111-4111-8111-111111111111";
const BASE = "http://127.0.0.1:8000";
const DIAL_URL = `${BASE}/api/segmentations/${SEG_ID}/include-level`;

function state(overrides: Partial<IncludeLevelState> = {}): IncludeLevelState {
  return {
    include_level: null,
    default_include_level: 0.5,
    minimum: 0,
    maximum: 1,
    run_version: 1,
    object_count: 12,
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
    expect(screen.getByRole("button", { name: "Apply" })).toBeDisabled();
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
    await userEvent.click(screen.getByTestId("include-level-apply"));

    await waitFor(() => expect(onReextracted).toHaveBeenCalled());
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
      "No stored result is kept for this image, so the include level cannot be moved without running the model again. Running it once saves one, and the level can be moved freely from then on. Run the model on this image from the labeling header, and the threshold can be adjusted afterwards."
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
    await userEvent.click(await screen.findByRole("button", { name: "Apply" }));

    await waitFor(() => expect(bodies).toEqual([{ include_level: 0.41 }]));
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
});
