/**
 * What the include-level dial must and must not do.
 *
 * Two of these are about restraint rather than function. The dial must not
 * queue work while the user is dragging -- one job rewrites every candidate on
 * the image, and a job per pixel of travel would queue a hundred for one
 * gesture against a one-slot queue. And when it cannot move it must show the
 * server's own sentence, because the two reasons it can be blocked are
 * different futures and a generic "try running again" hides which one the user
 * is in.
 */

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";

import { server } from "@/test/msw/server";

import { IncludeLevelDial } from "./IncludeLevelDial";
import type { IncludeLevelState } from "./api";

const SEG_ID = "11111111-1111-4111-8111-111111111111";
const BASE = "http://127.0.0.1:8000";
const DIAL_URL = `${BASE}/api/segmentations/${SEG_ID}/include-level`;

const NO_MAP_SENTENCE =
  "No stored result is kept for this image, so the include level cannot be " +
  "moved without running the model again. Running it once saves one, and the " +
  "level can be moved freely from then on.";

const LEGACY_SENTENCE =
  "The stored result for this image was written by an earlier version of " +
  "QuantEM, which recorded probabilities differently.";

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

function renderDial(props: Partial<{ onReextracted: () => void }> = {}) {
  return render(<IncludeLevelDial segmentationId={SEG_ID} {...props} />);
}

describe("IncludeLevelDial", () => {
  it("starts at the model's own level when nobody has moved the dial", async () => {
    serveState(state({ include_level: null, default_include_level: 0.5 }));

    renderDial();

    expect(await screen.findByTestId("include-level-value")).toHaveTextContent(
      "0.50"
    );
    expect(screen.getByTestId("include-level-status")).toHaveTextContent(
      "came straight from the model"
    );
  });

  it("shows the level the objects on screen were actually found at", async () => {
    serveState(state({ include_level: 0.32, object_count: 41 }));

    renderDial();

    expect(await screen.findByTestId("include-level-value")).toHaveTextContent(
      "0.32"
    );
    expect(screen.getByTestId("include-level-status")).toHaveTextContent(
      "41 objects"
    );
  });

  it("queues nothing while the slider is being dragged", async () => {
    serveState(state({ include_level: 0.5 }));
    const posted = vi.fn();
    server.use(
      http.post(DIAL_URL, async () => {
        posted();
        return HttpResponse.json({ job_id: "job-1", include_level: 0.2 });
      })
    );

    renderDial();
    const slider = await screen.findByRole("slider");

    // A drag, as the browser delivers one: a change event per step of travel.
    for (const value of [0.45, 0.4, 0.35, 0.3, 0.25, 0.2]) {
      fireEvent.change(slider, { target: { value: String(value) } });
    }

    expect(screen.getByTestId("include-level-value")).toHaveTextContent("0.20");
    expect(posted).not.toHaveBeenCalled();
  });

  it("cannot be applied until the level actually moves", async () => {
    serveState(state({ include_level: 0.5 }));

    renderDial();

    await waitFor(() =>
      expect(screen.getByTestId("include-level-apply")).toBeDisabled()
    );
  });

  it("queues one job when the level is applied", async () => {
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

    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]).toEqual({ include_level: 0.2 });
  });

  it("tells the screen to redraw once the re-extract lands", async () => {
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
    const slider = await screen.findByRole("slider");
    fireEvent.change(slider, { target: { value: "0.48" } });
    await userEvent.click(screen.getByTestId("include-level-apply"));

    await waitFor(() => expect(onReextracted).toHaveBeenCalled());
  });

  describe("when the dial cannot move", () => {
    it("shows the sentence for a result that was never saved", async () => {
      serveState(
        state({
          can_move: false,
          detail: NO_MAP_SENTENCE,
          error_code: "probability_map_missing",
        })
      );

      renderDial();

      const blocked = await screen.findByTestId("include-level-blocked");
      expect(blocked).toHaveTextContent("No stored result is kept");
      expect(blocked).toHaveTextContent("from then on");
      expect(screen.getByTestId("include-level-apply")).toBeDisabled();
    });

    it("shows a different sentence for a result from an older build", async () => {
      serveState(
        state({
          can_move: false,
          detail: LEGACY_SENTENCE,
          error_code: "probability_map_missing",
        })
      );

      renderDial();

      const blocked = await screen.findByTestId("include-level-blocked");
      expect(blocked).toHaveTextContent("earlier version");
      // The one assertion that catches the two cases being collapsed into one
      // generic "run it again": that message belongs to the other case only.
      expect(blocked).not.toHaveTextContent("No stored result is kept");
    });

    it("names a control rather than a request", async () => {
      serveState(state({ can_move: false, detail: NO_MAP_SENTENCE }));

      renderDial();

      const blocked = await screen.findByTestId("include-level-blocked");
      const text = blocked.textContent ?? "";
      // Invariant I-12: no route, no verb, no internal name in copy.
      expect(text).not.toMatch(/\/api\//);
      expect(text).not.toMatch(/\bPOST\b/);
      expect(text).not.toMatch(/reextract_at_include_level/);
      expect(text).toMatch(/labeling header/);
    });
  });

  it("promises preservation, and says the model does not re-run", async () => {
    serveState(state({ include_level: 0.5 }));

    renderDial();

    expect(
      await screen.findByText(/Only my own guesses are redone/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/The model does not run again/)
    ).toBeInTheDocument();
  });

  it("never says the word threshold to the user", async () => {
    serveState(state({ include_level: 0.4 }));

    const { container } = renderDial();
    await screen.findByTestId("include-level-value");

    expect(container.textContent ?? "").not.toMatch(/threshold/i);
  });
});
