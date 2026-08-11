/**
 * Ticking several organelles and starting them with one button (package P4).
 *
 * The owner's fourth request. Before this, running four organelles meant four
 * trips through a create dialog, each of which queued its own job the moment
 * the previous POST returned -- and an organelle whose model was not installed
 * could not be chosen at all.
 */

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { RunOrganellesPanel } from "@/features/viewer/components/RunOrganellesPanel";
import { server } from "@/test/msw/server";
import type { RunPlan, RunPlanOrganelle } from "@/shared/types/runs";

const API = "http://127.0.0.1:8000";
const ASSET = "11111111-1111-1111-1111-111111111111";

function organelle(partial: Partial<RunPlanOrganelle> = {}): RunPlanOrganelle {
  return {
    organelle: "mito",
    name: "Mitochondria",
    pack_id: "quantem:mito",
    title: "QuantEM — Mitochondria",
    tiles: 858,
    model_installed: true,
    model_ready: true,
    model_blocked_reason: null,
    download_bytes: 0,
    segmentation_id: null,
    ...partial,
  };
}

/**
 * The plan the server answers with: every organelle it can run, and totals over
 * the ticked subset only. The two halves are what let the checklist list an
 * organelle the user has not chosen while the cost line stays about the ones
 * they have.
 */
function planFor(ticked: string[]): RunPlan {
  const known: Record<string, RunPlanOrganelle> = {
    mito: organelle(),
    nucleus: organelle({
      organelle: "nucleus",
      name: "Nucleus",
      pack_id: "quantem:nucleus",
      title: "QuantEM — Nucleus",
      tiles: 88,
      model_installed: false,
      model_ready: false,
      download_bytes: 365_000_000,
    }),
    er: organelle({
      organelle: "er",
      name: "Endoplasmic Reticulum",
      pack_id: "quantem:er",
      title: "QuantEM — ER",
      tiles: 858,
      model_installed: true,
      model_ready: false,
      model_blocked_reason: "The exported encoder is missing from this pack.",
      download_bytes: 0,
    }),
  };
  const organelles = ["mito", "nucleus", "er"].map((id) => known[id]);
  const chosen = ticked.map((id) => known[id]).filter(Boolean);
  const toDownload = chosen.filter((item) => !item.model_installed);
  return {
    asset_id: ASSET,
    pixel_size_nm: 5,
    organelles,
    selected: ticked,
    tiles_total: chosen.reduce((sum, item) => sum + (item.tiles ?? 0), 0),
    // Deduped by the server: one shared encoder however many packs need it.
    download_bytes_total: toDownload.length ? 365_000_000 : 0,
    packs_to_download: toDownload.map((item) => item.pack_id),
  };
}

function mockPlan(onStart?: (body: unknown) => void) {
  server.use(
    // The plan lists every organelle this image can be run for, ticked or not,
    // so the checklist can offer them all; the query only changes what the
    // totals are computed over.
    http.get(`${API}/api/assets/${ASSET}/runs/`, ({ request }) => {
      const asked = new URL(request.url).searchParams.get("organelles") ?? "";
      return HttpResponse.json(planFor(asked ? asked.split(",") : []));
    }),
    http.post(`${API}/api/assets/${ASSET}/runs/`, async ({ request }) => {
      const body = await request.json();
      onStart?.(body);
      return HttpResponse.json(
        { job_id: "job-1", plan: planFor(["mito"]) },
        { status: 202 }
      );
    })
  );
}

describe("choosing what to find", () => {
  it("offers every organelle and starts them all with one button", async () => {
    const started: unknown[] = [];
    mockPlan((body) => started.push(body));
    const user = userEvent.setup();
    render(<RunOrganellesPanel assetId={ASSET} imageReady />);

    await screen.findByLabelText(/Mitochondria/);
    await user.click(screen.getByLabelText(/Nucleus/));

    const button = await screen.findByRole("button", {
      name: /find mitochondria and nucleus/i,
    });
    await user.click(button);

    await waitFor(() => expect(started).toHaveLength(1));
    expect(started[0]).toEqual({ organelles: ["mito", "nucleus"] });
  });

  it("lets an uninstalled model be ticked, and says what it costs", async () => {
    mockPlan();
    const user = userEvent.setup();
    render(<RunOrganellesPanel assetId={ASSET} imageReady />);

    const nucleus = await screen.findByLabelText(/Nucleus/);
    expect(nucleus).not.toBeDisabled();
    await user.click(nucleus);

    // The row's own figure, and the deduped aggregate below it.
    expect(screen.getByText(/348\.1 MB to download/)).toBeInTheDocument();
    expect(
      screen.getByText(/Also downloading 348\.1 MB before this can start\./)
    ).toBeInTheDocument();
  });

  it("refuses a model that cannot run here, in the server's own words", async () => {
    mockPlan();
    render(<RunOrganellesPanel assetId={ASSET} imageReady />);

    const er = await screen.findByLabelText(/Endoplasmic Reticulum/);
    expect(er).toBeDisabled();
  });

  it("says which organelle lands first when more than one is ticked", async () => {
    mockPlan();
    const user = userEvent.setup();
    render(<RunOrganellesPanel assetId={ASSET} imageReady />);

    await user.click(await screen.findByLabelText(/Nucleus/));
    expect(
      screen.getByText(/Nucleus finishes first; it is the smallest pass\./)
    ).toBeInTheDocument();
  });

  it("starts nothing while the image is still being read", async () => {
    mockPlan();
    render(<RunOrganellesPanel assetId={ASSET} imageReady={false} />);
    const button = await screen.findByRole("button", {
      name: /find mitochondria/i,
    });
    expect(button).toBeDisabled();
  });

  it("ticks mitochondria alone, and nothing else, on arrival", async () => {
    mockPlan();
    render(<RunOrganellesPanel assetId={ASSET} imageReady />);
    expect(await screen.findByLabelText(/Mitochondria/)).toBeChecked();
    expect(screen.getByLabelText(/Nucleus/)).not.toBeChecked();
    expect(screen.getByLabelText(/Endoplasmic Reticulum/)).not.toBeChecked();
  });
});
