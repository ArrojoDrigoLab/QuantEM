import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";

import { ModelsScreen } from "@/features/models/ModelsScreen";
import type { ModelCatalogue, ModelPack, OrganelleKey } from "@/shared/types/finetune";
import { server } from "@/test/msw/server";

const API = "http://127.0.0.1:8000";
const ORGANELLES: OrganelleKey[] = ["mito", "er", "nucleus", "ld"];

function makePack(family: "quantem" | "omniem", organelle: OrganelleKey): ModelPack {
  return {
    id: `${family}:${organelle}`,
    family,
    organelle,
    title: `${family} ${organelle}`,
    installed: family === "omniem" && organelle === "mito",
    download_bytes: 100,
    canonical_nm: 8,
    tile_size: 512,
    default_threshold: 0.5,
    decoder: "decoder",
    neck: "neck",
    adapt: "adapt",
    licence: "licence",
    notes: "",
    runnable: true,
    reason: null,
  };
}

function catalogue(): ModelCatalogue {
  return {
    packs: ["omniem", "quantem"].flatMap((family) =>
      ORGANELLES.map((organelle) =>
        makePack(family as "omniem" | "quantem", organelle)
      )
    ),
    adapted: [],
    device: { kind: "cpu", name: "CPU", cuda: false, mps: false },
  };
}

function renderScreen() {
  server.use(http.get(`${API}/api/models/`, () => HttpResponse.json(catalogue())));
  return render(
    <MemoryRouter>
      <ModelsScreen />
    </MemoryRouter>
  );
}

describe("ModelsScreen", () => {
  it("shows the two simple model families and no architecture or local-folder UI", async () => {
    renderScreen();

    expect(await screen.findByText("OmniEM (Large model)")).toBeInTheDocument();
    expect(screen.getByText("QuantEM (Basic model)")).toBeInTheDocument();
    expect(
      screen.getByText(/Model weights are not included in the application/i)
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Back to Home" })).toHaveAttribute(
      "href",
      "/"
    );
    expect(screen.queryByText("This machine")).not.toBeInTheDocument();
    expect(screen.queryByText(/architecture/i)).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /local folder/i })
    ).not.toBeInTheDocument();
  });

  it("removes a downloaded pack after confirmation", async () => {
    const user = userEvent.setup();
    let removed = "";
    server.use(
      http.delete(`${API}/api/models/:packId/`, ({ params }) => {
        removed = String(params.packId);
        return new HttpResponse(null, { status: 204 });
      })
    );
    renderScreen();

    await user.click(await screen.findByRole("button", { name: "Remove" }));
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Remove" })
    );

    await waitFor(() => expect(removed).toBe("omniem:mito"));
  });

  it("downloads the selected pack without exposing local-folder installation", async () => {
    const user = userEvent.setup();
    let requested = "";
    server.use(
      http.post(`${API}/api/models/:packId/install/`, ({ params }) => {
        requested = String(params.packId);
        return HttpResponse.json({ status: "PENDING", job_id: "job-1" }, { status: 202 });
      })
    );
    renderScreen();

    await user.click((await screen.findAllByRole("button", { name: "Download" }))[0]);

    await waitFor(() => expect(requested).toBe("omniem:er"));
    expect(screen.queryByRole("button", { name: /local folder/i })).not.toBeInTheDocument();
  });

  it("shows the backend's model-action error on the affected row", async () => {
    const user = userEvent.setup();
    server.use(
      http.post(`${API}/api/models/:packId/install/`, () =>
        HttpResponse.json(
          { error: "The registry digest did not match." },
          { status: 409 }
        )
      )
    );
    renderScreen();

    await user.click((await screen.findAllByRole("button", { name: "Download" }))[0]);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The registry digest did not match."
    );
  });

  it("does not mutate the model cache while any download is active", async () => {
    const activeCatalogue = catalogue();
    activeCatalogue.packs = activeCatalogue.packs.map((pack) =>
      pack.id === "quantem:er"
        ? {
            ...pack,
            active_install: {
              job_id: "job-1",
              status: "RUNNING",
              progress_current_bytes: 50,
              progress_total_bytes: 100,
            },
          }
        : pack
    );
    server.use(
      http.get(`${API}/api/models/`, () => HttpResponse.json(activeCatalogue))
    );
    render(
      <MemoryRouter>
        <ModelsScreen />
      </MemoryRouter>
    );

    expect((await screen.findAllByText("Downloading…")).length).toBeGreaterThan(0);
    screen.getAllByRole("button", { name: /Download|Remove/ }).forEach((button) =>
      expect(button).toBeDisabled()
    );
  });
});
