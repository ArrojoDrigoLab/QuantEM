/**
 * What the models screen claims about installing, and what it lets you do.
 *
 * Both were wrong in a way a user could act on: the panel told them a pack
 * "can only be installed from a copy already on this machine" because nothing
 * could verify a download -- untrue since release bundles, whose MANIFEST.json
 * carries a SHA-256 per file that the installer re-hashes -- and the install
 * help described a maintainer's directory (`head.pt` beside
 * `resolved_config.yaml`, example `D:\models\mito_quantem`) that nobody who
 * downloaded a release has. The adapted list, meanwhile, was read-only: the
 * only Apply button in the product was on step 6 of a wizard you could not get
 * back to.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { ModelsScreen } from "@/features/models/ModelsScreen";
import { server } from "@/test/msw/server";
import { EMPTY_JOB_QUEUE_STATUS } from "@/test/msw/handlers";
import type { Job, JobQueueItem } from "@/shared/types/jobs";
import type { ModelCatalogue, ModelPack } from "@/shared/types/finetune";

const API = "http://127.0.0.1:8000";

/** A download job as `GET /api/jobs/<id>/` reports it. */
function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: "job-dl-1",
    type: "install_model_pack",
    priority: "default",
    status: "RUNNING",
    progress: 0,
    created_at: "2026-02-01T00:00:00Z",
    updated_at: "2026-02-01T00:00:00Z",
    attempts: 1,
    max_attempts: 1,
    next_run_at: "2026-02-01T00:00:00Z",
    payload_json: { pack_id: "quantem:mito" },
    cancel_requested: false,
    resource_class: "cpu",
    queue_name: "default",
    tags: ["model:quantem:mito"],
    ...overrides,
  };
}

/** A terminal job row as `GET /api/jobs/queue-status/` lists it. */
function makeQueueItem(overrides: Partial<JobQueueItem> = {}): JobQueueItem {
  return {
    id: "job-old-1",
    type: "install_model_pack",
    task_label: "Download model pack",
    status: "FAILED",
    progress: 62,
    cancel_requested: false,
    queue_name: "default",
    resource_class: "cpu",
    created_at: "2026-02-01T00:00:00Z",
    started_at: "2026-02-01T00:00:10Z",
    finished_at: "2026-02-01T00:05:00Z",
    image: null,
    segmentation: null,
    ...overrides,
  };
}

function pack(overrides: Partial<ModelPack> = {}): ModelPack {
  return {
    id: "quantem:mito",
    family: "quantem",
    organelle: "mito",
    title: "QuantEM — Mitochondria",
    installed: false,
    download_bytes: 662337373,
    canonical_nm: 8,
    tile_size: 512,
    default_threshold: 0.5,
    decoder: "affinity_mws",
    neck: "naive_1x1",
    adapt: "last_n",
    licence: "see NOTICE",
    notes: "",
    runnable: false,
    reason: "Not installed yet.",
    encoder_tier: null,
    ...overrides,
  };
}

function catalogue(overrides: Partial<ModelCatalogue> = {}): ModelCatalogue {
  return {
    packs: [pack()],
    adapted: [],
    device: { kind: "cpu", name: "CPU", cuda: false, mps: false },
    ...overrides,
  };
}

function renderScreen(body: ModelCatalogue = catalogue()) {
  server.use(http.get(`${API}/api/models/`, () => HttpResponse.json(body)));
  return render(
    <MemoryRouter>
      <ModelsScreen />
    </MemoryRouter>
  );
}

describe("ModelsScreen", () => {
  it("claims verification for both install routes, and no dead ends", async () => {
    renderScreen();

    expect(
      await screen.findByText(/verifies every file against its published digest/i)
    ).toBeInTheDocument();
    // The pre-download copy. A build whose backend still refuses the download
    // reports the server's refusal on the button instead; this panel must not
    // contradict a backend that can.
    expect(
      screen.queryByText(/does not download weights/i)
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/no digests/i)).not.toBeInTheDocument();
  });

  it("describes the folder shapes a release actually unzips to", async () => {
    const user = userEvent.setup();
    renderScreen();

    await user.click(
      await screen.findByRole("button", { name: "Install from a local folder" })
    );

    const input = screen.getByLabelText("Directory on this machine");
    // Owner ruling D8. This placeholder used to be a literal beginning with
    // the drive letter of the machine the release was built on, shown as
    // guidance to every user on every platform. With no `storage` block on the
    // catalogue there is nothing to build an example from, and the field says
    // what it wants in words instead of naming somebody else's disk.
    expect(input).toHaveAttribute(
      "placeholder",
      "the folder you unzipped a QuantEM model release into"
    );
    const shapes = input.parentElement as HTMLElement;
    expect(within(shapes).getByText("packs/quantem__mito/")).toBeInTheDocument();
    expect(within(shapes).getByText("MANIFEST.json")).toBeInTheDocument();
    expect(within(shapes).getByText(/unzipped a QuantEM model release into/)).toBeInTheDocument();
  });

  it("shows an example path built by the server, never one typed into the source", async () => {
    // D8's fix: the server composes the example from its own resolved models
    // directory, so a Mac gets a POSIX path and Windows gets a Windows one,
    // and neither gets the build machine's drive.
    const user = userEvent.setup();
    renderScreen({
      ...catalogue(),
      storage: {
        models_dir: "/Users/somebody/QuantEM/data/models",
        local_source_example: "/Users/somebody/QuantEM/quantem-models",
      },
    } as ModelCatalogue);

    await user.click(
      await screen.findByRole("button", { name: "Install from a local folder" })
    );

    expect(screen.getByLabelText("Directory on this machine")).toHaveAttribute(
      "placeholder",
      "e.g. /Users/somebody/QuantEM/quantem-models"
    );
  });

  it("can apply an adapted model without the wizard", async () => {
    // The screen listed these and offered nothing to do with them.
    const user = userEvent.setup();
    let appliedId: string | null = null;
    server.use(
      http.post(`${API}/api/adapters/:adapterId/apply/`, ({ params }) => {
        appliedId = String(params.adapterId);
        return HttpResponse.json({ id: appliedId });
      })
    );
    renderScreen(
      catalogue({
        adapted: [
          {
            id: "adapted:11111111-2222-3333-4444-555555555555",
            base: "quantem:mito",
            name: "mito @ liver",
            created_at: "2026-02-01T00:00:00Z",
            calibrated_threshold: 0.45,
            heldout_dice: 0.9,
            split_mode: "image-disjoint",
            mode: "head",
            segmentation_id: "seg-1",
            applied_at: null,
          },
        ],
      })
    );

    const row = (await screen.findByText("mito @ liver")).closest(
      "li"
    ) as HTMLElement;
    await user.click(within(row).getByRole("button", { name: "Apply" }));

    // The catalogue prefixes the id; the endpoint takes the bare uuid.
    await waitFor(() => {
      expect(appliedId).toBe("11111111-2222-3333-4444-555555555555");
    });
  });

  it("never shows a score without the split mode that produced it", async () => {
    renderScreen(
      catalogue({
        adapted: [
          {
            id: "adapted:ad-1",
            base: "quantem:mito",
            name: "mito @ liver",
            created_at: "2026-02-01T00:00:00Z",
            calibrated_threshold: 0.45,
            heldout_dice: 0.9,
            split_mode: "within-image",
            mode: "head",
            segmentation_id: "seg-1",
            applied_at: null,
          },
        ],
      })
    );

    const row = (await screen.findByText("mito @ liver")).closest(
      "li"
    ) as HTMLElement;
    expect(within(row).getByText(/Dice 0\.900/)).toBeInTheDocument();
    expect(within(row).getByText(/within-image/)).toBeInTheDocument();
  });

  /**
   * The download flow: `POST install` with no source is a real 202 + job with
   * progress (fraction reported, bytes in the job's message), failure text
   * verbatim from the job, and a cancel that works. An older backend refuses
   * the empty-body POST outright, and that refusal is shown verbatim too.
   */
  describe("downloading a pack", () => {
    it("offers a Download button that names the size", async () => {
      renderScreen();

      expect(
        await screen.findByRole("button", { name: "Download (631.7 MB)" })
      ).toBeInTheDocument();
    });

    it("shows the job's progress and message while it runs", async () => {
      const user = userEvent.setup();
      server.use(
        http.post(`${API}/api/models/quantem%3Amito/install/`, () =>
          HttpResponse.json({ job_id: "job-dl-1" }, { status: 202 })
        ),
        http.get(`${API}/api/jobs/job-dl-1/`, () =>
          HttpResponse.json(
            makeJob({
              status: "RUNNING",
              progress: 40,
              message: "265.1 MB of 662.3 MB",
            })
          )
        )
      );
      renderScreen();

      await user.click(
        await screen.findByRole("button", { name: "Download (631.7 MB)" })
      );

      expect(await screen.findByText("Downloading…")).toBeInTheDocument();
      expect(screen.getByText("265.1 MB of 662.3 MB")).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Cancel" })
      ).toBeInTheDocument();
      // One download at a time, and the local route waits for it.
      expect(
        screen.getByRole("button", { name: "Install from a local folder" })
      ).toBeDisabled();
    });

    it("refetches the catalogue when the job succeeds", async () => {
      const user = userEvent.setup();
      let catalogueCalls = 0;
      server.use(
        http.get(`${API}/api/models/`, () => {
          catalogueCalls += 1;
          return HttpResponse.json(
            catalogue({
              packs: [pack({ installed: catalogueCalls > 1, runnable: catalogueCalls > 1, reason: null })],
            })
          );
        }),
        http.post(`${API}/api/models/quantem%3Amito/install/`, () =>
          HttpResponse.json({ job_id: "job-dl-1" }, { status: 202 })
        ),
        http.get(`${API}/api/jobs/job-dl-1/`, () =>
          HttpResponse.json(
            makeJob({ status: "SUCCESS", progress: 100, message: "installed" })
          )
        )
      );
      render(
        <MemoryRouter>
          <ModelsScreen />
        </MemoryRouter>
      );

      await user.click(
        await screen.findByRole("button", { name: "Download (631.7 MB)" })
      );

      // The details line reads "Installed · <licence>", one paragraph.
      expect(await screen.findByText(/Installed ·/)).toBeInTheDocument();
      expect(
        screen.queryByRole("button", { name: /Download/ })
      ).not.toBeInTheDocument();
    });

    it("prints a failed job's message verbatim and offers to retry", async () => {
      const user = userEvent.setup();
      const reason =
        "sha256 mismatch for encoder_trunk.safetensors: expected 9f31…, got " +
        "a207…. The downloaded file was discarded; nothing was installed.";
      server.use(
        http.post(`${API}/api/models/quantem%3Amito/install/`, () =>
          HttpResponse.json({ job_id: "job-dl-1" }, { status: 202 })
        ),
        http.get(`${API}/api/jobs/job-dl-1/`, () =>
          HttpResponse.json(
            makeJob({ status: "FAILED", progress: 62, message: reason })
          )
        )
      );
      renderScreen();

      await user.click(
        await screen.findByRole("button", { name: "Download (631.7 MB)" })
      );

      expect(await screen.findByText(new RegExp("sha256 mismatch"))).toBeInTheDocument();
      expect(screen.getByRole("alert").textContent).toContain(reason);
      // The way out is another attempt, so the button has to come back.
      expect(
        screen.getByRole("button", { name: "Download (631.7 MB)" })
      ).toBeInTheDocument();
    });

    it("shows the server's refusal verbatim when the download cannot start", async () => {
      // An older backend: install-with-no-source is a 501 naming the
      // release-bundle route. The one honest thing to render is its words.
      const user = userEvent.setup();
      const refusal =
        "quantem:mito is not installed, and QuantEM cannot download it for " +
        "you: the remote model registry is not implemented.";
      server.use(
        http.post(`${API}/api/models/quantem%3Amito/install/`, () =>
          HttpResponse.json({ error: refusal }, { status: 501 })
        )
      );
      renderScreen();

      await user.click(
        await screen.findByRole("button", { name: "Download (631.7 MB)" })
      );

      expect(await screen.findByRole("alert")).toHaveTextContent(
        "the remote model registry is not implemented"
      );
    });

    it("cancels a running download through POST /cancel/", async () => {
      const user = userEvent.setup();
      let cancelled = false;
      server.use(
        http.post(`${API}/api/models/quantem%3Amito/install/`, () =>
          HttpResponse.json({ job_id: "job-dl-1" }, { status: 202 })
        ),
        http.get(`${API}/api/jobs/job-dl-1/`, () =>
          HttpResponse.json(
            cancelled
              ? makeJob({ status: "CANCELLED", progress: 40 })
              : makeJob({ status: "RUNNING", progress: 40, message: "…" })
          )
        ),
        http.post(`${API}/api/jobs/job-dl-1/cancel/`, () => {
          cancelled = true;
          return HttpResponse.json({ status: "cancel_requested" });
        })
      );
      renderScreen();

      await user.click(
        await screen.findByRole("button", { name: "Download (631.7 MB)" })
      );
      await user.click(await screen.findByRole("button", { name: "Cancel" }));

      await waitFor(
        () => {
          expect(
            screen.getByText(/Download cancelled\. Nothing was installed/)
          ).toBeInTheDocument();
        },
        { timeout: 3000 }
      );
      expect(cancelled).toBe(true);
    });

    it("removes a queued download with DELETE, the only exit a queued job has", async () => {
      const user = userEvent.setup();
      let deleted = false;
      server.use(
        http.post(`${API}/api/models/quantem%3Amito/install/`, () =>
          HttpResponse.json({ job_id: "job-dl-1" }, { status: 202 })
        ),
        http.get(`${API}/api/jobs/job-dl-1/`, () =>
          HttpResponse.json(makeJob({ status: "PENDING", progress: 0 }))
        ),
        http.delete(`${API}/api/jobs/job-dl-1/`, () => {
          deleted = true;
          return new HttpResponse(null, { status: 204 });
        })
      );
      renderScreen();

      await user.click(
        await screen.findByRole("button", { name: "Download (631.7 MB)" })
      );
      expect(await screen.findByText("Waiting in the queue…")).toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "Cancel" }));

      expect(
        await screen.findByText(/removed from the queue before it started/)
      ).toBeInTheDocument();
      expect(deleted).toBe(true);
      // Polling a deleted row would 404 forever; the flow returns to offering
      // the download.
      expect(
        screen.getByRole("button", { name: "Download (631.7 MB)" })
      ).toBeInTheDocument();
    });
  });

  /**
   * An install already underway that this card did not start.
   *
   * uat13 #1: while all four installer-requested downloads were RUNNING,
   * every pack card said "not installed" with a live Download button, and
   * clicking it queued a real duplicate gigabyte download. The catalogue now
   * names the active job per pack (`active_install`), the card adopts it into
   * the same poll the local download uses, and the install POST answers 409
   * while one is active.
   */
  describe("an install the card did not start", () => {
    const activeInstall = {
      job_id: 42,
      status: "RUNNING" as const,
      progress_current_bytes: 214 * 1024 * 1024,
      progress_total_bytes: 1243 * 1024 * 1024,
    };

    it("shows the installing state instead of the Download button, polling the named job", async () => {
      server.use(
        http.get(`${API}/api/jobs/42/`, () =>
          HttpResponse.json(
            makeJob({
              id: "42",
              status: "RUNNING",
              progress: 17,
              message: "downloading quantem:mito: 214 of 1243 MB",
            })
          )
        )
      );
      renderScreen(catalogue({ packs: [pack({ active_install: activeInstall })] }));

      expect(
        await screen.findByTestId("download-progress-quantem:mito")
      ).toBeInTheDocument();
      // The button that queued the duplicate download must be gone.
      expect(
        screen.queryByRole("button", { name: "Download (631.7 MB)" })
      ).not.toBeInTheDocument();
      // Adopted into the same poll: the job's own message shows verbatim,
      // and the job can be cancelled from here.
      expect(
        await screen.findByText("downloading quantem:mito: 214 of 1243 MB")
      ).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Install from a local folder" })
      ).toBeDisabled();
    });

    it("renders the catalogue's byte snapshot until the poll answers", async () => {
      // The job row is unreadable (say, a race with Clear done) — the
      // catalogue snapshot is still enough to say what is happening.
      server.use(
        http.get(`${API}/api/jobs/42/`, () => new HttpResponse(null, { status: 404 }))
      );
      renderScreen(catalogue({ packs: [pack({ active_install: activeInstall })] }));

      expect(
        await screen.findByText("Installing — 214.0 MB of 1.2 GB")
      ).toBeInTheDocument();
      expect(
        screen.queryByRole("button", { name: "Download (631.7 MB)" })
      ).not.toBeInTheDocument();
    });

    it("says Queued for a queued install that has moved no bytes", async () => {
      server.use(
        http.get(`${API}/api/jobs/42/`, () => new HttpResponse(null, { status: 404 }))
      );
      renderScreen(
        catalogue({
          packs: [
            pack({
              active_install: {
                job_id: 42,
                status: "QUEUED",
                progress_current_bytes: null,
                progress_total_bytes: null,
              },
            }),
          ],
        })
      );

      expect(await screen.findByText("Queued")).toBeInTheDocument();
      expect(
        screen.queryByRole("button", { name: "Download (631.7 MB)" })
      ).not.toBeInTheDocument();
    });

    it("surfaces a 409 verbatim and flips to the install it lost the race to", async () => {
      // The race: the click lands before the catalogue could say an install
      // is active. The server refuses; the card shows the refusal and
      // refreshes the list, which brings `active_install` in.
      const user = userEvent.setup();
      const refusal =
        "An install of quantem:mito is already active (job 42); " +
        "watch or cancel it instead of starting another.";
      let catalogueCalls = 0;
      server.use(
        http.get(`${API}/api/models/`, () => {
          catalogueCalls += 1;
          return HttpResponse.json(
            catalogue({
              packs: [
                pack(catalogueCalls > 1 ? { active_install: activeInstall } : {}),
              ],
            })
          );
        }),
        http.post(`${API}/api/models/quantem%3Amito/install/`, () =>
          HttpResponse.json({ error: refusal }, { status: 409 })
        ),
        http.get(`${API}/api/jobs/42/`, () =>
          HttpResponse.json(
            makeJob({ id: "42", status: "RUNNING", progress: 17 })
          )
        )
      );
      render(
        <MemoryRouter>
          <ModelsScreen />
        </MemoryRouter>
      );

      await user.click(
        await screen.findByRole("button", { name: "Download (631.7 MB)" })
      );

      expect(await screen.findByRole("alert")).toHaveTextContent(
        "already active"
      );
      // The refreshed catalogue owns the card now: installing state, no
      // second Download button to click again.
      expect(
        await screen.findByTestId("download-progress-quantem:mito")
      ).toBeInTheDocument();
      expect(
        screen.queryByRole("button", { name: "Download (631.7 MB)" })
      ).not.toBeInTheDocument();
    });
  });

  /**
   * A failure must survive navigation. The card's own download state dies with
   * the unmount, so before this the screen greeted a returning user with "Not
   * installed yet" and a fresh Download button however many times the same
   * install had already failed — the PermissionError that explained it lived
   * in the jobs API alone. `GET /api/models/` carries no last-failure info, so
   * the screen reads the jobs API: queue-status for the FAILED list, then the
   * job row for the pack id and the verbatim message.
   */
  describe("remembering a failed install across visits", () => {
    const reason =
      "promoting the verified download into the model cache failed: " +
      "[WinError 5] Access is denied: 'staging\\quantem__mito' -> " +
      "'packs\\quantem__mito'. Nothing was installed.";

    function serveFailureHistory(job: Job) {
      server.use(
        http.get(`${API}/api/jobs/queue-status/`, () =>
          HttpResponse.json({
            ...EMPTY_JOB_QUEUE_STATUS,
            failed: [makeQueueItem({ id: job.id, status: job.status })],
          })
        ),
        http.get(`${API}/api/jobs/${job.id}/`, () => HttpResponse.json(job))
      );
    }

    it("surfaces the last failed install verbatim, with its timestamp, from a fresh mount", async () => {
      serveFailureHistory(
        makeJob({
          id: "job-old-1",
          status: "FAILED",
          message: reason,
          finished_at: "2026-02-01T00:05:00Z",
        })
      );
      renderScreen();

      const notice = await screen.findByTestId("last-failed-install-quantem:mito");
      expect(notice).toHaveTextContent("The last download of this pack failed");
      // Verbatim: the message is the only text separating a dead network from
      // a digest mismatch from this access-denied promote.
      expect(notice.textContent).toContain(reason);
      expect(notice.textContent).toContain(
        new Date(Date.parse("2026-02-01T00:05:00Z")).toLocaleString()
      );
      // And the action calls itself what it is.
      expect(
        screen.getByRole("button", { name: "Retry download (631.7 MB)" })
      ).toBeInTheDocument();
      expect(
        screen.queryByRole("button", { name: "Download (631.7 MB)" })
      ).not.toBeInTheDocument();
    });

    it("retrying actually re-requests the install and clears the old notice", async () => {
      const user = userEvent.setup();
      serveFailureHistory(
        makeJob({ id: "job-old-1", status: "FAILED", message: reason })
      );
      let installRequests = 0;
      server.use(
        http.post(`${API}/api/models/quantem%3Amito/install/`, () => {
          installRequests += 1;
          return HttpResponse.json({ job_id: "job-dl-1" }, { status: 202 });
        }),
        http.get(`${API}/api/jobs/job-dl-1/`, () =>
          HttpResponse.json(
            makeJob({ id: "job-dl-1", status: "RUNNING", progress: 10 })
          )
        )
      );
      renderScreen();

      await user.click(
        await screen.findByRole("button", { name: "Retry download (631.7 MB)" })
      );

      expect(await screen.findByText("Downloading…")).toBeInTheDocument();
      expect(installRequests).toBe(1);
      // The live attempt owns the card now; printing the stale failure beside
      // a progress bar would read as the retry already having failed.
      expect(
        screen.queryByTestId("last-failed-install-quantem:mito")
      ).not.toBeInTheDocument();
    });

    it("does not pin another pack's failure on this card", async () => {
      serveFailureHistory(
        makeJob({
          id: "job-old-1",
          status: "FAILED",
          message: reason,
          payload_json: { pack_id: "omniem:ld" },
          tags: ["model:omniem:ld"],
        })
      );
      renderScreen();

      expect(
        await screen.findByRole("button", { name: "Download (631.7 MB)" })
      ).toBeInTheDocument();
      expect(
        screen.queryByTestId("last-failed-install-quantem:mito")
      ).not.toBeInTheDocument();
    });

    it("ignores cancelled rows: a cancellation is the user's own act, not a failure", async () => {
      serveFailureHistory(
        makeJob({ id: "job-old-1", status: "CANCELLED", message: "cancelling" })
      );
      renderScreen();

      expect(
        await screen.findByRole("button", { name: "Download (631.7 MB)" })
      ).toBeInTheDocument();
      expect(
        screen.queryByTestId("last-failed-install-quantem:mito")
      ).not.toBeInTheDocument();
    });
  });
});
