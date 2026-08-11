import { createRef } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  ImageUploadPanel,
  type ImageUploadPanelHandle,
} from "@/features/library/components/ImageUploadPanel";
import { uploadAsset } from "@/shared/api/assets";
import { getSystemStatus } from "@/shared/api/jobs";
import { server } from "@/test/msw/server";
import { EMPTY_MODEL_CATALOGUE } from "@/test/msw/handlers";
import type { ModelPack } from "@/shared/types/finetune";
import type { AssetDetail } from "@/shared/types/images";
import type { SystemStatus } from "@/shared/types/jobs";

vi.mock("@/shared/api/assets", async () => {
  const actual =
    await vi.importActual<typeof import("@/shared/api/assets")>(
      "@/shared/api/assets"
    );
  return { ...actual, uploadAsset: vi.fn() };
});

vi.mock("@/shared/api/jobs", async () => {
  const actual =
    await vi.importActual<typeof import("@/shared/api/jobs")>("@/shared/api/jobs");
  return { ...actual, getSystemStatus: vi.fn() };
});

function makeAsset(): AssetDetail {
  return {
    id: "asset-1",
    file_path: "",
    original_filename: "scan.png",
    display_name: "scan",
    is_eval_set: false,
    width: 512,
    height: 512,
    channels: 1,
    bit_depth: 8,
    pixel_size_nm: 4.2,
    preprocess_stage: "ENCODING",
    preprocess_progress: 0,
    tags: [],
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
}

/**
 * A PNG, which is a *definitely* uncalibrated import.
 *
 * `extract_png_metadata` never reads a pixel size -- Pillow's `pHYs` is not
 * consulted -- so choosing one is what lets the form state flatly that the run
 * will be uncalibrated rather than hedge.
 */
function pngFile(name = "scan.png"): File {
  return new File(["fake"], name, { type: "image/png" });
}

/**
 * A minimal classic TIFF that declares `pixelSizeNm`, or nothing at all.
 *
 * Real bytes rather than a mocked probe: the whole point of reading the file is
 * that the form's claim matches what the server will store from the same bytes,
 * and a mock would assert only that the form calls a function.
 */
function tiffFile(
  pixelSizeNm: number | null,
  name = "stack.tif"
): File {
  const tagCount = pixelSizeNm === null ? 1 : 3;
  const ifdSize = 2 + tagCount * 12 + 4;
  const header = new DataView(new ArrayBuffer(8));
  header.setUint16(0, 0x4949, true);
  header.setUint16(2, 42, true);
  header.setUint32(4, 8, true);

  const ifd = new DataView(new ArrayBuffer(ifdSize));
  ifd.setUint16(0, tagCount, true);
  // ImageWidth, so even the uncalibrated file is a structurally real TIFF.
  ifd.setUint16(2, 256, true);
  ifd.setUint16(4, 4, true);
  ifd.setUint32(6, 1, true);
  ifd.setUint32(10, 512, true);

  const parts: ArrayBuffer[] = [header.buffer, ifd.buffer];
  if (pixelSizeNm !== null) {
    // XResolution as pixels per centimetre, the way a real export writes it.
    ifd.setUint16(14, 282, true);
    ifd.setUint16(16, 5, true);
    ifd.setUint32(18, 1, true);
    ifd.setUint32(22, 8 + ifdSize, true);
    ifd.setUint16(26, 296, true);
    ifd.setUint16(28, 3, true);
    ifd.setUint32(30, 1, true);
    ifd.setUint16(34, 3, true); // centimetre

    const rational = new DataView(new ArrayBuffer(8));
    rational.setUint32(0, Math.round(10_000_000 / pixelSizeNm), true);
    rational.setUint32(4, 1, true);
    parts.push(rational.buffer);
  }

  return new File(parts, name, { type: "image/tiff" });
}

/**
 * Submit the form directly.
 *
 * jsdom does not implement `HTMLFormElement.requestSubmit`, so clicking a
 * `type="submit"` button never dispatches `submit` here even though it does in
 * every real browser. Dispatching the event is the closest faithful stand-in.
 *
 * Anchored on the form's test id rather than on the submit button's caption:
 * that caption changes to name the uncalibrated choice, and a helper that has
 * to know which wording is on screen would break every test that is not about
 * it. (The file input is deliberately *outside* the form -- it has to exist
 * before a file is chosen, and the form does not.)
 */
function submitUploadForm() {
  fireEvent.submit(screen.getByTestId("import-form"));
}

/**
 * Open the "Start a segmentation run now" section.
 *
 * Closed on arrival: importing an image imports an image, and four
 * whole-image-run checkboxes are not the "optional add-ons for the resolution
 * and notes" the import surface is supposed to be.
 */
async function openRunOptions(user: ReturnType<typeof userEvent.setup>) {
  const toggle = await screen.findByRole("button", {
    name: /start a segmentation run now/i,
  });
  if (toggle.getAttribute("aria-expanded") !== "true") await user.click(toggle);
}

/**
 * Get to the state the run checkboxes live in: a file chosen, and the
 * segmentation section unfolded.
 */
async function openRunOptionsWithFile(user: ReturnType<typeof userEvent.setup>) {
  const input = await openPanelWithServerFormats();
  await user.upload(input, pngFile());
  await openRunOptions(user);
}

/** A pack that resamples: `canonical_nm` is what makes it warn. */
function pack(overrides: Partial<ModelPack> = {}): ModelPack {
  return {
    id: "quantem:mito",
    family: "quantem",
    organelle: "mito",
    title: "QuantEM — Mitochondria",
    installed: true,
    download_bytes: 662337373,
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
    encoder_tier: null,
    ...overrides,
  };
}

/** The catalogue the import form reads to decide what its checkboxes will run. */
function serveCatalogue(packs: ModelPack[]) {
  server.use(
    http.get("http://127.0.0.1:8000/api/models/", () =>
      HttpResponse.json({ ...EMPTY_MODEL_CATALOGUE, packs })
    )
  );
}

/** Open the panel and wait for the server-driven accept list to land. */
async function openPanelWithServerFormats() {
  const input = await screen.findByLabelText(/image file/i);
  await waitFor(() => expect(input).toHaveAttribute("accept", ".tif,.tiff,.png"));
  return input;
}

/**
 * Drop files on the panel.
 *
 * The zone is the empty state; once a queue exists the panel itself is still
 * the drop target (and so is the whole Library page), which is what makes
 * "drop three more onto the list" work.
 */
function dropFiles(files: File[]) {
  const zone =
    screen.queryByTestId("import-drop-zone") ?? screen.getByTestId("import-panel");
  fireEvent.drop(zone, { dataTransfer: { files, types: ["Files"] } });
}

/**
 * A `/api/system/status/` body.
 *
 * The cast is deliberate and is the one seam this package could not close:
 * `max_upload_bytes` is a real field on the endpoint and is read through
 * `readMaxUploadBytes`, but the `SystemStatus` interface lives in
 * `shared/types/jobs.ts`, which is not this package's file to edit. See the
 * report.
 */
function systemStatus(maxUploadBytes: number | null = 64 * 1024 ** 3): SystemStatus {
  return {
    cuda_available: false,
    supported_upload_formats: [".tif", ".tiff", ".png"],
    ...(maxUploadBytes === null ? {} : { max_upload_bytes: maxUploadBytes }),
  } as SystemStatus;
}

/** A file of a stated size, for the size-limit checks. */
function fileOfSize(bytes: number, name = "huge.tif"): File {
  return new File([new Uint8Array(bytes)], name, { type: "image/tiff" });
}

/**
 * Ask for a run, explicitly.
 *
 * Nothing is ticked on arrival any more, so every test about what a *run* does
 * has to say which run it means. That is the point of the change: the form no
 * longer decides on the user's behalf that four whole-image inference passes
 * should start.
 */
async function tickOrganelle(
  user: ReturnType<typeof userEvent.setup>,
  label: RegExp
) {
  const box = await screen.findByLabelText(label);
  await user.click(box);
  expect(box).toBeChecked();
  return box;
}

describe("ImageUploadPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getSystemStatus).mockResolvedValue(systemStatus());
    vi.mocked(uploadAsset).mockResolvedValue(makeAsset());
  });

  it("drives the accepted formats from supported_upload_formats", async () => {
    render(<ImageUploadPanel />);

    const input = await screen.findByLabelText(/image file/i);
    await waitFor(() => {
      expect(input).toHaveAttribute("accept", ".tif,.tiff,.png");
    });
    expect(screen.getByText(/\.tif, \.tiff, \.png/)).toBeInTheDocument();
  });

  it("accepts a PNG, which the server accepts too", async () => {
    const user = userEvent.setup();
    render(<ImageUploadPanel onUploaded={vi.fn()} />);

    const input = await openPanelWithServerFormats();
    await user.upload(input, pngFile());

    expect(screen.getByText("scan.png")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
    // The display name is derived by stripping a *known* extension.
    expect(screen.getByLabelText(/display name/i)).toHaveValue("scan");
  });

  it("rejects an unsupported extension inline rather than with alert()", async () => {
    const alertSpy = vi.spyOn(window, "alert").mockImplementation(() => {});
    // applyAccept: false so the component's own guard runs -- a drag-drop or a
    // browser that ignores `accept` reaches exactly this path.
    const user = userEvent.setup({ applyAccept: false });
    render(<ImageUploadPanel />);

    const input = await openPanelWithServerFormats();
    await user.upload(input, new File(["x"], "notes.txt", { type: "text/plain" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /not a supported format/i
    );
    expect(alertSpy).not.toHaveBeenCalled();
  });

  it("sends a typed pixel size with the upload", async () => {
    const user = userEvent.setup();
    render(<ImageUploadPanel onUploaded={vi.fn()} />);

    const input = await openPanelWithServerFormats();
    await user.upload(input, pngFile());
    await user.type(screen.getByLabelText(/pixel size/i), "4.2");
    submitUploadForm();

    await waitFor(() => {
      expect(uploadAsset).toHaveBeenCalledWith(
        expect.any(File),
        expect.objectContaining({ pixelSizeNm: 4.2 })
      );
    });
  });

  it("omits the pixel size when left blank so the file's own value survives", async () => {
    const user = userEvent.setup();
    render(<ImageUploadPanel onUploaded={vi.fn()} />);

    const input = await openPanelWithServerFormats();
    await user.upload(input, pngFile());
    submitUploadForm();

    await waitFor(() => {
      expect(uploadAsset).toHaveBeenCalledWith(
        expect.any(File),
        expect.objectContaining({ pixelSizeNm: null })
      );
    });
  });

  /**
   * The grouping fields, and the property that matters most about them.
   *
   * An import that ignores them must produce exactly the request this form
   * produced before they existed: no nag, no blocking validation, no field
   * silently defaulted to something. An unorganised library is where every user
   * starts and where many will stay.
   */
  describe("experiment and dataset at import", () => {
    it("sends nothing at all when neither is touched", async () => {
      const user = userEvent.setup();
      render(<ImageUploadPanel onUploaded={vi.fn()} />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      submitUploadForm();

      await waitFor(() => expect(uploadAsset).toHaveBeenCalled());
      const options = vi.mocked(uploadAsset).mock.calls[0][1] ?? {};
      expect(options.experimentId).toBeUndefined();
      expect(options.experimentName).toBeUndefined();
      expect(options.datasetId).toBeUndefined();
      expect(options.datasetName).toBeUndefined();
    });

    it("does not block the import when nothing is chosen", async () => {
      const user = userEvent.setup();
      render(<ImageUploadPanel onUploaded={vi.fn()} />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      submitUploadForm();

      await waitFor(() => expect(uploadAsset).toHaveBeenCalledTimes(1));
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });

    it("sends a typed experiment name with the import", async () => {
      const user = userEvent.setup();
      render(<ImageUploadPanel onUploaded={vi.fn()} />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      await user.selectOptions(screen.getByLabelText("Experiment"), "__new__");
      await user.type(
        screen.getByLabelText("New experiment…"),
        "Fasted cohort"
      );
      submitUploadForm();

      await waitFor(() =>
        expect(uploadAsset).toHaveBeenCalledWith(
          expect.any(File),
          expect.objectContaining({ experimentName: "Fasted cohort" })
        )
      );
    });

    /**
     * A dataset lives inside exactly one experiment, so there is nothing to
     * choose from until one is named. Disabled with a reason rather than
     * hidden: a control that vanishes teaches nothing about why.
     */
    it("cannot name a dataset before an experiment", async () => {
      const user = userEvent.setup();
      render(<ImageUploadPanel onUploaded={vi.fn()} />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());

      expect(screen.getByLabelText("Dataset")).toBeDisabled();
      expect(
        screen.getByText("A dataset sits inside an experiment. Choose one first.")
      ).toBeInTheDocument();
    });

    it("applies the same experiment to every file in a batch", async () => {
      const user = userEvent.setup();
      render(<ImageUploadPanel onUploaded={vi.fn()} />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, [pngFile("a.png"), pngFile("b.png")]);
      await user.selectOptions(screen.getByLabelText("Experiment"), "__new__");
      await user.type(screen.getByLabelText("New experiment…"), "Fasted cohort");
      submitUploadForm();

      await waitFor(() => expect(uploadAsset).toHaveBeenCalledTimes(2));
      // The same typed name on both: the server resolves it to one row, so two
      // files land in one experiment rather than in two identical ones.
      for (const call of vi.mocked(uploadAsset).mock.calls) {
        expect(call[1]?.experimentName).toBe("Fasted cohort");
      }
    });
  });

  it("blocks a non-positive pixel size before it reaches the server", async () => {
    const user = userEvent.setup();
    render(<ImageUploadPanel onUploaded={vi.fn()} />);

    const input = await openPanelWithServerFormats();
    await user.upload(input, pngFile());
    await user.type(screen.getByLabelText(/pixel size/i), "0");
    submitUploadForm();

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Pixel size must be greater than zero."
    );
    expect(uploadAsset).not.toHaveBeenCalled();
  });

  /**
   * The third door into an uncalibrated run, and the one everybody starts on.
   *
   * Each ticked "Segment ..." box queues the same whole-image inference pass
   * the create dialog and "Run Full Segmentation" both stop to warn about. This
   * form's only text was the units framing -- "areas and distances stay in
   * pixels" -- which reads as a reporting detail that can be fixed later. It
   * cannot: the pack resamples before it looks for anything, so the pixel size
   * decides which objects exist.
   */
  describe("uncalibrated segmentation on import", () => {
    it("says the object set changes when an organelle is ticked with no pixel size", async () => {
      serveCatalogue([pack()]);
      const user = userEvent.setup();
      render(<ImageUploadPanel />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      await openRunOptions(user);
      await tickOrganelle(user, /segment mitochondria/i);

      expect(
        await screen.findByText(/quantem:mito \(8 nm\/px\)/)
      ).toBeInTheDocument();
      // The claim that was missing. Not "the units change".
      expect(
        screen.getByText(
          /changes which objects exist, not just the units they are reported in/i
        )
      ).toBeInTheDocument();
      expect(
        screen.getByText(/completely different number of objects/i)
      ).toBeInTheDocument();
      // And the choice is named on the control that makes it, with the number
      // of runs it will start.
      expect(
        await screen.findByRole("button", {
          name: "Import and start 1 uncalibrated run",
        })
      ).toBeInTheDocument();
    });

    it("clears the warning as soon as a pixel size is typed", async () => {
      serveCatalogue([pack()]);
      const user = userEvent.setup();
      render(<ImageUploadPanel />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      await openRunOptions(user);
      await tickOrganelle(user, /segment mitochondria/i);
      await screen.findByText(/quantem:mito \(8 nm\/px\)/);
      await user.type(screen.getByLabelText(/pixel size/i), "5");

      await waitFor(() => {
        expect(
          screen.queryByText(/changes which objects exist/i)
        ).not.toBeInTheDocument();
      });
      // The run is still going to happen, so the button still says so. It used
      // to fall back to a plain "Upload" here -- the one case where the run was
      // certain got the quietest caption on the form.
      expect(
        screen.getByRole("button", { name: "Import and start 1 segmentation run" })
      ).toBeInTheDocument();
    });

    it("says nothing when no organelle is ticked", async () => {
      serveCatalogue([pack()]);
      const user = userEvent.setup();
      render(<ImageUploadPanel />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      // Tick and untick, so this is a form that has been looked at rather than
      // one that simply started empty.
      await openRunOptions(user);
      const mito = await tickOrganelle(user, /segment mitochondria/i);
      await screen.findByText(/quantem:mito \(8 nm\/px\)/);
      await user.click(mito);

      await waitFor(() => {
        expect(
          screen.queryByText(/changes which objects exist/i)
        ).not.toBeInTheDocument();
      });
      expect(screen.getByRole("button", { name: "Import image" })).toBeInTheDocument();
    });

    /**
     * Both ER packs declare `canonical_nm: null` and genuinely run at native
     * scale, so there is nothing to warn about. Warning anyway would train
     * people to click through the warning that matters.
     */
    it("says nothing for a pack that runs at native scale by design", async () => {
      serveCatalogue([
        pack({ id: "quantem:er", organelle: "er", canonical_nm: null }),
      ]);
      const user = userEvent.setup();
      render(<ImageUploadPanel />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      await openRunOptions(user);
      await tickOrganelle(user, /segment er/i);

      await waitFor(() => {
        expect(
          screen.queryByText(/changes which objects exist/i)
        ).not.toBeInTheDocument();
      });
      // The run is still named and counted; only the scale warning is absent.
      expect(
        screen.getByRole("button", { name: "Import and start 1 segmentation run" })
      ).toBeInTheDocument();
    });

    it("names every affected pack, not just the first", async () => {
      serveCatalogue([
        pack(),
        pack({
          id: "quantem:nucleus",
          organelle: "nucleus",
          canonical_nm: 25,
        }),
      ]);
      const user = userEvent.setup();
      render(<ImageUploadPanel />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      await openRunOptions(user);
      await tickOrganelle(user, /segment mitochondria/i);
      await openRunOptions(user);
      await tickOrganelle(user, /segment nucleus/i);

      expect(
        await screen.findByText(
          /quantem:mito \(8 nm\/px\), quantem:nucleus \(25 nm\/px\)/
        )
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Import and start 2 uncalibrated runs" })
      ).toBeInTheDocument();
    });

    /**
     * "You can set or change it later" is true of the number and false of
     * everything downstream once a run has been queued. Read on its own it is
     * what made an uncalibrated import look reversible.
     */
    it("stops promising that setting the pixel size later fixes the run", async () => {
      serveCatalogue([pack()]);
      const user = userEvent.setup();
      render(<ImageUploadPanel />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      await openRunOptions(user);
      await tickOrganelle(user, /segment mitochondria/i);

      expect(
        await screen.findByText(/does not re-run anything segmented now/i)
      ).toBeInTheDocument();
    });
  });

  /**
   * The reported complaint, and the reason it matters more than a wording nit.
   *
   * The helper text under the pixel-size box says "Leave blank to use the value
   * in the file". Doing exactly that with a TIFF that declares 5 nm/px produced
   * the full uncalibrated warning and a button reading "Import and segment
   * uncalibrated", over an import that came back `pixel_size_nm: 5.0,
   * file_declared_pixel_size_nm: 5.0`. Firing on the commonest correct workflow
   * is how a warning stops being read -- and this is the one warning in the
   * application that must still be believed when it is right.
   */
  describe("reading the file's own pixel size", () => {
    it("does not warn about a TIFF that declares one", async () => {
      serveCatalogue([pack()]);
      const user = userEvent.setup();
      render(<ImageUploadPanel />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, tiffFile(5));
      // Ticked deliberately: with nothing ticked there is no run to warn about
      // and this test would pass without reading the file at all.
      await openRunOptions(user);
      await tickOrganelle(user, /segment mitochondria/i);

      // The blank box is the documented workflow, not a mistake.
      expect(
        await screen.findByText(/stack\.tif declares 5 nm\/px/)
      ).toBeInTheDocument();
      await waitFor(() => {
        expect(
          screen.queryByText(/changes which objects exist/i)
        ).not.toBeInTheDocument();
      });
      const button = screen.getByRole("button", {
        name: "Import and start 1 segmentation run",
      });
      expect(button).toBeInTheDocument();
      expect(button).not.toHaveTextContent("uncalibrated");
    });

    it("says the typed value wins when both exist", async () => {
      // The server does the same: a posted pixel size is never overwritten by
      // the file's own metadata.
      serveCatalogue([pack()]);
      const user = userEvent.setup();
      render(<ImageUploadPanel />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, tiffFile(5));
      await screen.findByText(/stack\.tif declares 5 nm\/px/);
      await user.type(screen.getByLabelText(/pixel size/i), "8");

      expect(
        await screen.findByText(/The value typed below is used instead/)
      ).toBeInTheDocument();
    });

    it("still states it flatly for a TIFF that declares nothing", async () => {
      // The case the warning exists for. Read, understood, and empty -- so it
      // keeps the unhedged wording and the unhedged button.
      serveCatalogue([pack()]);
      const user = userEvent.setup();
      render(<ImageUploadPanel />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, tiffFile(null, "bare.tif"));
      await openRunOptions(user);
      await tickOrganelle(user, /segment mitochondria/i);

      expect(
        await screen.findByText(/No pixel size, and a model that needs one/)
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Import and start 1 uncalibrated run" })
      ).toBeInTheDocument();
      expect(screen.queryByText(/declares/)).not.toBeInTheDocument();
    });

    it("hedges rather than asserting when the file cannot be read", async () => {
      serveCatalogue([pack()]);
      const user = userEvent.setup();
      render(<ImageUploadPanel />);

      const input = await openPanelWithServerFormats();
      // BigTIFF: a real format this build does not parse here.
      const header = new DataView(new ArrayBuffer(16));
      header.setUint16(0, 0x4949, true);
      header.setUint16(2, 43, true);
      await user.upload(input, new File([header.buffer], "huge.tif"));
      await openRunOptions(user);
      await tickOrganelle(user, /segment mitochondria/i);

      expect(
        await screen.findByText(/If no pixel size arrives with this image/)
      ).toBeInTheDocument();
      expect(
        screen.getByText(/could not read one out of huge\.tif/)
      ).toBeInTheDocument();
      // The button counts the run, but must not assert what the paragraph just
      // declined to.
      const button = screen.getByRole("button", {
        name: "Import and start 1 segmentation run",
      });
      expect(button).toBeInTheDocument();
      expect(button).not.toHaveTextContent("uncalibrated");
    });
  });

  /**
   * All four boxes used to be ticked, so importing one image queued four
   * whole-image CPU passes.
   *
   * The first fix restricted that to *installed* packs, which addressed a
   * different complaint -- on a two-pack machine, two of the four runs FAILED
   * -- and left the four passes wherever the install was complete. The runs
   * that succeeded were never the smaller problem: they are minutes to tens of
   * minutes each, on organelles nobody had asked about, and the only way to
   * stop one is the Library's job sidebar.
   */
  describe("what the organelle boxes start as", () => {
    it("ticks nothing, even when every pack can run here", async () => {
      serveCatalogue([
        pack(),
        pack({ id: "quantem:er", organelle: "er", canonical_nm: null }),
        pack({ id: "quantem:nucleus", organelle: "nucleus", canonical_nm: 25 }),
        pack({ id: "quantem:ld", organelle: "ld" }),
      ]);
      const user = userEvent.setup();
      render(<ImageUploadPanel />);
      await openRunOptionsWithFile(user);

      // The full install: the case the old default turned into four runs.
      const mito = await screen.findByLabelText(/segment mitochondria/i);
      await waitFor(() => expect(mito).toBeEnabled());
      for (const label of [
        /segment mitochondria/i,
        /segment er/i,
        /segment nucleus/i,
        /segment lipid droplets/i,
      ]) {
        expect(screen.getByLabelText(label)).not.toBeChecked();
      }
    });

    it("says that nothing runs unless a box is ticked, and where else to run it", async () => {
      serveCatalogue([pack()]);
      const user = userEvent.setup();
      render(<ImageUploadPanel />);
      await openRunOptionsWithFile(user);

      expect(
        await screen.findByText(/Nothing is segmented unless you tick a box/)
      ).toBeInTheDocument();
      expect(
        screen.getByText(/start any of them later from the labeling screen/)
      ).toBeInTheDocument();
    });

    it("imports without queueing anything when no box is touched", async () => {
      serveCatalogue([pack()]);
      const user = userEvent.setup();
      render(<ImageUploadPanel onUploaded={vi.fn()} />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      expect(screen.getByRole("button", { name: "Import image" })).toBeInTheDocument();
      submitUploadForm();

      await waitFor(() => {
        expect(uploadAsset).toHaveBeenCalledWith(
          expect.any(File),
          expect.objectContaining({
            segmentMito: false,
            segmentEr: false,
            segmentNucleus: false,
            segmentLd: false,
          })
        );
      });
    });

    it("does not offer a run that is known to fail, and says why", async () => {
      serveCatalogue([
        pack({
          id: "quantem:nucleus",
          organelle: "nucleus",
          installed: false,
          runnable: false,
          reason: "Not installed yet.",
        }),
      ]);
      const user = userEvent.setup();
      render(<ImageUploadPanel />);
      await openRunOptionsWithFile(user);

      const nucleus = await screen.findByLabelText(/segment nucleus/i);
      await waitFor(() => expect(nucleus).toBeDisabled());
      expect(screen.getByText(/Not installed yet\./)).toBeInTheDocument();
    });

    it("leaves a pack the catalogue says nothing about enabled", async () => {
      // "Unknown" is not "blocked": offer it, and do not claim it works.
      serveCatalogue([pack()]);
      const user = userEvent.setup();
      render(<ImageUploadPanel />);
      await openRunOptionsWithFile(user);

      const ld = await screen.findByLabelText(/segment lipid droplets/i);
      await waitFor(() => expect(ld).toBeEnabled());
      expect(ld).not.toBeChecked();
    });

    it("does not tick a box when the catalogue lands", async () => {
      // The catalogue arrives after the first render. It used to bring the
      // default with it, so a box could tick itself under the user's cursor.
      serveCatalogue([pack()]);
      const user = userEvent.setup();
      render(<ImageUploadPanel />);
      await openRunOptionsWithFile(user);

      const mito = await screen.findByLabelText(/segment mitochondria/i);
      await user.click(mito);
      expect(mito).toBeChecked();
      await user.click(mito);

      expect(mito).not.toBeChecked();
      await waitFor(() => expect(mito).not.toBeChecked());
    });

    it("sends exactly the ticked set with the upload", async () => {
      serveCatalogue([
        pack(),
        pack({ id: "quantem:er", organelle: "er", canonical_nm: null }),
      ]);
      const user = userEvent.setup();
      render(<ImageUploadPanel onUploaded={vi.fn()} />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      await openRunOptions(user);
      await tickOrganelle(user, /segment mitochondria/i);
      await openRunOptions(user);
      await tickOrganelle(user, /segment er/i);
      submitUploadForm();

      await waitFor(() => {
        expect(uploadAsset).toHaveBeenCalledWith(
          expect.any(File),
          expect.objectContaining({
            segmentMito: true,
            segmentEr: true,
            segmentNucleus: false,
            segmentLd: false,
          })
        );
      });
    });

    it("counts the runs on the button", async () => {
      serveCatalogue([
        pack(),
        pack({ id: "quantem:er", organelle: "er", canonical_nm: null }),
      ]);
      const user = userEvent.setup();
      render(<ImageUploadPanel />);

      const input = await openPanelWithServerFormats();
      // Calibrated, so the uncalibrated wording is out of the way and this is
      // purely about the count. This is exactly the case that read "Upload".
      await user.upload(input, pngFile());
      await user.type(screen.getByLabelText(/pixel size/i), "5");
      await openRunOptions(user);
      await tickOrganelle(user, /segment mitochondria/i);

      expect(
        screen.getByRole("button", { name: "Import and start 1 segmentation run" })
      ).toBeInTheDocument();

      await openRunOptions(user);
      await tickOrganelle(user, /segment er/i);
      expect(
        screen.getByRole("button", { name: "Import and start 2 segmentation runs" })
      ).toBeInTheDocument();
    });
  });

  /**
   * The Tags box collected text, posted it as `tag_names`, and the server threw
   * it away: there is no tag field on `Asset` and no tag anywhere in the Python
   * tree. Typing "PV" did nothing, the library still showed no tags, and search
   * still matched only names and filenames. Notes is the field that exists, is
   * stored, and *is* searched.
   */
  describe("the free-text field", () => {
    it("offers Notes, not a Tags box nothing reads", async () => {
      const user = userEvent.setup();
      render(<ImageUploadPanel />);
      await user.upload(await openPanelWithServerFormats(), pngFile());

      expect(await screen.findByLabelText(/^notes/i)).toBeInTheDocument();
      expect(screen.queryByLabelText(/tags/i)).not.toBeInTheDocument();
    });

    it("says what typing there achieves", async () => {
      const user = userEvent.setup();
      render(<ImageUploadPanel />);
      await user.upload(await openPanelWithServerFormats(), pngFile());

      expect(
        await screen.findByText(/search box matches notes as well as names/i)
      ).toBeInTheDocument();
    });

    it("sends what was typed, under the name the server reads", async () => {
      const user = userEvent.setup();
      render(<ImageUploadPanel onUploaded={vi.fn()} />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      await user.type(screen.getByLabelText(/^notes/i), "PV, day 14");
      submitUploadForm();

      await waitFor(() => {
        expect(uploadAsset).toHaveBeenCalledWith(
          expect.any(File),
          expect.objectContaining({ notes: "PV, day 14" })
        );
      });
      const [, options] = vi.mocked(uploadAsset).mock.calls[0];
      expect(options).not.toHaveProperty("tagNames");
    });

    it("sends nothing at all when the box is left blank", async () => {
      const user = userEvent.setup();
      render(<ImageUploadPanel onUploaded={vi.fn()} />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      submitUploadForm();

      await waitFor(() => {
        expect(uploadAsset).toHaveBeenCalledWith(
          expect.any(File),
          expect.objectContaining({ notes: undefined })
        );
      });
    });
  });

  /**
   * The owner's first complaint, verbatim: "uploading is not intuitive.
   * Clicking the 'Import an Image' button should open the file selector
   * automatically, but also that area should allow for dropping a file directly
   * into it, either of which should prepopulate the file into the accordion
   * area, and at that point just having optional add-ons for the resolution and
   * notes."
   */
  describe("getting a file into the panel", () => {
    it("has no accordion to expand: the panel is the drop zone", async () => {
      render(<ImageUploadPanel />);

      await screen.findByTestId("import-drop-zone");
      // The old header button. Expanding a section was the whole first step.
      expect(
        screen.queryByRole("button", { name: /import image$/i })
      ).not.toBeInTheDocument();
      expect(screen.queryByText("▶ Import image")).not.toBeInTheDocument();
    });

    /**
     * The drop zone is the file input's `<label>`, which is what makes one
     * click on it open the OS dialog through the browser's own handling. This
     * asserts the wiring rather than the dialog: jsdom has no file chooser, and
     * a test that spied on `HTMLInputElement.click` would pass just as happily
     * over a zone that opened nothing when clicked for real.
     */
    it("wires the whole zone to the file input, so one click opens the picker", async () => {
      render(<ImageUploadPanel />);

      const zone = await screen.findByTestId("import-drop-zone");
      const input = screen.getByLabelText(/image file/i);
      expect(zone.tagName).toBe("LABEL");
      expect(zone.getAttribute("for")).toBe(input.id);
      expect(input).toHaveAttribute("type", "file");
    });

    it("opens the picker for the Library's own Import button", async () => {
      const ref = createRef<ImageUploadPanelHandle>();
      render(<ImageUploadPanel ref={ref} />);

      const input = await screen.findByLabelText(/image file/i);
      const clicked = vi.spyOn(input, "click");
      ref.current?.openFilePicker();

      expect(clicked).toHaveBeenCalledTimes(1);
    });

    it("says the planned sentence, driven by the server's format list", async () => {
      render(<ImageUploadPanel />);

      expect(await screen.findByText("Drop your images here")).toBeInTheDocument();
      expect(screen.getByText("Choose files…")).toBeInTheDocument();
      expect(
        await screen.findByText(
          "TIFF or PNG from this computer. Nothing leaves this machine."
        )
      ).toBeInTheDocument();
    });

    it("names the formats it really accepts when the server accepts something else", async () => {
      vi.mocked(getSystemStatus).mockResolvedValue({
        cuda_available: false,
        supported_upload_formats: [".png"],
      });
      render(<ImageUploadPanel />);

      expect(
        await screen.findByText(
          "PNG from this computer. Nothing leaves this machine."
        )
      ).toBeInTheDocument();
    });

    /** There was no drop handler anywhere in the application. */
    it("takes a dropped file and prepopulates it", async () => {
      render(<ImageUploadPanel />);

      const zone = await screen.findByTestId("import-drop-zone");
      fireEvent.drop(zone, {
        dataTransfer: { files: [pngFile("dropped.png")], types: ["Files"] },
      });

      expect(await screen.findByText("dropped.png")).toBeInTheDocument();
      expect(screen.getByLabelText(/display name/i)).toHaveValue("dropped");
    });

    it("lights the zone up while a file is over it", async () => {
      render(<ImageUploadPanel />);

      const zone = await screen.findByTestId("import-drop-zone");
      expect(zone).toHaveAttribute("data-drag-active", "false");

      fireEvent.dragEnter(zone, { dataTransfer: { types: ["Files"] } });
      expect(zone).toHaveAttribute("data-drag-active", "true");

      fireEvent.dragLeave(zone, { dataTransfer: { types: ["Files"] } });
      expect(zone).toHaveAttribute("data-drag-active", "false");
    });

    it("lights up for a drag anywhere on the page, before the pointer arrives", async () => {
      render(<ImageUploadPanel pageDragActive />);

      expect(await screen.findByTestId("import-drop-zone")).toHaveAttribute(
        "data-drag-active",
        "true"
      );
    });

    it("puts the same validation on the drop path as on the picker", async () => {
      render(<ImageUploadPanel />);

      const zone = await screen.findByTestId("import-drop-zone");
      fireEvent.drop(zone, {
        dataTransfer: {
          files: [new File(["x"], "notes.txt", { type: "text/plain" })],
          types: ["Files"],
        },
      });

      expect(await screen.findByRole("alert")).toHaveTextContent(
        /not a supported format/i
      );
      expect(screen.getByTestId("import-drop-zone")).toBeInTheDocument();
    });

    it("takes every dropped file, not just the first", async () => {
      render(<ImageUploadPanel />);

      dropFiles([pngFile("first.png"), pngFile("second.png"), pngFile("third.png")]);

      expect(await screen.findByText("first.png")).toBeInTheDocument();
      expect(screen.getByText("second.png")).toBeInTheDocument();
      expect(screen.getByText("third.png")).toBeInTheDocument();
      expect(screen.getAllByTestId("import-file-row")).toHaveLength(3);
      expect(screen.queryByRole("alert")).toBeNull();
    });

    it("shows the chosen file's name and size", async () => {
      const user = userEvent.setup();
      render(<ImageUploadPanel />);

      const input = await openPanelWithServerFormats();
      // 2048 bytes, so the size is a number the assertion can name exactly.
      await user.upload(
        input,
        new File([new Uint8Array(2048)], "grid2.tif", { type: "image/tiff" })
      );

      const chosen = await screen.findByTestId("import-chosen-file");
      expect(chosen).toHaveTextContent("grid2.tif");
      expect(chosen).toHaveTextContent("2.0 KB");
    });

    it("keeps the resolution and notes as optional extras, not a form to fill in", async () => {
      const user = userEvent.setup();
      render(<ImageUploadPanel />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());

      expect(
        await screen.findByText("Optional — you can do this later")
      ).toBeInTheDocument();
      // Four whole-image inference runs are not an "optional add-on for the
      // resolution and notes", so they start folded away.
      const toggle = screen.getByRole("button", {
        name: /start a segmentation run now/i,
      });
      expect(toggle).toHaveAttribute("aria-expanded", "false");
      expect(
        screen.queryByLabelText(/segment mitochondria/i)
      ).not.toBeInTheDocument();
      // ...and the fold can never hide a queued run.
      expect(toggle).toHaveTextContent("Nothing selected");
    });

    it("counts the ticked runs on the folded-up toggle", async () => {
      serveCatalogue([pack()]);
      const user = userEvent.setup();
      render(<ImageUploadPanel />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      await openRunOptions(user);
      await tickOrganelle(user, /segment mitochondria/i);
      await user.click(
        screen.getByRole("button", { name: /start a segmentation run now/i })
      );

      expect(
        screen.getByRole("button", { name: /start a segmentation run now/i })
      ).toHaveTextContent("1 run selected");
    });

    it("can put the file back and start again", async () => {
      const user = userEvent.setup();
      render(<ImageUploadPanel />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      await screen.findByTestId("import-chosen-file");

      await user.click(screen.getByRole("button", { name: "Remove" }));

      expect(await screen.findByTestId("import-drop-zone")).toBeInTheDocument();
      expect(screen.queryByTestId("import-chosen-file")).not.toBeInTheDocument();
    });
  });

  /**
   * "It showed Uploading... for a long time, and then seemed to have failed
   * (reverted back to upload button)."
   *
   * It had not failed. `handleSubmit` cleared every field and called
   * `setExpanded(false)` on success, so the accordion header -- a button
   * reading "▶ Import image" -- was all that was left, and a successful import
   * and a reset form were pixel-for-pixel identical. There was no toast
   * anywhere in the application, no inline confirmation, and no highlight on
   * the new card.
   */
  describe("what success looks like", () => {
    it("hands the created asset up instead of quietly collapsing", async () => {
      const onUploaded = vi.fn();
      const user = userEvent.setup();
      render(<ImageUploadPanel onUploaded={onUploaded} />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      submitUploadForm();

      await waitFor(() => expect(onUploaded).toHaveBeenCalledTimes(1));
      // The batch position travels with it: the Library opens a lone import by
      // itself and must not do that in the middle of a plate.
      expect(onUploaded).toHaveBeenCalledWith(
        expect.objectContaining({ id: "asset-1" }),
        { index: 1, total: 1 }
      );
      // The panel is still here and still usable -- it is ready for the next
      // import, not hidden behind a closed accordion.
      expect(await screen.findByTestId("import-drop-zone")).toBeInTheDocument();
      expect(screen.getByText("Drop your images here")).toBeInTheDocument();
    });

    it("keeps the file and the error on screen when the import really fails", async () => {
      vi.mocked(uploadAsset).mockRejectedValue(new Error("Request Entity Too Large"));
      const onUploaded = vi.fn();
      const user = userEvent.setup();
      render(<ImageUploadPanel onUploaded={onUploaded} />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      submitUploadForm();

      expect(await screen.findByRole("alert")).toHaveTextContent(
        "Request Entity Too Large"
      );
      expect(onUploaded).not.toHaveBeenCalled();
      // A failure and a success must not look the same. This one keeps the
      // file, so pressing the button again retries it.
      expect(screen.getByTestId("import-chosen-file")).toHaveTextContent(
        "scan.png"
      );
    });
  });

  /**
   * "Choose files…", plural, over an input that took exactly one.
   *
   * `#file-input` had no `multiple`, so the OS picker refused the second file
   * and a drop of forty imported one and printed a sentence about it. A
   * postdoc coming off a session has a plate. Everything below is the queue
   * that makes forty images one trip through this panel instead of forty.
   */
  describe("a plate, not a picture", () => {
    it("lets the picker take more than one file", async () => {
      render(<ImageUploadPanel />);

      expect(await screen.findByLabelText(/image file/i)).toHaveAttribute(
        "multiple"
      );
    });

    it("imports each file as its own request, in the order they were queued", async () => {
      const onUploaded = vi.fn();
      render(<ImageUploadPanel onUploaded={onUploaded} />);
      await openPanelWithServerFormats();

      dropFiles([pngFile("a.png"), pngFile("b.png"), pngFile("c.png")]);
      await screen.findByText("c.png");
      submitUploadForm();

      await waitFor(() => expect(uploadAsset).toHaveBeenCalledTimes(3));
      expect(
        vi.mocked(uploadAsset).mock.calls.map(([file]) => file.name)
      ).toEqual(["a.png", "b.png", "c.png"]);
      // Each one is handed up as it lands, with its place in the batch, so the
      // Library can pin it immediately and can tell a plate from a picture.
      expect(onUploaded).toHaveBeenCalledTimes(3);
      expect(onUploaded.mock.calls.map(([, batch]) => batch)).toEqual([
        { index: 1, total: 3 },
        { index: 2, total: 3 },
        { index: 3, total: 3 },
      ]);
    });

    it("puts the optional pixel size and notes on every image", async () => {
      const user = userEvent.setup();
      render(<ImageUploadPanel />);
      await openPanelWithServerFormats();

      dropFiles([pngFile("a.png"), pngFile("b.png")]);
      await screen.findByText("b.png");
      await user.type(screen.getByLabelText("Pixel size, nm per pixel"), "4.2");
      await user.type(screen.getByLabelText(/^notes$/i), "day 14");
      submitUploadForm();

      await waitFor(() => expect(uploadAsset).toHaveBeenCalledTimes(2));
      for (const [, options] of vi.mocked(uploadAsset).mock.calls) {
        expect(options).toMatchObject({ pixelSizeNm: 4.2, notes: "day 14" });
      }
      // And each keeps its own name rather than inheriting one from the batch.
      expect(
        vi.mocked(uploadAsset).mock.calls.map(([, options]) => options?.displayName)
      ).toEqual(["a", "b"]);
    });

    it("states what each image will be imported at, before anything is sent", async () => {
      const user = userEvent.setup();
      render(<ImageUploadPanel />);
      await openPanelWithServerFormats();

      dropFiles([tiffFile(5, "declares.tif"), pngFile("silent.png")]);
      await screen.findByText("silent.png");
      await waitFor(() =>
        expect(
          screen.getByText(/1 image says 5 nm\/px in the file/)
        ).toBeInTheDocument()
      );
      expect(
        screen.getByText(/1 image does not say\. Until you set one/)
      ).toBeInTheDocument();

      const rows = screen.getAllByTestId("import-file-row");
      expect(rows[0]).toHaveTextContent("importing at 5 nm/px (from the file)");
      expect(rows[1]).toHaveTextContent("no pixel size: measured in pixels");

      await user.type(screen.getByLabelText("Pixel size, nm per pixel"), "4.2");
      expect(screen.getAllByTestId("import-file-row")[1]).toHaveTextContent(
        "importing at 4.2 nm/px (you typed it)"
      );
    });

    /**
     * The critic's "fabricates 43 calibrations", in the one place a batch can
     * commit it. A typed value reaches the images that declare nothing; the
     * twelve that carry a real tag keep it until the user says otherwise, in
     * as many words.
     */
    it("does not overwrite a declared pixel size unless asked to", async () => {
      const user = userEvent.setup();
      render(<ImageUploadPanel />);
      await openPanelWithServerFormats();

      dropFiles([tiffFile(5, "declares.tif"), pngFile("silent.png")]);
      await screen.findByText("silent.png");
      await waitFor(() =>
        expect(
          screen.getByText(/1 image says 5 nm\/px in the file/)
        ).toBeInTheDocument()
      );
      await user.type(screen.getByLabelText("Pixel size, nm per pixel"), "4.2");
      submitUploadForm();

      await waitFor(() => expect(uploadAsset).toHaveBeenCalledTimes(2));
      expect(
        vi.mocked(uploadAsset).mock.calls.map(([file, options]) => [
          file.name,
          options?.pixelSizeNm,
        ])
      ).toEqual([
        // Nothing sent, so the server keeps the file's own 5 nm/px.
        ["declares.tif", null],
        ["silent.png", 4.2],
      ]);
    });

    it("will overwrite them when the user ticks the box that says so", async () => {
      const user = userEvent.setup();
      render(<ImageUploadPanel />);
      await openPanelWithServerFormats();

      dropFiles([tiffFile(5, "declares.tif"), pngFile("silent.png")]);
      await screen.findByText("silent.png");
      await user.type(screen.getByLabelText("Pixel size, nm per pixel"), "4.2");
      await user.click(
        await screen.findByLabelText(/replacing the pixel size 1 image declares/i)
      );
      submitUploadForm();

      await waitFor(() => expect(uploadAsset).toHaveBeenCalledTimes(2));
      for (const [, options] of vi.mocked(uploadAsset).mock.calls) {
        expect(options?.pixelSizeNm).toBe(4.2);
      }
    });

    /**
     * One corrupt TIFF in a plate of forty is a fact about that TIFF. The
     * queue carries on, the failure is named where it happened, and the
     * button retries exactly the ones that failed.
     */
    it("keeps going when one file fails, and says which", async () => {
      const onUploaded = vi.fn();
      vi.mocked(uploadAsset)
        .mockResolvedValueOnce(makeAsset())
        .mockRejectedValueOnce(
          new Error("Error reading TIFF file: not a TIFF file.")
        )
        .mockResolvedValueOnce(makeAsset());
      render(<ImageUploadPanel onUploaded={onUploaded} />);
      await openPanelWithServerFormats();

      dropFiles([pngFile("a.png"), pngFile("bad.png"), pngFile("c.png")]);
      await screen.findByText("c.png");
      submitUploadForm();

      await waitFor(() => expect(uploadAsset).toHaveBeenCalledTimes(3));
      expect(onUploaded).toHaveBeenCalledTimes(2);
      expect(await screen.findByRole("alert")).toHaveTextContent(
        "Error reading TIFF file: not a TIFF file."
      );
      // The two that worked have left the queue; the one that did not is still
      // here, on its own, ready to be retried.
      await waitFor(() =>
        expect(screen.getAllByTestId("import-chosen-file")).toHaveLength(1)
      );
      expect(screen.getByTestId("import-chosen-file")).toHaveTextContent("bad.png");
      expect(screen.getByTestId("import-batch-summary")).toHaveTextContent(
        "Imported 2 of 3 images."
      );
      expect(
        screen.getByRole("button", { name: "Try it again" })
      ).toBeInTheDocument();
    });

    it("counts the runs across the whole batch, not per image", async () => {
      serveCatalogue([pack()]);
      const user = userEvent.setup();
      render(<ImageUploadPanel />);
      await openPanelWithServerFormats();

      dropFiles([pngFile("a.png"), pngFile("b.png"), pngFile("c.png")]);
      await screen.findByText("c.png");
      await openRunOptions(user);
      await tickOrganelle(user, /segment mitochondria/i);

      expect(
        screen.getByRole("button", { name: /start a segmentation run now/i })
      ).toHaveTextContent("3 runs selected (1 per image)");
      expect(
        screen.getByRole("button", {
          name: "Import 3 images and start 3 uncalibrated runs",
        })
      ).toBeInTheDocument();
    });

    it("does not queue the same file twice", async () => {
      render(<ImageUploadPanel />);
      await openPanelWithServerFormats();

      // The same file, dropped again -- name, size and modification time all
      // match, which is what dropping the same folder twice looks like.
      const alreadyQueued = pngFile("b.png");
      dropFiles([pngFile("a.png"), alreadyQueued]);
      await screen.findByText("b.png");
      dropFiles([alreadyQueued, pngFile("c.png")]);

      expect(await screen.findByText("c.png")).toBeInTheDocument();
      expect(screen.getAllByTestId("import-file-row")).toHaveLength(3);
      expect(await screen.findByRole("alert")).toHaveTextContent(
        "b.png is already in this list."
      );
    });

    it("refuses the unreadable ones and still queues the rest", async () => {
      render(<ImageUploadPanel />);
      await openPanelWithServerFormats();

      dropFiles([
        pngFile("a.png"),
        new File(["x"], "notes.txt", { type: "text/plain" }),
        pngFile("c.png"),
      ]);

      expect(await screen.findByRole("alert")).toHaveTextContent(
        "notes.txt is not a supported format."
      );
      expect(screen.getAllByTestId("import-file-row")).toHaveLength(2);
    });
  });

  /**
   * The server has always refused an upload over `QUANTEM_MAX_UPLOAD_BYTES`,
   * and the user could never see it: waitress rejects from the request headers
   * and closes the socket while the browser is still streaming, which the
   * browser reports as a plain network error, minutes into an upload that was
   * never going to be accepted. `/api/system/status/` now publishes the number.
   */
  describe("a file the server cannot accept", () => {
    it("is refused in the picker, naming its size and the limit", async () => {
      vi.mocked(getSystemStatus).mockResolvedValue(systemStatus(4096 + 2048));
      render(<ImageUploadPanel />);
      await openPanelWithServerFormats();

      dropFiles([fileOfSize(4096, "enormous.tif")]);

      expect(await screen.findByRole("alert")).toHaveTextContent(
        "enormous.tif is 4.0 KB. This build imports files up to 6.0 KB, so it was not added."
      );
      expect(screen.getByTestId("import-drop-zone")).toBeInTheDocument();
      expect(uploadAsset).not.toHaveBeenCalled();
    });

    it("does not take the rest of the plate down with it", async () => {
      vi.mocked(getSystemStatus).mockResolvedValue(systemStatus(4096 + 2048));
      render(<ImageUploadPanel />);
      await openPanelWithServerFormats();

      dropFiles([pngFile("small.png"), fileOfSize(8192, "enormous.tif")]);

      expect(await screen.findByRole("alert")).toHaveTextContent(
        "enormous.tif is 8.0 KB."
      );
      expect(screen.getByTestId("import-chosen-file")).toHaveTextContent(
        "small.png"
      );
    });

    /**
     * A server that has not said must not have a limit invented for it: an
     * older build, or a status request that failed, would otherwise start
     * refusing imports it would have accepted.
     */
    it("checks nothing when the server has not published a limit", async () => {
      vi.mocked(getSystemStatus).mockResolvedValue(systemStatus(null));
      render(<ImageUploadPanel />);
      await openPanelWithServerFormats();

      dropFiles([fileOfSize(8192, "enormous.tif")]);

      expect(await screen.findByText("enormous.tif")).toBeInTheDocument();
      expect(screen.queryByRole("alert")).toBeNull();
    });
  });
});
