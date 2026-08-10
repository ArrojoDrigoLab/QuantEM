import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ImageUploadPanel } from "@/features/library/components/ImageUploadPanel";
import { uploadAsset } from "@/shared/api/assets";
import { getSystemStatus } from "@/shared/api/jobs";
import { server } from "@/test/msw/server";
import { EMPTY_MODEL_CATALOGUE } from "@/test/msw/handlers";
import type { ModelPack } from "@/shared/types/finetune";
import type { AssetDetail } from "@/shared/types/images";

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
 * Anchored on the file input rather than on the submit button's caption: that
 * caption changes to name the uncalibrated choice, and a helper that has to
 * know which wording is on screen would break every test that is not about it.
 */
function submitUploadForm() {
  const form = screen.getByLabelText(/image file/i).closest("form");
  if (!form) throw new Error("Upload form not found.");
  fireEvent.submit(form);
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
    vi.mocked(getSystemStatus).mockResolvedValue({
      cuda_available: false,
      supported_upload_formats: [".tif", ".tiff", ".png"],
    });
    vi.mocked(uploadAsset).mockResolvedValue(makeAsset());
  });

  it("drives the accepted formats from supported_upload_formats", async () => {
    render(<ImageUploadPanel defaultExpanded />);

    const input = await screen.findByLabelText(/image file/i);
    await waitFor(() => {
      expect(input).toHaveAttribute("accept", ".tif,.tiff,.png");
    });
    expect(screen.getByText(/\.tif, \.tiff, \.png/)).toBeInTheDocument();
  });

  it("accepts a PNG, which the server accepts too", async () => {
    const user = userEvent.setup();
    render(<ImageUploadPanel defaultExpanded onUploaded={vi.fn()} />);

    const input = await openPanelWithServerFormats();
    await user.upload(input, pngFile());

    expect(screen.getByText(/scan\.png/)).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
    // The display name is derived by stripping a *known* extension.
    expect(screen.getByLabelText(/display name/i)).toHaveValue("scan");
  });

  it("rejects an unsupported extension inline rather than with alert()", async () => {
    const alertSpy = vi.spyOn(window, "alert").mockImplementation(() => {});
    // applyAccept: false so the component's own guard runs -- a drag-drop or a
    // browser that ignores `accept` reaches exactly this path.
    const user = userEvent.setup({ applyAccept: false });
    render(<ImageUploadPanel defaultExpanded />);

    const input = await openPanelWithServerFormats();
    await user.upload(input, new File(["x"], "notes.txt", { type: "text/plain" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /not a supported format/i
    );
    expect(alertSpy).not.toHaveBeenCalled();
  });

  it("sends a typed pixel size with the upload", async () => {
    const user = userEvent.setup();
    render(<ImageUploadPanel defaultExpanded onUploaded={vi.fn()} />);

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
    render(<ImageUploadPanel defaultExpanded onUploaded={vi.fn()} />);

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

  it("blocks a non-positive pixel size before it reaches the server", async () => {
    const user = userEvent.setup();
    render(<ImageUploadPanel defaultExpanded onUploaded={vi.fn()} />);

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
      render(<ImageUploadPanel defaultExpanded />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
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
      render(<ImageUploadPanel defaultExpanded />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
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
      render(<ImageUploadPanel defaultExpanded />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      // Tick and untick, so this is a form that has been looked at rather than
      // one that simply started empty.
      const mito = await tickOrganelle(user, /segment mitochondria/i);
      await screen.findByText(/quantem:mito \(8 nm\/px\)/);
      await user.click(mito);

      await waitFor(() => {
        expect(
          screen.queryByText(/changes which objects exist/i)
        ).not.toBeInTheDocument();
      });
      expect(screen.getByRole("button", { name: "Upload" })).toBeInTheDocument();
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
      render(<ImageUploadPanel defaultExpanded />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
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
      render(<ImageUploadPanel defaultExpanded />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      await tickOrganelle(user, /segment mitochondria/i);
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
      render(<ImageUploadPanel defaultExpanded />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
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
      render(<ImageUploadPanel defaultExpanded />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, tiffFile(5));
      // Ticked deliberately: with nothing ticked there is no run to warn about
      // and this test would pass without reading the file at all.
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
      render(<ImageUploadPanel defaultExpanded />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, tiffFile(5));
      await screen.findByText(/stack\.tif declares 5 nm\/px/);
      await user.type(screen.getByLabelText(/pixel size/i), "8");

      expect(
        await screen.findByText(/The value typed above is used instead/)
      ).toBeInTheDocument();
    });

    it("still states it flatly for a TIFF that declares nothing", async () => {
      // The case the warning exists for. Read, understood, and empty -- so it
      // keeps the unhedged wording and the unhedged button.
      serveCatalogue([pack()]);
      const user = userEvent.setup();
      render(<ImageUploadPanel defaultExpanded />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, tiffFile(null, "bare.tif"));
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
      render(<ImageUploadPanel defaultExpanded />);

      const input = await openPanelWithServerFormats();
      // BigTIFF: a real format this build does not parse here.
      const header = new DataView(new ArrayBuffer(16));
      header.setUint16(0, 0x4949, true);
      header.setUint16(2, 43, true);
      await user.upload(input, new File([header.buffer], "huge.tif"));
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
      render(<ImageUploadPanel defaultExpanded />);

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
      render(<ImageUploadPanel defaultExpanded />);

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
      render(<ImageUploadPanel defaultExpanded onUploaded={vi.fn()} />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      expect(screen.getByRole("button", { name: "Upload" })).toBeInTheDocument();
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
      render(<ImageUploadPanel defaultExpanded />);

      const nucleus = await screen.findByLabelText(/segment nucleus/i);
      await waitFor(() => expect(nucleus).toBeDisabled());
      expect(screen.getByText(/Not installed yet\./)).toBeInTheDocument();
    });

    it("leaves a pack the catalogue says nothing about enabled", async () => {
      // "Unknown" is not "blocked": offer it, and do not claim it works.
      serveCatalogue([pack()]);
      render(<ImageUploadPanel defaultExpanded />);

      const ld = await screen.findByLabelText(/segment lipid droplets/i);
      await waitFor(() => expect(ld).toBeEnabled());
      expect(ld).not.toBeChecked();
    });

    it("does not tick a box when the catalogue lands", async () => {
      // The catalogue arrives after the first render. It used to bring the
      // default with it, so a box could tick itself under the user's cursor.
      serveCatalogue([pack()]);
      const user = userEvent.setup();
      render(<ImageUploadPanel defaultExpanded />);

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
      render(<ImageUploadPanel defaultExpanded onUploaded={vi.fn()} />);

      const input = await openPanelWithServerFormats();
      await user.upload(input, pngFile());
      await tickOrganelle(user, /segment mitochondria/i);
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
      render(<ImageUploadPanel defaultExpanded />);

      const input = await openPanelWithServerFormats();
      // Calibrated, so the uncalibrated wording is out of the way and this is
      // purely about the count. This is exactly the case that read "Upload".
      await user.upload(input, pngFile());
      await user.type(screen.getByLabelText(/pixel size/i), "5");
      await tickOrganelle(user, /segment mitochondria/i);

      expect(
        screen.getByRole("button", { name: "Import and start 1 segmentation run" })
      ).toBeInTheDocument();

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
      render(<ImageUploadPanel defaultExpanded />);

      expect(await screen.findByLabelText(/^notes/i)).toBeInTheDocument();
      expect(screen.queryByLabelText(/tags/i)).not.toBeInTheDocument();
    });

    it("says what typing there achieves", async () => {
      render(<ImageUploadPanel defaultExpanded />);

      expect(
        await screen.findByText(/search box matches notes as well as names/i)
      ).toBeInTheDocument();
    });

    it("sends what was typed, under the name the server reads", async () => {
      const user = userEvent.setup();
      render(<ImageUploadPanel defaultExpanded onUploaded={vi.fn()} />);

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
      render(<ImageUploadPanel defaultExpanded onUploaded={vi.fn()} />);

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
});
