import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SegmentationCreatePanel } from "@/features/viewer/components/SegmentationCreatePanel";
import { createAssetSegmentation } from "@/shared/api/assets";
import { server } from "@/test/msw/server";
import { EMPTY_MODEL_CATALOGUE } from "@/test/msw/handlers";
import type { ModelPack } from "@/shared/types/finetune";

vi.mock("@/shared/api/assets", async () => {
  const actual =
    await vi.importActual<typeof import("@/shared/api/assets")>(
      "@/shared/api/assets"
    );
  return { ...actual, createAssetSegmentation: vi.fn() };
});

function mitoPack(overrides: Partial<ModelPack> = {}): ModelPack {
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

describe("SegmentationCreatePanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("offers the four organelle presets plus the tissue mask", () => {
    render(<SegmentationCreatePanel imageId="img-1" />);

    expect(screen.getByRole("button", { name: "Mitochondria" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "ER" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Nucleus" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Lipid Droplets" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Tissue Mask" })).toBeInTheDocument();
  });

  it("does not offer segmentation types QuantEM dropped", () => {
    render(<SegmentationCreatePanel imageId="img-1" />);

    expect(screen.queryByRole("button", { name: "Cells" })).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Secretory Granules" })
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /DeepContact/ })
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /CDeep3M|Cellpose/ })
    ).not.toBeInTheDocument();
  });

  it("filters out quick options whose segmentation type already exists", () => {
    render(
      <SegmentationCreatePanel
        imageId="img-1"
        existingSegmentationTypes={["Mitochondria", "Endoplasmic Reticulum"]}
      />
    );

    expect(screen.queryByRole("button", { name: "Mitochondria" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "ER" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Nucleus" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Lipid Droplets" })).toBeInTheDocument();
  });

  it("does not offer a membrane refinement toggle", () => {
    render(<SegmentationCreatePanel imageId="img-1" />);

    expect(
      screen.queryByLabelText(/membrane refinement/i)
    ).not.toBeInTheDocument();
  });

  it("asks before creating, because creating queues a whole-image run", async () => {
    // Creating a segmentation POSTs to an endpoint that enqueues a full-image
    // inference pass on the spot -- ~50s of CPU -- and nothing asked.
    const user = userEvent.setup();
    render(<SegmentationCreatePanel imageId="img-1" />);

    await user.click(screen.getByRole("button", { name: "Mitochondria" }));

    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    expect(
      screen.getByText(/queues one inference pass over the whole image/i)
    ).toBeInTheDocument();
    expect(createAssetSegmentation).not.toHaveBeenCalled();
  });

  it("creates only after the run is confirmed", async () => {
    const user = userEvent.setup();
    vi.mocked(createAssetSegmentation).mockResolvedValue({
      id: "seg-1",
    } as never);
    render(<SegmentationCreatePanel imageId="img-1" />);

    await user.click(screen.getByRole("button", { name: "Mitochondria" }));
    await user.click(await screen.findByRole("button", { name: /Create and run/ }));

    await waitFor(() => {
      expect(createAssetSegmentation).toHaveBeenCalledWith("img-1", {
        segmentation_type_name: "Mitochondria",
      });
    });
  });

  it("creates nothing when the confirmation is cancelled", async () => {
    const user = userEvent.setup();
    render(<SegmentationCreatePanel imageId="img-1" />);

    await user.click(screen.getByRole("button", { name: "Mitochondria" }));
    await user.click(await screen.findByRole("button", { name: "Cancel" }));

    expect(createAssetSegmentation).not.toHaveBeenCalled();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("warns up front when the model that would run cannot run", async () => {
    // The clean-install path: the run is queued, fails on a missing encoder,
    // and the queue banner replaces the message that said why.
    server.use(
      http.get("http://127.0.0.1:8000/api/models/", () =>
        HttpResponse.json({ ...EMPTY_MODEL_CATALOGUE, packs: [mitoPack()] })
      )
    );
    const user = userEvent.setup();
    render(<SegmentationCreatePanel imageId="img-1" />);

    await user.click(screen.getByRole("button", { name: "Mitochondria" }));

    expect(
      await screen.findByText(/quantem:mito cannot run on this machine/i)
    ).toBeInTheDocument();
    expect(screen.getByText(/Not installed yet\./)).toBeInTheDocument();
    // Still creatable: annotating by hand is a legitimate reason to want it.
    expect(screen.getByRole("button", { name: "Create anyway" })).toBeInTheDocument();
  });

  /**
   * Paper-cut 3: the dialog offered only the default family's pack. On a
   * machine with only OmniEM — Mitochondria installed it said "quantem:mito
   * cannot run on this machine — Create anyway", never mentioning the
   * installed model that the labeling screen's picker would have offered one
   * click later. The choice belongs in this dialog, where the run is queued.
   */
  describe("offering both families", () => {
    const bothFamilies = [
      mitoPack(), // quantem:mito, not installed
      mitoPack({
        id: "omniem:mito",
        family: "omniem",
        title: "OmniEM — Mitochondria",
        installed: true,
        runnable: true,
        reason: null,
      }),
    ];

    function useCatalogue(packs = bothFamilies) {
      server.use(
        http.get("http://127.0.0.1:8000/api/models/", () =>
          HttpResponse.json({ ...EMPTY_MODEL_CATALOGUE, packs })
        )
      );
    }

    it("preselects the installed family when the default cannot run", async () => {
      useCatalogue();
      const user = userEvent.setup();
      render(<SegmentationCreatePanel imageId="img-1" />);

      await user.click(screen.getByRole("button", { name: "Mitochondria" }));

      // Never "cannot run — Create anyway" while an installed alternative
      // exists: the alternative is selected, said out loud, and runnable.
      const picker = await screen.findByRole("combobox", {
        name: "Model to run",
      });
      expect(picker).toHaveValue("omniem:mito");
      expect(
        screen.getByText(/quantem:mito cannot run on this machine/)
      ).toBeInTheDocument();
      expect(
        screen.getByText(/the installed omniem:mito is selected instead/)
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Create and run" })
      ).toBeInTheDocument();
      expect(
        screen.queryByRole("button", { name: "Create anyway" })
      ).not.toBeInTheDocument();
    });

    it("creates with the selected family's source model", async () => {
      useCatalogue();
      const user = userEvent.setup();
      vi.mocked(createAssetSegmentation).mockResolvedValue({
        id: "seg-1",
      } as never);
      render(<SegmentationCreatePanel imageId="img-1" />);

      await user.click(screen.getByRole("button", { name: "Mitochondria" }));
      await user.click(
        await screen.findByRole("button", { name: "Create and run" })
      );

      await waitFor(() => {
        expect(createAssetSegmentation).toHaveBeenCalledWith("img-1", {
          segmentation_type_name: "Mitochondria",
          source_model: "omniem:mito",
        });
      });
    });

    it("still lets the user pick the blocked default, saying what that costs", async () => {
      useCatalogue();
      const user = userEvent.setup();
      render(<SegmentationCreatePanel imageId="img-1" />);

      await user.click(screen.getByRole("button", { name: "Mitochondria" }));
      await user.selectOptions(
        await screen.findByRole("combobox", { name: "Model to run" }),
        "quantem:mito"
      );

      expect(
        await screen.findByRole("button", { name: "Create anyway" })
      ).toBeInTheDocument();
      expect(screen.getByText(/Not installed yet\./)).toBeInTheDocument();
    });

    it("keeps the default selected when both families can run", async () => {
      useCatalogue([
        mitoPack({ installed: true, runnable: true, reason: null }),
        bothFamilies[1],
      ]);
      const user = userEvent.setup();
      render(<SegmentationCreatePanel imageId="img-1" />);

      await user.click(screen.getByRole("button", { name: "Mitochondria" }));

      const picker = await screen.findByRole("combobox", {
        name: "Model to run",
      });
      expect(picker).toHaveValue("quantem:mito");
      // Nothing was substituted, so nothing claims to have been.
      expect(
        screen.queryByText(/is selected instead/)
      ).not.toBeInTheDocument();
    });

    it("offers no picker when the catalogue knows only one pack", async () => {
      useCatalogue([mitoPack()]);
      const user = userEvent.setup();
      render(<SegmentationCreatePanel imageId="img-1" />);

      await user.click(screen.getByRole("button", { name: "Mitochondria" }));

      expect(await screen.findByRole("dialog")).toBeInTheDocument();
      expect(
        screen.queryByRole("combobox", { name: "Model to run" })
      ).not.toBeInTheDocument();
    });
  });

  // The dialog used to be byte-identical whether or not the image had a pixel
  // size, even though six of the eight packs resample to a canonical_nm and
  // `predict_region` silently falls back to native scale without one. That is
  // the condition that made every downstream number wrong, and this is the last
  // gate before the user spends the minute.
  it("says what running an uncalibrated image costs before the run is queued", async () => {
    server.use(
      http.get("http://127.0.0.1:8000/api/models/", () =>
        HttpResponse.json({
          ...EMPTY_MODEL_CATALOGUE,
          packs: [mitoPack({ installed: true, runnable: true, reason: null })],
        })
      )
    );
    const user = userEvent.setup();
    render(<SegmentationCreatePanel imageId="img-1" pixelSizeNm={null} />);

    await user.click(screen.getByRole("button", { name: "Mitochondria" }));

    expect(await screen.findByText(/This image has no pixel size/i)).toBeInTheDocument();
    expect(
      screen.getByText(
        /quantem:mito will run at native scale instead of the 8 nm\/px it was trained at/i
      )
    ).toBeInTheDocument();
    // The destructive-by-omission choice is named on the button too.
    expect(
      screen.getByRole("button", { name: "Create and run uncalibrated" })
    ).toBeInTheDocument();
  });

  it("says nothing about scale when the image is calibrated", async () => {
    server.use(
      http.get("http://127.0.0.1:8000/api/models/", () =>
        HttpResponse.json({
          ...EMPTY_MODEL_CATALOGUE,
          packs: [mitoPack({ installed: true, runnable: true, reason: null })],
        })
      )
    );
    const user = userEvent.setup();
    render(<SegmentationCreatePanel imageId="img-1" pixelSizeNm={5} />);

    await user.click(screen.getByRole("button", { name: "Mitochondria" }));

    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    expect(screen.queryByText(/no pixel size/i)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create and run" })).toBeInTheDocument();
  });

  it("says nothing about scale for a pack that runs at native resolution by design", async () => {
    // Both ER packs declare `canonical_nm: null`. Warning there would be false.
    server.use(
      http.get("http://127.0.0.1:8000/api/models/", () =>
        HttpResponse.json({
          ...EMPTY_MODEL_CATALOGUE,
          packs: [
            mitoPack({
              id: "quantem:er",
              organelle: "er",
              title: "QuantEM — Endoplasmic Reticulum",
              canonical_nm: null,
              installed: true,
              runnable: true,
              reason: null,
            }),
          ],
        })
      )
    );
    const user = userEvent.setup();
    render(<SegmentationCreatePanel imageId="img-1" pixelSizeNm={null} />);

    await user.click(screen.getByRole("button", { name: "ER" }));

    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    expect(screen.queryByText(/no pixel size/i)).not.toBeInTheDocument();
  });

  it("stays silent when the caller does not know the pixel size", async () => {
    // `undefined` is "this screen cannot say", which is not the same claim as
    // "this image is uncalibrated".
    server.use(
      http.get("http://127.0.0.1:8000/api/models/", () =>
        HttpResponse.json({
          ...EMPTY_MODEL_CATALOGUE,
          packs: [mitoPack({ installed: true, runnable: true, reason: null })],
        })
      )
    );
    const user = userEvent.setup();
    render(<SegmentationCreatePanel imageId="img-1" />);

    await user.click(screen.getByRole("button", { name: "Mitochondria" }));

    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    expect(screen.queryByText(/no pixel size/i)).not.toBeInTheDocument();
  });

  /**
   * Paper-cut 8: the model and device render as chips inside the sentence, so
   * read as a unit — a screen reader, or anything walking the accessibility
   * tree — the paragraph came out as "The run will use on .". The names must
   * be part of one accessible sentence; the chips stay as presentation.
   */
  it("carries the model and device in one accessible sentence, not chip fragments", async () => {
    server.use(
      http.get("http://127.0.0.1:8000/api/models/", () =>
        HttpResponse.json({
          ...EMPTY_MODEL_CATALOGUE,
          packs: [mitoPack({ installed: true, runnable: true, reason: null })],
        })
      )
    );
    const user = userEvent.setup();
    render(<SegmentationCreatePanel imageId="img-1" pixelSizeNm={5} />);

    await user.click(screen.getByRole("button", { name: "Mitochondria" }));

    const dialog = await screen.findByRole("dialog");
    const accessibleSentence = dialog.querySelector(".sr-only");
    expect(accessibleSentence).toHaveTextContent(
      "The run will use quantem:mito on CPU."
    );
    // The chip copy is presentation only, or the reader hears it twice.
    const chip = screen.getByText("quantem:mito", { selector: "strong" });
    expect(chip.closest("[aria-hidden='true']")).not.toBeNull();
  });

  it("creates a manual-only mask with no confirmation, because it queues nothing", async () => {
    const user = userEvent.setup();
    vi.mocked(createAssetSegmentation).mockResolvedValue({ id: "seg-2" } as never);
    render(<SegmentationCreatePanel imageId="img-1" />);

    await user.click(screen.getByRole("button", { name: "Tissue Mask" }));

    await waitFor(() => {
      expect(createAssetSegmentation).toHaveBeenCalledWith("img-1", {
        segmentation_type_name: "Tissue Mask",
      });
    });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
