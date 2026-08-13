import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SegmentationCreatePanel } from "@/features/viewer/components/SegmentationCreatePanel";
import { createAssetSegmentation } from "@/shared/api/assets";
import type { ModelPack } from "@/shared/types/finetune";
import { server } from "@/test/msw/server";

vi.mock("@/shared/api/assets", async () => {
  const actual = await vi.importActual<typeof import("@/shared/api/assets")>(
    "@/shared/api/assets"
  );
  return { ...actual, createAssetSegmentation: vi.fn() };
});

function installedMitoPack(): ModelPack {
  return {
    id: "omniem:mito",
    family: "omniem",
    organelle: "mito",
    title: "OmniEM — Mitochondria",
    installed: true,
    download_bytes: 100,
    canonical_nm: 8,
    tile_size: 518,
    default_threshold: 0.5,
    decoder: "affinity_mws",
    neck: "naive_1x1",
    adapt: "lora",
    licence: "licence",
    notes: "",
    runnable: true,
    reason: null,
  };
}

function downloadableMitoPack(): ModelPack {
  return {
    ...installedMitoPack(),
    id: "quantem:mito",
    family: "quantem",
    title: "QuantEM — Mitochondria",
    installed: false,
    download_bytes: 1_234_000_000,
    runnable: false,
    reason: "Not installed yet.",
  };
}

function useCatalogue(packs: ModelPack[]) {
  server.use(
    http.get("http://127.0.0.1:8000/api/models/", () =>
      HttpResponse.json({ packs, adapted: [], device: null })
    )
  );
}

describe("SegmentationCreatePanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    server.use(
      http.get("http://127.0.0.1:8000/api/segmentation-types/", () =>
        HttpResponse.json([])
      )
    );
  });

  it("offers the four built-in organelles", () => {
    render(<SegmentationCreatePanel imageId="img-1" />);
    expect(screen.getByRole("button", { name: "Mitochondria" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "ER" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Nucleus" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Lipid Droplets" })).toBeInTheDocument();
  });

  it("filters built-ins that already exist on the image", () => {
    render(
      <SegmentationCreatePanel
        imageId="img-1"
        existingSegmentationTypes={["Mitochondria", "Endoplasmic Reticulum"]}
      />
    );

    expect(screen.queryByRole("button", { name: "Mitochondria" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "ER" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Nucleus" })).toBeInTheDocument();
  });

  it("allows manual segmentation when no model is downloaded", async () => {
    useCatalogue([]);
    const user = userEvent.setup();
    vi.mocked(createAssetSegmentation).mockResolvedValue({ id: "seg-1" } as never);
    render(<SegmentationCreatePanel imageId="img-1" />);

    await user.click(screen.getByRole("button", { name: "Mitochondria" }));
    expect(await screen.findByRole("dialog", { name: "Start mitochondria segmentation" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /^QuantEM\b/ })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /^OmniEM\b/ })).toBeInTheDocument();
    await user.click(screen.getByRole("radio", { name: "Manual segmentation" }));
    expect(screen.queryByText(/cannot run on this machine/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/The run will use/i)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Start manual segmentation" }));
    await waitFor(() =>
      expect(createAssetSegmentation).toHaveBeenCalledWith("img-1", {
        segmentation_type_name: "Mitochondria",
        run_inference: false,
        source_model: undefined,
      })
    );
  });

  it("preselects an installed model and explains immediate inference", async () => {
    useCatalogue([installedMitoPack()]);
    const user = userEvent.setup();
    render(
      <SegmentationCreatePanel
        imageId="img-1"
        imageSizeBytes={201 * 1024 * 1024}
      />
    );

    await user.click(screen.getByRole("button", { name: "Mitochondria" }));
    expect(await screen.findByRole("radio", { name: /^OmniEM\b/ })).toBeChecked();
    expect(
      screen.getByText(/selected model will run on all tiles/i)
    ).toBeInTheDocument();
    expect(screen.getByText(/larger than 200 MB/i)).toBeInTheDocument();
  });

  it("shows missing models and launches an automatic first-run download", async () => {
    useCatalogue([downloadableMitoPack(), installedMitoPack()]);
    const user = userEvent.setup();
    const onCreated = vi.fn();
    const created = { id: "seg-1" } as never;
    vi.mocked(createAssetSegmentation).mockResolvedValue(created);
    render(<SegmentationCreatePanel imageId="img-1" onCreated={onCreated} />);

    await user.click(screen.getByRole("button", { name: "Mitochondria" }));
    expect(await screen.findByRole("radio", { name: /^QuantEM\b/ })).toBeChecked();
    expect(
      screen.getByRole("img", {
        name: "Model is not downloaded. Will automatically download (1.2GB) on first run",
      })
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Run model" }));
    await waitFor(() =>
      expect(onCreated).toHaveBeenCalledWith(created, {
        sourceModel: "quantem:mito",
        runModel: true,
      })
    );
    expect(createAssetSegmentation).toHaveBeenCalledWith("img-1", {
      segmentation_type_name: "Mitochondria",
      run_inference: false,
      source_model: undefined,
    });
  });

  it("does not offer an installed pack that the backend says cannot run", async () => {
    useCatalogue([{ ...installedMitoPack(), runnable: false, reason: "Unsupported" }]);
    const user = userEvent.setup();
    render(<SegmentationCreatePanel imageId="img-1" />);

    await user.click(screen.getByRole("button", { name: "Mitochondria" }));
    expect(await screen.findByRole("radio", { name: "Manual segmentation" })).toBeChecked();
    expect(screen.getByRole("radio", { name: /^OmniEM\b/ })).toBeDisabled();
  });

  it("keeps manual segmentation available when a model is installed", async () => {
    useCatalogue([installedMitoPack()]);
    const user = userEvent.setup();
    vi.mocked(createAssetSegmentation).mockResolvedValue({ id: "seg-1" } as never);
    render(<SegmentationCreatePanel imageId="img-1" />);

    await user.click(screen.getByRole("button", { name: "Mitochondria" }));
    await user.click(await screen.findByRole("radio", { name: "Manual segmentation" }));
    expect(screen.queryByText(/selected model will run/i)).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Start manual segmentation" }));

    await waitFor(() =>
      expect(createAssetSegmentation).toHaveBeenCalledWith("img-1", {
        segmentation_type_name: "Mitochondria",
        run_inference: false,
        source_model: undefined,
      })
    );
  });

  it("creates reusable custom types without opening the model dialog", async () => {
    server.use(
      http.get("http://127.0.0.1:8000/api/segmentation-types/", () =>
        HttpResponse.json([
          {
            id: "custom-vesicles",
            internal_name: "Vesicles",
            short_name: "Vesicles",
            long_name: "Vesicles",
            kind: "custom",
            default_color: null,
            created_at: "2026-01-01T00:00:00Z",
            updated_at: "2026-01-01T00:00:00Z",
          },
        ])
      )
    );
    const user = userEvent.setup();
    vi.mocked(createAssetSegmentation).mockResolvedValue({ id: "seg-2" } as never);
    render(<SegmentationCreatePanel imageId="img-1" />);

    await user.click(await screen.findByRole("button", { name: "Vesicles" }));
    await waitFor(() =>
      expect(createAssetSegmentation).toHaveBeenCalledWith("img-1", {
        segmentation_type_id: "custom-vesicles",
      })
    );
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("creates image-specific analysis masks without model inference", async () => {
    const user = userEvent.setup();
    vi.mocked(createAssetSegmentation).mockResolvedValue({ id: "seg-2" } as never);
    render(<SegmentationCreatePanel imageId="img-1" />);

    await user.type(screen.getByLabelText("Mask name"), "Cells mask");
    await user.click(screen.getByRole("button", { name: "Create analysis mask" }));

    await waitFor(() =>
      expect(createAssetSegmentation).toHaveBeenCalledWith("img-1", {
        segmentation_type_name: "Analysis Segmentation Mask",
        analysis_name: "Cells mask",
      })
    );
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("keeps the reusable custom segmentation workflow intact", async () => {
    const user = userEvent.setup();
    vi.mocked(createAssetSegmentation).mockResolvedValue({ id: "seg-3" } as never);
    render(<SegmentationCreatePanel imageId="img-1" />);

    await user.click(screen.getByRole("button", { name: "Create custom segmentation" }));
    const dialog = await screen.findByRole("dialog");
    await user.type(screen.getByLabelText("Segmentation name"), "Vesicles");
    await user.click(screen.getByLabelText("Object-based segmentation"));
    await user.click(
      within(dialog).getByRole("button", { name: "Create custom segmentation" })
    );

    await waitFor(() =>
      expect(createAssetSegmentation).toHaveBeenCalledWith("img-1", {
        segmentation_type_name: "Vesicles",
        measurement_mode: "global",
      })
    );
  });
});
