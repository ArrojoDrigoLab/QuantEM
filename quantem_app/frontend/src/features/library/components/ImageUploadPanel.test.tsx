import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ImageUploadPanel } from "@/features/library/components/ImageUploadPanel";
import { uploadAsset } from "@/shared/api/assets";
import { getSystemStatus } from "@/shared/api/jobs";
import type { AssetDetail } from "@/shared/types/images";
import type { SystemStatus } from "@/shared/types/jobs";

vi.mock("@/shared/api/assets", async () => {
  const actual = await vi.importActual<typeof import("@/shared/api/assets")>(
    "@/shared/api/assets"
  );
  return { ...actual, uploadAsset: vi.fn() };
});

vi.mock("@/shared/api/jobs", async () => {
  const actual = await vi.importActual<typeof import("@/shared/api/jobs")>(
    "@/shared/api/jobs"
  );
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
    pixel_size_nm: null,
    preprocess_stage: "ENCODING",
    preprocess_progress: 0,
    tags: [],
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
}

function pngFile(name = "scan.png"): File {
  return new File(["fake"], name, { type: "image/png" });
}

function systemStatus(): SystemStatus {
  return {
    cuda_available: false,
    supported_upload_formats: [".tif", ".tiff", ".png"],
  };
}

async function openPanelWithServerFormats() {
  const input = await screen.findByLabelText("Image file");
  await waitFor(() => expect(input).toHaveAttribute("accept", ".tif,.tiff,.png"));
  return input;
}

function submitUploadForm() {
  fireEvent.submit(screen.getByTestId("import-form"));
}

describe("ImageUploadPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getSystemStatus).mockResolvedValue(systemStatus());
    vi.mocked(uploadAsset).mockResolvedValue(makeAsset());
  });

  it("uses the formats reported by the server", async () => {
    render(<ImageUploadPanel />);

    const input = await openPanelWithServerFormats();
    expect(input).toHaveAttribute("multiple");
    expect(screen.getByText(/\.tif, \.tiff, \.png/)).toBeInTheDocument();
  });

  it("shows a compact drop area after files have been added and accepts more files there", async () => {
    const user = userEvent.setup();
    render(<ImageUploadPanel />);

    await user.upload(await openPanelWithServerFormats(), pngFile("first.png"));
    const additionalZone = await screen.findByTestId("import-add-more-drop-zone");
    expect(additionalZone).toHaveTextContent("Drop more images here");
    expect(additionalZone.tagName).toBe("LABEL");

    fireEvent.drop(additionalZone, {
      dataTransfer: { files: [pngFile("second.png")], types: ["Files"] },
    });

    expect(await screen.findByText("second.png")).toBeInTheDocument();
    expect(screen.getAllByTestId("import-file-row")).toHaveLength(2);
  });

  it("defaults every import to a dated new experiment and Dataset 1", async () => {
    const user = userEvent.setup();
    render(<ImageUploadPanel />);

    await user.upload(await openPanelWithServerFormats(), pngFile());

    expect(
      (screen.getByLabelText("New experiment…") as HTMLInputElement).value
    ).toMatch(/^New Experiment \d{4}-\d{2}-\d{2}$/);
    expect(screen.getByLabelText("New dataset…")).toHaveValue("Dataset 1");

    submitUploadForm();

    await waitFor(() => expect(uploadAsset).toHaveBeenCalledTimes(1));
    expect(vi.mocked(uploadAsset).mock.calls[0][1]).toMatchObject({
      experimentName: expect.stringMatching(/^New Experiment \d{4}-\d{2}-\d{2}$/),
      datasetName: "Dataset 1",
    });
  });

  it("applies the default experiment and dataset to every image in a batch", async () => {
    const user = userEvent.setup();
    render(<ImageUploadPanel />);

    await user.upload(await openPanelWithServerFormats(), [
      pngFile("first.png"),
      pngFile("second.png"),
    ]);
    submitUploadForm();

    await waitFor(() => expect(uploadAsset).toHaveBeenCalledTimes(2));
    for (const [, options] of vi.mocked(uploadAsset).mock.calls) {
      expect(options).toMatchObject({
        experimentName: expect.stringMatching(/^New Experiment \d{4}-\d{2}-\d{2}$/),
        datasetName: "Dataset 1",
      });
    }
  });

  it("puts the metadata explanation under pixel size and exposes its tooltip", async () => {
    const user = userEvent.setup();
    render(<ImageUploadPanel />);

    await user.upload(await openPanelWithServerFormats(), pngFile());

    const pixelSize = screen.getByLabelText("Pixel size, nm per pixel");
    expect(pixelSize.nextElementSibling).toHaveTextContent(
      "Could not parse 1 image resolution from its metadata. Enter value here if known."
    );
    expect(
      screen.getByTitle(
        "Optional, you can change or set these later. Required for some analysis measurements"
      )
    ).toBeInTheDocument();
  });

  it("places a three-line Notes (optional) box below the experiment and dataset", async () => {
    const user = userEvent.setup();
    render(<ImageUploadPanel />);

    await user.upload(await openPanelWithServerFormats(), pngFile());

    const notes = screen.getByLabelText("Notes (optional)");
    expect(notes.tagName).toBe("TEXTAREA");
    expect(notes).toHaveAttribute("rows", "3");
    expect(
      screen.getByLabelText("Experiment").compareDocumentPosition(notes)
    ).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
  });

  it("sends pixel size and notes for every image", async () => {
    const user = userEvent.setup();
    render(<ImageUploadPanel />);

    await user.upload(await openPanelWithServerFormats(), [
      pngFile("first.png"),
      pngFile("second.png"),
    ]);
    await user.type(screen.getByLabelText("Pixel size, nm per pixel"), "4.2");
    await user.type(screen.getByLabelText("Notes (optional)"), "day 14");
    submitUploadForm();

    await waitFor(() => expect(uploadAsset).toHaveBeenCalledTimes(2));
    for (const [, options] of vi.mocked(uploadAsset).mock.calls) {
      expect(options).toMatchObject({ pixelSizeNm: 4.2, notes: "day 14" });
    }
  });

  it("rejects a non-positive pixel size before starting the upload", async () => {
    const user = userEvent.setup();
    render(<ImageUploadPanel />);

    await user.upload(await openPanelWithServerFormats(), pngFile());
    await user.type(screen.getByLabelText("Pixel size, nm per pixel"), "0");
    submitUploadForm();

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Pixel size must be greater than zero."
    );
    expect(uploadAsset).not.toHaveBeenCalled();
  });

  it("does not offer or queue segmentation during import", async () => {
    const user = userEvent.setup();
    render(<ImageUploadPanel />);

    await user.upload(await openPanelWithServerFormats(), pngFile());
    expect(
      screen.queryByRole("button", { name: /start a segmentation run now/i })
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Import image" })).toBeInTheDocument();

    submitUploadForm();

    await waitFor(() => expect(uploadAsset).toHaveBeenCalledTimes(1));
    const options = vi.mocked(uploadAsset).mock.calls[0][1];
    expect(options).not.toHaveProperty("segmentMito");
    expect(options).not.toHaveProperty("segmentEr");
    expect(options).not.toHaveProperty("segmentNucleus");
    expect(options).not.toHaveProperty("segmentLd");
  });

  it("keeps the drop target available after a successful import", async () => {
    const user = userEvent.setup();
    render(<ImageUploadPanel />);

    await user.upload(await openPanelWithServerFormats(), pngFile());
    submitUploadForm();

    await waitFor(() => expect(uploadAsset).toHaveBeenCalledTimes(1));
    expect(await screen.findByTestId("import-drop-zone")).toBeInTheDocument();
    expect(screen.getByText("Drop your images here")).toBeInTheDocument();
  });
});
