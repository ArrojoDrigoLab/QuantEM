import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { PixelSizeEditor, PixelSizeTag } from "@/shared/ui/PixelSize";
import { updateAsset } from "@/shared/api/assets";
import { ApiRequestError } from "@/shared/api/core/http";
import type { AssetDetail } from "@/shared/types/images";

vi.mock("@/shared/api/assets", async () => {
  const actual =
    await vi.importActual<typeof import("@/shared/api/assets")>(
      "@/shared/api/assets"
    );
  return { ...actual, updateAsset: vi.fn() };
});

function makeAsset(overrides: Partial<AssetDetail> = {}): AssetDetail {
  return {
    id: "asset-1",
    file_path: "",
    original_filename: "liver01.tif",
    display_name: "Liver 01",
    is_eval_set: false,
    width: 1024,
    height: 1024,
    channels: 1,
    bit_depth: 8,
    pixel_size_nm: null,
    preprocess_stage: "DONE",
    preprocess_progress: 100,
    tags: [],
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

describe("PixelSizeEditor", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(updateAsset).mockImplementation(async (_id, updates) =>
      makeAsset({ pixel_size_nm: updates.pixel_size_nm ?? null })
    );
  });

  it("calibrates an image that arrived without a pixel size", async () => {
    // The blocker: PATCH /api/assets/<id>/ {"pixel_size_nm": 4.2} always worked
    // and nothing in the client ever called it, so an untagged EM export was
    // stuck at calibrated: false forever.
    const user = userEvent.setup();
    const onSaved = vi.fn();
    render(<PixelSizeEditor asset={makeAsset()} onSaved={onSaved} />);

    expect(screen.getByText("Pixel size not set")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Set pixel size" }));
    await user.type(screen.getByLabelText("Pixel size"), "4.2");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(updateAsset).toHaveBeenCalledWith("asset-1", { pixel_size_nm: 4.2 });
    });
    expect(onSaved).toHaveBeenCalledWith(
      expect.objectContaining({ pixel_size_nm: 4.2 })
    );
  });

  it("shows a file-derived value as a numeric-only resolution tag", () => {
    render(
      <PixelSizeEditor
        asset={makeAsset({
          pixel_size_nm: 5,
          renditions: [
            {
              id: "rend-1",
              type: "FULL",
              metadata: { source_metadata: { pixel_size_nm: 5 } },
            },
          ],
        })}
      />
    );

    expect(screen.getByText("5 nm/px")).toBeInTheDocument();
    expect(screen.queryByText(/from file|entered by hand/i)).not.toBeInTheDocument();
  });

  it("shows an edited value as the same numeric-only resolution tag", () => {
    render(
      <PixelSizeEditor
        asset={makeAsset({
          pixel_size_nm: 4.2,
          renditions: [
            {
              id: "rend-1",
              type: "FULL",
              metadata: { source_metadata: { pixel_size_nm: 5 } },
            },
          ],
        })}
      />
    );

    expect(screen.getByText("4.2 nm/px")).toBeInTheDocument();
    expect(screen.queryByText(/from file|entered by hand/i)).not.toBeInTheDocument();
  });

  it("clears the calibration when the field is emptied", async () => {
    const user = userEvent.setup();
    render(<PixelSizeEditor asset={makeAsset({ pixel_size_nm: 4.2 })} />);

    await user.click(screen.getByRole("button", { name: "Edit pixel size" }));
    await user.clear(screen.getByLabelText("Pixel size"));
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(updateAsset).toHaveBeenCalledWith("asset-1", { pixel_size_nm: null });
    });
  });

  it("rejects a non-positive value without calling the API", async () => {
    const user = userEvent.setup();
    render(<PixelSizeEditor asset={makeAsset()} />);

    await user.click(screen.getByRole("button", { name: "Set pixel size" }));
    await user.type(screen.getByLabelText("Pixel size"), "-1");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(
      await screen.findByText("Pixel size must be greater than zero.")
    ).toBeInTheDocument();
    expect(updateAsset).not.toHaveBeenCalled();
  });

  it("shows a server rejection as a sentence, not a document", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.mocked(updateAsset).mockRejectedValue(
      new ApiRequestError("<!DOCTYPE html><html><body>Forbidden</body></html>", {
        status: 403,
      })
    );
    const user = userEvent.setup();
    render(<PixelSizeEditor asset={makeAsset()} />);

    await user.click(screen.getByRole("button", { name: "Set pixel size" }));
    await user.type(screen.getByLabelText("Pixel size"), "4.2");
    await user.click(screen.getByRole("button", { name: "Save" }));

    const message = await screen.findByText(/could not be saved/i);
    expect(message.textContent).toContain("HTTP 403");
    expect(message.textContent).not.toContain("<");
  });
});

describe("PixelSizeTag", () => {
  it("normalizes calibrated values and contains no extra copy", () => {
    render(<PixelSizeTag valueNm={5} />);

    const tag = screen.getByText("5 nm/px");
    expect(tag).toHaveTextContent(/^5 nm\/px$/);
    expect(tag).not.toHaveAttribute("title");
  });

  it("uses the shared unset state when no resolution is available", () => {
    render(<PixelSizeTag valueNm={null} />);

    expect(screen.getByText("Pixel size not set")).toBeInTheDocument();
  });
});
