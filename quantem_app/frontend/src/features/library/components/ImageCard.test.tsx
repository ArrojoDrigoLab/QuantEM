import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ImageCard } from "@/features/library/components/ImageCard";
import type { HomeEntry } from "@/shared/types/images";

function makeEntry(overrides: Partial<HomeEntry> = {}): HomeEntry {
  return {
    id: "11111111-2222-3333-4444-555555555555",
    display_name: "Liver 01",
    original_filename: "liver01.tif",
    metadata_summary: "1024x1024",
    width: 1024,
    height: 1024,
    pixel_size_nm: 4.2,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-02T00:00:00Z",
    preprocess_stage: "DONE",
    preprocess_progress: 100,
    ngff_ready: true,
    can_open: true,
    ...overrides,
  };
}

describe("ImageCard", () => {
  it("links through the hash route so copy-link and middle-click work", () => {
    render(<ImageCard image={makeEntry()} onOpen={vi.fn()} />);

    // HashRouter with `base: './'`: a bare "/assets/<id>/viewer" href navigates
    // for real on middle-click and lands on a white screen.
    const link = screen.getByRole("link", { name: "Liver 01" });
    expect(link.getAttribute("href")).toBe(
      "#/assets/11111111-2222-3333-4444-555555555555/viewer"
    );
  });

  it("still opens in-app on a plain click without navigating", async () => {
    const user = userEvent.setup();
    const onOpen = vi.fn();
    render(<ImageCard image={makeEntry()} onOpen={onOpen} />);

    await user.click(screen.getByRole("link", { name: "Liver 01" }));

    expect(onOpen).toHaveBeenCalledWith("11111111-2222-3333-4444-555555555555");
  });

  /**
   * The card used to render the badge in `compact` mode, which dropped the
   * provenance suffix entirely -- so a value read from the file and one typed
   * by a person were distinguishable only by the badge colour and a `title`
   * tooltip, neither of which a keyboard or touch user can reach. This is the
   * screen where images are compared side by side and it is the number every
   * downstream measurement is built on, so it is said in words.
   */
  it("says on the card where the pixel size came from, not just in a colour", () => {
    render(
      <ImageCard
        image={makeEntry({ pixel_size_nm: 4.2, file_declared_pixel_size_nm: 4.2 })}
        onOpen={vi.fn()}
      />
    );

    expect(screen.getByText("4.2 nm/px · from file")).toBeInTheDocument();
  });

  it("marks a hand-entered pixel size as such on the card", () => {
    render(
      <ImageCard
        image={makeEntry({ pixel_size_nm: 4.2, file_declared_pixel_size_nm: null })}
        onOpen={vi.fn()}
      />
    );

    // Exactly what the viewer header says about the same image.
    expect(screen.getByText("4.2 nm/px · entered by hand")).toBeInTheDocument();
  });

  /**
   * A payload with no `file_declared_pixel_size_nm` at all cannot say where the
   * value came from, and must not be relabelled "entered by hand" -- that
   * asserts the file declared nothing. It says it does not know instead of
   * going silent, because a bare number beside labelled neighbours reads as
   * whatever they say.
   */
  it("admits when the payload does not record the provenance", () => {
    render(<ImageCard image={makeEntry()} onOpen={vi.fn()} />);

    expect(screen.getByText("4.2 nm/px · source not recorded")).toBeInTheDocument();
  });

  it("flags an uncalibrated image on the card, not three screens in", () => {
    render(
      <ImageCard image={makeEntry({ pixel_size_nm: null })} onOpen={vi.fn()} />
    );

    expect(screen.getByText("Pixel size not set")).toBeInTheDocument();
  });

  // The grid is row-virtualised against a constant card height and the card is
  // `overflow: hidden`, so whichever block is allowed to grow decides what gets
  // cut off. It has to be the preview: when it was the text block instead (an
  // `aspect-[4/3]` thumbnail plus an unbounded body), a two-line display name
  // pushed the status and pixel-size badges past the bottom edge and the card
  // silently stopped saying whether the image was calibrated. jsdom does no
  // layout, so this asserts the rule rather than the pixels; the pixels are
  // checked by driving the real grid.
  it("lets the preview absorb the slack so the badge row can never be clipped", () => {
    const { container } = render(
      <ImageCard
        image={makeEntry({
          display_name:
            "liver CD3 ROI4 no tag whole block section 12 rescan second pass",
        })}
        onOpen={vi.fn()}
      />
    );

    const card = container.querySelector("article");
    expect(card).not.toBeNull();
    const blocks = Array.from(card!.children).filter(
      (child) => !child.className.includes("absolute")
    );
    const [preview, body] = blocks;

    expect(preview.className).toContain("flex-1");
    // Without `min-h-0` a flex item refuses to shrink below its content.
    expect(preview.className).toContain("min-h-0");
    expect(preview.className).not.toContain("aspect-");
    expect(body.className).toContain("shrink-0");
    expect(body).toContainElement(screen.getByText(/4\.2 nm\/px/));
  });

  it("keeps a long filename to one line rather than pushing the badges out", () => {
    render(
      <ImageCard
        image={makeEntry({
          original_filename:
            "liver_CD3_ROI4_notag_wholeblock_section12_rescan_secondpass_v3.ome.tiff",
        })}
        onOpen={vi.fn()}
      />
    );

    const filename = screen.getByText(
      "liver_CD3_ROI4_notag_wholeblock_section12_rescan_secondpass_v3.ome.tiff"
    );
    expect(filename.className).toContain("truncate");
    // Ellipsised on the card, so the whole string still has to be reachable.
    expect(filename).toHaveAttribute(
      "title",
      "liver_CD3_ROI4_notag_wholeblock_section12_rescan_secondpass_v3.ome.tiff"
    );
  });
});
