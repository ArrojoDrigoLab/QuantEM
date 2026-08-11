import { describe, expect, it } from "vitest";
import {
  getStageDisplay,
  isLibraryEntryProcessing,
  isLibraryEntryUnfinished,
} from "@/features/library/components/imageCardUtils";
import type { HomeEntry } from "@/shared/types/images";

function makeEntry(overrides: Partial<HomeEntry> = {}): HomeEntry {
  return {
    id: "asset-1",
    display_name: "Liver 01",
    original_filename: "liver01.tif",
    metadata_summary: "1024x1024",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    preprocess_stage: "NONE",
    preprocess_progress: 0,
    can_open: false,
    ...overrides,
  };
}

describe("getStageDisplay", () => {
  /**
   * The reported defect, exactly.
   *
   * `is_workable` is `openable is not None` (`assets/serializers.py:203`) --
   * true within a second of the upload finishing, long before the pyramid
   * exists. It was tested *first*, so the branch that reports a percentage was
   * unreachable and a 475 MP import read "NGFF pending" for its whole 100 s
   * while `preprocess_progress` sat in the same payload being written once a
   * second.
   */
  it("reports the real percentage on a workable asset that is still encoding", () => {
    const entry = makeEntry({
      is_workable: true,
      ngff_ready: false,
      preprocess_stage: "ENCODING",
      preprocess_progress: 62,
    });

    expect(getStageDisplay(entry)).toBe("Preparing 62%");
  });

  it("never says NGFF, which names a file format the user cannot act on", () => {
    for (const stage of ["NONE", "ENCODING", "SAM", "FEATURES", "DONE"] as const) {
      const text = getStageDisplay(
        makeEntry({ is_workable: true, ngff_ready: false, preprocess_stage: stage })
      );
      expect(text).not.toMatch(/ngff/i);
    }
  });

  it("rounds and clamps rather than printing a fractional or impossible percent", () => {
    expect(
      getStageDisplay(
        makeEntry({ preprocess_stage: "ENCODING", preprocess_progress: 61.6 })
      )
    ).toBe("Preparing 62%");
    expect(
      getStageDisplay(
        makeEntry({ preprocess_stage: "ENCODING", preprocess_progress: 140 })
      )
    ).toBe("Preparing 100%");
    expect(
      getStageDisplay(
        makeEntry({
          preprocess_stage: "ENCODING",
          preprocess_progress: Number.NaN,
        })
      )
    ).toBe("Preparing 0%");
  });

  /**
   * Observed live on a 475 MP import: the pyramid becomes valid at ~50% and
   * `ngff_ready` flips while the stage is still `ENCODING`. The card was then
   * saying "Preparing 50%" in a green badge, one line under a strip saying the
   * image was ready to open -- two answers to the same question.
   */
  it("says Ready as soon as the image can actually be opened", () => {
    expect(
      getStageDisplay(
        makeEntry({
          preprocess_stage: "ENCODING",
          preprocess_progress: 50,
          ngff_ready: true,
        })
      )
    ).toBe("Ready");
  });

  it("separates queued from running, because a stalled queue looks like this", () => {
    expect(
      getStageDisplay(makeEntry({ preprocess_stage: "NONE", ngff_ready: false }))
    ).toBe("Queued");
  });

  it("still reports the terminal states", () => {
    expect(getStageDisplay(makeEntry({ preprocess_stage: "DONE" }))).toBe("Ready");
    expect(getStageDisplay(makeEntry({ preprocess_stage: "SKIPPED" }))).toBe("Ready");
    expect(getStageDisplay(makeEntry({ preprocess_stage: "FAILED" }))).toBe("Failed");
    expect(getStageDisplay(makeEntry({ preprocess_stage: "CANCELLED" }))).toBe(
      "Cancelled"
    );
  });

  /**
   * A FAILED asset can still be `is_workable`, and it used to hit the
   * "NGFF pending" branch first -- so a failed import advertised itself as
   * still in progress, forever.
   */
  it("does not describe a failed import as still in progress", () => {
    expect(
      getStageDisplay(
        makeEntry({
          is_workable: true,
          ngff_ready: false,
          preprocess_stage: "FAILED",
          preprocess_progress: 45,
        })
      )
    ).toBe("Failed");
  });
});

describe("isLibraryEntryUnfinished", () => {
  it("keeps polling a just-created asset whose job has not started", () => {
    // `isLibraryEntryProcessing` says false here, which is why polling used to
    // stop and a fresh import froze on its first badge until a reload.
    const fresh = makeEntry({ preprocess_stage: "NONE", ngff_ready: false });
    expect(isLibraryEntryProcessing(fresh)).toBe(false);
    expect(isLibraryEntryUnfinished(fresh)).toBe(true);
  });

  it("stops at every terminal stage", () => {
    for (const stage of ["DONE", "FAILED", "CANCELLED", "SKIPPED"] as const) {
      expect(isLibraryEntryUnfinished(makeEntry({ preprocess_stage: stage }))).toBe(
        false
      );
    }
  });

  it("stops once a queued asset has a pyramid", () => {
    expect(
      isLibraryEntryUnfinished(
        makeEntry({ preprocess_stage: "NONE", ngff_ready: true })
      )
    ).toBe(false);
  });
});
