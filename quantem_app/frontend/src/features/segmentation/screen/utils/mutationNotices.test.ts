import { describe, expect, it } from "vitest";
import {
  collectMutationNotices,
  mutationNoticeMessage,
} from "@/features/segmentation/screen/utils/mutationNotices";

const STORED_ONE = {
  created: 1,
  updated: 0,
  deleted: 0,
  outlines: null,
  measurement: null,
};

describe("collectMutationNotices", () => {
  it("says nothing about one outline in, one object out, measured", () => {
    expect(
      collectMutationNotices(STORED_ONE, { nothingStoredMessage: "unused" })
    ).toEqual([]);
  });

  it("repeats an outline that separated", () => {
    const notices = collectMutationNotices({
      created: 2,
      updated: 0,
      deleted: 0,
      outlines: {
        separated: [{ index: 0, areas: 2, kept: 2 }],
        dropped: [],
        detail: "segments[0] crosses itself: it encloses 2 separate areas.",
      },
    });
    expect(notices).toEqual([
      "segments[0] crosses itself: it encloses 2 separate areas.",
    ]);
  });

  /**
   * The finding this module exists for: `confirm-batch` answers 200 with
   * `created: 0` for an outline narrower than a pixel, and every call site but
   * one read that as a plain success.
   */
  it("repeats an outline that was refused for being narrower than a pixel", () => {
    const notices = collectMutationNotices({
      created: 0,
      updated: 0,
      deleted: 0,
      outlines: {
        separated: [],
        dropped: [{ index: 0, areas: 1, kept: 0 }],
        detail: "segments[0] was not stored: the outline spans 1 pixel or less.",
      },
    });
    expect(notices).toEqual([
      "segments[0] was not stored: the outline spans 1 pixel or less.",
    ]);
  });

  it("repeats objects whose geometry was committed but not measured", () => {
    const notices = collectMutationNotices({
      created: 1,
      updated: 0,
      deleted: 0,
      measurement: {
        measured: 0,
        unmeasured_ids: ["a"],
        detail: "The image could not be opened, so these were not measured.",
      },
    });
    expect(notices).toEqual([
      "The image could not be opened, so these were not measured.",
    ]);
  });

  it("reports both when a batch separated and then failed to measure", () => {
    const notices = collectMutationNotices({
      created: 2,
      updated: 0,
      deleted: 0,
      outlines: { separated: [{ index: 0, areas: 2, kept: 2 }], detail: "It split." },
      measurement: { measured: 0, unmeasured_ids: ["a"], detail: "Not measured." },
    });
    expect(notices).toEqual(["It split.", "Not measured."]);
  });

  /**
   * The backstop, and the only signal the merge path has: `merge_overlaps`
   * unions each outline with whatever it overlaps before the size filter runs,
   * so the server deliberately makes no per-outline claim -- but 0/0/0 is still
   * a fact, and it means the store is unchanged.
   */
  it("says nothing changed when the counts are all zero and the server gave no reason", () => {
    expect(
      collectMutationNotices(
        { created: 0, updated: 0, deleted: 0 },
        { nothingStoredMessage: "Nothing was stored." }
      )
    ).toEqual(["Nothing was stored."]);
  });

  it("prefers the server's own sentence over the fallback", () => {
    expect(
      collectMutationNotices(
        {
          created: 0,
          updated: 0,
          deleted: 0,
          outlines: { separated: [], dropped: [], detail: "The server's reason." },
        },
        { nothingStoredMessage: "Nothing was stored." }
      )
    ).toEqual(["The server's reason."]);
  });

  it("stays silent on a no-op when the caller gave no fallback", () => {
    expect(collectMutationNotices({ created: 0, updated: 0, deleted: 0 })).toEqual([]);
  });

  it("treats a null response as nothing to say", () => {
    // `submitConfirmedGeometriesOptimistically` returns null when the
    // segmentation changed under the request; there is no outcome to report.
    expect(
      collectMutationNotices(null, { nothingStoredMessage: "Nothing was stored." })
    ).toEqual([]);
  });
});

describe("mutationNoticeMessage", () => {
  it("is null when there is nothing to say, so no empty toast can be shown", () => {
    expect(mutationNoticeMessage(STORED_ONE)).toBeNull();
  });

  it("joins several notices into one sentence", () => {
    expect(
      mutationNoticeMessage({
        created: 2,
        updated: 0,
        deleted: 0,
        outlines: { separated: [], detail: "It split." },
        measurement: { measured: 0, unmeasured_ids: [], detail: "Not measured." },
      })
    ).toBe("It split. Not measured.");
  });
});
