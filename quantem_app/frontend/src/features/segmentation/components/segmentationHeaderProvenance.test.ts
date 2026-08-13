import { describe, expect, it } from "vitest";
import {
  NONE_SOURCE_MODEL,
  describeDisplayedObjects,
  resolveSourceModelLabel,
} from "@/features/segmentation/components/segmentationHeaderProvenance";
import type { ImageSegmentation, SourceModelOption } from "@/shared/types";

const OPTIONS: SourceModelOption[] = [
  {
    value: "quantem:mito",
    label: "QuantEM",
    model_family: "quantem",
    is_default: true,
    count: 214,
  },
  { value: "omniem:mito", label: "OmniEM", model_family: "omniem", count: 0 },
  { value: "manual", label: "Manual", model_family: "manual", count: 3 },
];

function makeSegmentation(
  overrides: Partial<ImageSegmentation> = {}
): ImageSegmentation {
  return {
    id: "seg-1",
    segmentation_type: {
      id: "type-1",
      internal_name: "quantem_internal_mito",
      short_name: "Mito",
      long_name: "Mitochondria",
      default_color: null,
      tags: [],
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    },
    status_stage: "CANDIDATES_READY",
    status_progress: 100,
    segment_counts: { CONFIRMED: 12, EXCLUDED: 0, INFERRED: 200, CANDIDATE: 2 },
    source_models: OPTIONS,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

describe("resolveSourceModelLabel", () => {
  it("maps a value to its human label", () => {
    expect(resolveSourceModelLabel("omniem:mito", OPTIONS)).toBe("OmniEM");
  });

  it("falls back to the raw value rather than inventing a name", () => {
    expect(resolveSourceModelLabel("adapted:abc123", OPTIONS)).toBe("adapted:abc123");
  });

  it("does not name a model for the None selection", () => {
    expect(resolveSourceModelLabel(NONE_SOURCE_MODEL, OPTIONS)).toBe("no model");
    expect(resolveSourceModelLabel(null, OPTIONS)).toBe("no model");
  });
});

describe("describeDisplayedObjects", () => {
  it("attributes the objects on screen to the model that produced them", () => {
    const described = describeDisplayedObjects({
      segmentation: makeSegmentation(),
      sourceModelOptions: OPTIONS,
      activeSourceModel: "quantem:mito",
    });

    // The confirmed count leads. The chip used to read "Objects shown: 214
    // from QuantEM" in green, with the confirmed number reachable only by
    // hovering -- and the confirmed number is the one the analysis measures
    // and the one that belongs in a figure legend.
    //
    // Separated, not "12 confirmed **of** 214": this fixture also has 3 manual
    // objects, so some of the 12 may not be QuantEM's and the two numbers are
    // not a fraction of one another. See "of" below for when they are.
    expect(described.summary).toBe("12 confirmed · 214 from QuantEM");
    expect(described.summary.indexOf("12")).toBeLessThan(
      described.summary.indexOf("214")
    );
    expect(described.tone).toBe("good");
    expect(described.detail).toContain("the number the analysis measures");
    expect(described.detail).toContain("At least 202 attributed to QuantEM");
  });

  /**
   * The reported failure: confirm QuantEM's output, flip the selector to
   * OmniEM to compare, and the header read **"41 confirmed of 3 from OmniEM"**
   * -- `confirmed` is segmentation-wide, `count` is the selected model's, and
   * the summary asserted a subset relation between them unconditionally. The
   * tooltip was already right; the visible text was false.
   */
  it("does not claim the confirmed objects are a subset of another model's", () => {
    const described = describeDisplayedObjects({
      segmentation: makeSegmentation({
        segment_counts: { CONFIRMED: 41, EXCLUDED: 0, INFERRED: 0, CANDIDATE: 173 },
      }),
      sourceModelOptions: [
        { value: "quantem:mito", label: "QuantEM", model_family: "quantem", count: 214 },
        { value: "omniem:mito", label: "OmniEM", model_family: "omniem", count: 3 },
        { value: "manual", label: "Manual", model_family: "manual", count: 0 },
      ],
      activeSourceModel: "omniem:mito",
    });

    expect(described.summary).toBe("41 confirmed · 3 from OmniEM");
    expect(described.summary).not.toContain("of 3");
    expect(described.detail).toContain("not a fraction of one another");
    // 41 > 3, so there is no remainder to quote in either direction.
    expect(described.detail).not.toContain("unconfirmed candidate");
  });

  it('says "of" only when this model produced every object here', () => {
    // One model, nothing hand-drawn, no second model: every confirmed object
    // is one of the 214, and the remainder is exact rather than a floor.
    const described = describeDisplayedObjects({
      segmentation: makeSegmentation(),
      sourceModelOptions: [
        { value: "quantem:mito", label: "QuantEM", model_family: "quantem", count: 214 },
        { value: "omniem:mito", label: "OmniEM", model_family: "omniem", count: 0 },
        { value: "manual", label: "Manual", model_family: "manual", count: 0 },
      ],
      activeSourceModel: "quantem:mito",
    });

    expect(described.summary).toBe("12 confirmed of 214 from QuantEM");
    expect(described.detail).toContain("The other 202 attributed to QuantEM");
    expect(described.detail).not.toContain("At least");
  });

  it("does not call it good when nothing has been confirmed", () => {
    // 214 candidates and nothing proofread means the analysis measures nothing,
    // however full of green the screen is.
    const described = describeDisplayedObjects({
      segmentation: makeSegmentation({
        segment_counts: { CONFIRMED: 0, EXCLUDED: 0, INFERRED: 0, CANDIDATE: 214 },
      }),
      sourceModelOptions: OPTIONS,
      activeSourceModel: "quantem:mito",
    });

    expect(described.summary).toBe("0 confirmed · 214 from QuantEM");
    expect(described.tone).toBe("warning");
  });

  it("never quotes a negative unconfirmed remainder", () => {
    // `CONFIRMED` is segmentation-wide and the count is per model, so objects
    // confirmed on another model's output can exceed this model's tally.
    const described = describeDisplayedObjects({
      segmentation: makeSegmentation({
        segment_counts: { CONFIRMED: 300, EXCLUDED: 0, INFERRED: 0, CANDIDATE: 0 },
      }),
      sourceModelOptions: OPTIONS,
      activeSourceModel: "quantem:mito",
    });

    expect(described.summary).toBe("300 confirmed · 214 from QuantEM");
    expect(described.detail).not.toContain("-");
    expect(described.detail).not.toContain("The other");
    expect(described.detail).not.toContain("At least");
  });

  /**
   * The reported failure: a calibrated 5 nm/px image, `quantem:mito`, run to
   * completion, zero candidates, `CANDIDATES_READY`. The chip said "No objects
   * from QuantEM yet" and the tooltip said "Nothing has been run with QuantEM
   * on this image ... Run model to produce some, or choose another
   * model" -- so the reading was *press the button*, when the button had been
   * pressed and would produce the same nothing, and the lever it named was the
   * wrong one. The backend had already diagnosed it; nothing rendered
   * `run_notice`.
   */
  describe("a run that finished and found nothing", () => {
    const NOTHING_FOUND = {
      kind: "no_objects",
      source_model: "quantem:mito",
      message: "This run finished without finding any objects.",
      next_steps: [
        "Check the image's pixel size (5 nm/px). It decides what size the model thinks these organelles are, and a wrong value makes a working model find nothing -- check it before the threshold, because lowering the threshold on a wrongly-scaled run does not bring the objects back.",
        "Lower the detection threshold and run again.",
        "Check that the selected model is trained for this organelle.",
      ],
    };

    function describeEmptyRun(
      overrides: Partial<ImageSegmentation> = {}
    ): ReturnType<typeof describeDisplayedObjects> {
      return describeDisplayedObjects({
        segmentation: makeSegmentation({
          segment_counts: { CONFIRMED: 0, EXCLUDED: 0, INFERRED: 0, CANDIDATE: 0 },
          source_models: [
            { ...OPTIONS[0], count: 0 },
            OPTIONS[1],
            { ...OPTIONS[2], count: 0 },
          ],
          run_notice: NOTHING_FOUND,
          ...overrides,
        }),
        sourceModelOptions: [
          { ...OPTIONS[0], count: 0 },
          OPTIONS[1],
          { ...OPTIONS[2], count: 0 },
        ],
        activeSourceModel: "quantem:mito",
        displayedSourceModel: "quantem:mito",
      });
    }

    it("says the run happened, instead of telling the user to run it", () => {
      const described = describeEmptyRun();

      expect(described.summary).toBe("Ran and found no objects");
      expect(described.summary).not.toMatch(/yet/);
      expect(described.detail).not.toContain("Nothing has been run");
      // "choose another model" is the lever that does not move; the pixel size
      // is the one that does.
      expect(described.detail).not.toContain("choose another model");
    });

    it("carries the server's finding, pixel size first", () => {
      const described = describeEmptyRun();

      expect(described.detail).toContain("This run finished without finding any objects.");
      expect(described.detail).toContain("5 nm/px");
      expect(described.detail.indexOf("pixel size")).toBeLessThan(
        described.detail.indexOf("Lower the detection threshold")
      );
      expect(described.tone).toBe("warning");
    });

    it("does not put the empty-result tag over another model's overlay", () => {
      const described = describeDisplayedObjects({
        segmentation: makeSegmentation({
          segment_counts: { CONFIRMED: 0, EXCLUDED: 0, INFERRED: 0, CANDIDATE: 0 },
          run_notice: NOTHING_FOUND,
        }),
        sourceModelOptions: [{ ...OPTIONS[0], count: 0 }, OPTIONS[1], OPTIONS[2]],
        activeSourceModel: "quantem:mito",
        displayedSourceModel: "omniem:mito",
      });

      expect(described.summary).not.toBe("Ran and found no objects");
      expect(described.detail).toContain("overlay still shows output from OmniEM");
    });

    it("uses the successful run as evidence when an empty result has no overlay", () => {
      const described = describeDisplayedObjects({
        segmentation: makeSegmentation({
          segment_counts: { CONFIRMED: 0, EXCLUDED: 0, INFERRED: 0, CANDIDATE: 0 },
          run_notice: NOTHING_FOUND,
        }),
        sourceModelOptions: [{ ...OPTIONS[0], count: 0 }, OPTIONS[1], OPTIONS[2]],
        activeSourceModel: "quantem:mito",
        displayedSourceModel: null,
      });

      expect(described.summary).toBe("Ran and found no objects");
    });

    it("wins over the unknown-count branch, which would have hidden it", () => {
      // A build that does not report per-model counts would otherwise render
      // "0 confirmed · QuantEM" over a run that found nothing -- true, and
      // silent about the only thing worth saying.
      const described = describeDisplayedObjects({
        segmentation: makeSegmentation({
          segment_counts: { CONFIRMED: 0, EXCLUDED: 0, INFERRED: 0, CANDIDATE: 0 },
          run_notice: NOTHING_FOUND,
        }),
        sourceModelOptions: [
          { value: "quantem:mito", label: "QuantEM", model_family: "quantem" },
        ],
        activeSourceModel: "quantem:mito",
        displayedSourceModel: "quantem:mito",
      });

      expect(described.summary).toBe("Ran and found no objects");
    });

    it("says nothing extra when the server sent no notice", () => {
      const described = describeEmptyRun({ run_notice: null });

      expect(described.summary).toBe("No objects from QuantEM yet");
    });

    it("renders the server's summary verbatim when it sends one", () => {
      const described = describeEmptyRun({
        run_notice: { ...NOTHING_FOUND, summary: "Ran and found no objects" },
      });

      expect(described.summary).toBe("Ran and found no objects");
    });
  });

  /**
   * The second empty run: a re-run over a proofread segmentation. Extraction
   * drops a candidate that lands on a confirmed or excluded object, so the run
   * is *expected* to add nothing -- and the chip that used to render here was
   * the ordinary neutral count, "12 confirmed · 214 from QuantEM", which says
   * nothing about the run that just finished. Worse, composing the chip line
   * client-side would say "Ran and found no objects" over twelve confirmed
   * objects, which is false; `run_notice.summary` is the server's own short
   * line for exactly this reason, and the chip must lead with it.
   */
  describe("a re-run over a proofread segmentation that added nothing", () => {
    const NOTHING_ADDED = {
      kind: "no_new_objects",
      source_model: "quantem:mito",
      summary: "Ran and added no new objects",
      message:
        "This run added no new objects. The 12 object(s) already labelled in this image are unchanged.",
      next_steps: [
        "Nothing changed: the 12 object(s) you have already labelled here are exactly as they were.",
        "A candidate that lands on an object you have already confirmed or excluded is not added again, so a re-run over a proofread image is expected to find nothing new.",
      ],
    };

    it("leads with the server's summary instead of the neutral count", () => {
      const described = describeDisplayedObjects({
        segmentation: makeSegmentation({ run_notice: NOTHING_ADDED }),
        sourceModelOptions: OPTIONS,
        activeSourceModel: "quantem:mito",
        displayedSourceModel: "quantem:mito",
      });

      expect(described.summary).toBe("Ran and added no new objects");
      expect(described.summary).not.toContain("214");
      // Not the empty-run wording: twelve objects are confirmed here, and
      // "found no objects" over them is false.
      expect(described.summary).not.toBe("Ran and found no objects");
      expect(described.detail).toContain(
        "This run added no new objects. The 12 object(s) already labelled in this image are unchanged."
      );
      expect(described.detail).toContain("expected to find nothing new");
      expect(described.tone).toBe("warning");
    });
  });

  /**
   * The reported failure: a user cancelled an ER re-run, the server marked the
   * segmentation FAILED with a specific message, and the header read
   * "190 confirmed of 190 from QuantEM" -- in green, over 190 objects the
   * cancelled run had nothing to do with. `status_error` reached no screen at
   * all; the chip had branches for `run_notice`, `count === null` and
   * `count === 0`, and none for FAILED.
   */
  describe("a run that failed", () => {
    const CANCELLED =
      "Cancelled before it finished, so it produced no result. Nothing was " +
      "saved; start it again when you are ready.";

    function describeFailed(overrides: Partial<ImageSegmentation> = {}) {
      return describeDisplayedObjects({
        segmentation: makeSegmentation({
          status_stage: "FAILED",
          status_error: CANCELLED,
          segment_counts: {
            CONFIRMED: 190,
            EXCLUDED: 0,
            INFERRED: 0,
            CANDIDATE: 0,
          },
          source_models: [{ ...OPTIONS[0], count: 190 }, OPTIONS[1], OPTIONS[2]],
          ...overrides,
        }),
        sourceModelOptions: [{ ...OPTIONS[0], count: 190 }, OPTIONS[1], OPTIONS[2]],
        activeSourceModel: "quantem:mito",
      });
    }

    it("does not report the previous run's objects as this run's result", () => {
      const described = describeFailed();

      // The exact string the user read over a cancelled run.
      expect(described.summary).not.toBe("190 confirmed of 190 from QuantEM");
      expect(described.summary).toContain("Last run failed");
      expect(described.tone).toBe("error");
    });

    it("says the objects on screen predate the failed run", () => {
      const described = describeFailed();

      expect(described.detail).toContain("saved no objects");
      expect(described.detail).toContain(
        "already on this segmentation before that run started"
      );
      expect(described.detail).toContain("Nothing on screen is that run's output.");
    });

    it("carries the server's own message, which names the cause", () => {
      const described = describeFailed();

      expect(described.detail).toContain(CANCELLED);
    });

    it("does not invent a reason when the server recorded none", () => {
      // `status_error` is a TextField, so "" is its default rather than null.
      const described = describeFailed({ status_error: "" });

      expect(described.detail).toContain("The server recorded no reason for it.");
    });

    it("says so even when the failed run left nothing behind", () => {
      const described = describeFailed({
        segment_counts: { CONFIRMED: 0, EXCLUDED: 0, INFERRED: 0, CANDIDATE: 0 },
        source_models: [{ ...OPTIONS[0], count: 0 }, OPTIONS[1], OPTIONS[2]],
      });

      expect(described.summary).toBe("Last run failed · no objects");
      // Not "No objects from QuantEM yet", which reads as "press the button"
      // and is what the `count === 0` branch would have produced.
      expect(described.summary).not.toContain("yet");
      expect(described.tone).toBe("error");
    });

    it("wins over the None selection, which would have hidden it", () => {
      const described = describeDisplayedObjects({
        segmentation: makeSegmentation({
          status_stage: "FAILED",
          status_error: CANCELLED,
        }),
        sourceModelOptions: OPTIONS,
        activeSourceModel: NONE_SOURCE_MODEL,
      });

      expect(described.summary).toContain("Last run failed");
      expect(described.tone).toBe("error");
    });

    it("leaves an unfailed segmentation exactly as it was", () => {
      const described = describeDisplayedObjects({
        segmentation: makeSegmentation({ status_error: "" }),
        sourceModelOptions: OPTIONS,
        activeSourceModel: "quantem:mito",
      });

      expect(described.summary).toBe("12 confirmed · 214 from QuantEM");
      expect(described.tone).toBe("good");
    });
  });

  it("does not treat selecting a model as evidence of a displayed result", () => {
    // Selecting a model is a request, not evidence. It may have run previously,
    // so the tooltip also must not claim that it definitely never ran.
    const described = describeDisplayedObjects({
      segmentation: makeSegmentation(),
      sourceModelOptions: OPTIONS,
      activeSourceModel: "omniem:mito",
    });

    expect(described.summary).toBe("No objects from OmniEM yet");
    expect(described.tone).toBe("warning");
    expect(described.detail).toContain(
      "No successful empty result from OmniEM is displayed"
    );
  });

  it("says manual-only for the None selection", () => {
    const described = describeDisplayedObjects({
      segmentation: makeSegmentation(),
      sourceModelOptions: OPTIONS,
      activeSourceModel: NONE_SOURCE_MODEL,
    });

    expect(described.summary).toBe("Objects shown: manual only (12 confirmed)");
    expect(described.tone).toBe("neutral");
  });

  it("reports an overlay still built from another model", () => {
    const described = describeDisplayedObjects({
      segmentation: makeSegmentation(),
      sourceModelOptions: OPTIONS,
      activeSourceModel: "quantem:mito",
      displayedSourceModel: "omniem:mito",
    });

    expect(described.tone).toBe("warning");
    expect(described.detail).toContain("overlay still shows output from OmniEM");
  });

  it("does not flag a matching overlay", () => {
    const described = describeDisplayedObjects({
      segmentation: makeSegmentation(),
      sourceModelOptions: OPTIONS,
      activeSourceModel: "quantem:mito",
      displayedSourceModel: "quantem:mito",
    });

    expect(described.tone).toBe("good");
    expect(described.detail).not.toContain("overlay still shows");
  });

  it("admits an unknown count rather than guessing zero", () => {
    const described = describeDisplayedObjects({
      segmentation: makeSegmentation(),
      sourceModelOptions: [
        { value: "quantem:mito", label: "QuantEM", model_family: "quantem" },
      ],
      activeSourceModel: "quantem:mito",
    });

    // Still leads with the confirmed count, which is known even when the
    // per-model tally is not.
    expect(described.summary).toBe("12 confirmed · QuantEM");
    expect(described.detail).toContain("did not report a per-model object count");
  });
});
