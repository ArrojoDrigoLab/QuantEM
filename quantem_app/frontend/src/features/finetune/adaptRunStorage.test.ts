import { beforeEach, describe, expect, it } from "vitest";
import {
  forgetAdaptRun,
  loadAdaptRun,
  rememberAdaptRun,
} from "@/features/finetune/adaptRunStorage";

const KEY = "quantem-adapt-runs-v1";

describe("adaptRunStorage", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("round-trips the ids needed to reattach to a run", () => {
    rememberAdaptRun("seg-1", { adapterId: "ad-1", jobId: "job-1" });

    expect(loadAdaptRun("seg-1")).toEqual({ adapterId: "ad-1", jobId: "job-1" });
  });

  it("keeps runs for different segmentations apart", () => {
    rememberAdaptRun("seg-1", { adapterId: "ad-1", jobId: "job-1" });
    rememberAdaptRun("seg-2", { adapterId: "ad-2", jobId: null });

    expect(loadAdaptRun("seg-1")?.adapterId).toBe("ad-1");
    expect(loadAdaptRun("seg-2")?.adapterId).toBe("ad-2");
    expect(loadAdaptRun("seg-3")).toBeNull();
  });

  it("forgets one run without touching the others", () => {
    rememberAdaptRun("seg-1", { adapterId: "ad-1", jobId: null });
    rememberAdaptRun("seg-2", { adapterId: "ad-2", jobId: null });

    forgetAdaptRun("seg-1");

    expect(loadAdaptRun("seg-1")).toBeNull();
    expect(loadAdaptRun("seg-2")?.adapterId).toBe("ad-2");
  });

  it("survives a corrupt store rather than taking the wizard down", () => {
    // The wizard renders before it reads this; a parse error here must not be
    // the reason someone cannot start a run.
    window.localStorage.setItem(KEY, "{not json");

    expect(loadAdaptRun("seg-1")).toBeNull();
    expect(() =>
      rememberAdaptRun("seg-1", { adapterId: "ad-1", jobId: null })
    ).not.toThrow();
    expect(loadAdaptRun("seg-1")?.adapterId).toBe("ad-1");
  });

  it("drops entries that are not a run handle", () => {
    window.localStorage.setItem(
      KEY,
      JSON.stringify({ "seg-1": { jobId: "job-1" }, "seg-2": 7 })
    );

    expect(loadAdaptRun("seg-1")).toBeNull();
    expect(loadAdaptRun("seg-2")).toBeNull();
  });

  it("treats a missing job id as a settled run rather than dropping the adapter", () => {
    window.localStorage.setItem(
      KEY,
      JSON.stringify({ "seg-1": { adapterId: "ad-1" } })
    );

    expect(loadAdaptRun("seg-1")).toEqual({ adapterId: "ad-1", jobId: null });
  });
});
