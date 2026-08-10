/**
 * The words a card says about an install it did not start.
 *
 * These strings are the whole fix for uat13 #1's first impression: a user who
 * chose packs in the installer opens Models mid-download and must read
 * "Installing — …" rather than "not installed" with a Download button.
 */

import { describe, expect, it } from "vitest";
import {
  activeInstallPercent,
  describeActiveInstall,
} from "@/features/models/activeInstall";
import type { ModelPackActiveInstall } from "@/shared/types/finetune";

function install(
  overrides: Partial<ModelPackActiveInstall> = {}
): ModelPackActiveInstall {
  return {
    job_id: 7,
    status: "RUNNING",
    progress_current_bytes: null,
    progress_total_bytes: null,
    ...overrides,
  };
}

describe("describeActiveInstall", () => {
  it("says Queued while nothing has moved", () => {
    expect(describeActiveInstall(install({ status: "QUEUED" }))).toBe("Queued");
  });

  it("says Installing without inventing bytes for a running job that reported none", () => {
    expect(describeActiveInstall(install({ status: "RUNNING" }))).toBe(
      "Installing…"
    );
  });

  it("reports bytes with the em-dash convention", () => {
    expect(
      describeActiveInstall(
        install({
          progress_current_bytes: 214 * 1024 * 1024,
          progress_total_bytes: 1243 * 1024 * 1024,
        })
      )
    ).toBe("Installing — 214.0 MB of 1.2 GB");
  });

  it("shows what arrived when the total is unknown", () => {
    expect(
      describeActiveInstall(install({ progress_current_bytes: 214 * 1024 * 1024 }))
    ).toBe("Installing — 214.0 MB");
  });
});

describe("activeInstallPercent", () => {
  it("is null without both byte counts", () => {
    expect(activeInstallPercent(install())).toBeNull();
    expect(
      activeInstallPercent(install({ progress_current_bytes: 100 }))
    ).toBeNull();
    expect(
      activeInstallPercent(
        install({ progress_current_bytes: 100, progress_total_bytes: 0 })
      )
    ).toBeNull();
  });

  it("is the byte fraction, clamped to 0–100", () => {
    expect(
      activeInstallPercent(
        install({ progress_current_bytes: 25, progress_total_bytes: 100 })
      )
    ).toBe(25);
    expect(
      activeInstallPercent(
        install({ progress_current_bytes: 200, progress_total_bytes: 100 })
      )
    ).toBe(100);
  });
});
