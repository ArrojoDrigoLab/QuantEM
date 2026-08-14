import { afterEach, describe, expect, it } from "vitest";

import { isDesktopTauriBuild } from "@/features/update/desktopRuntime";

interface TauriTestGlobal {
  isTauri?: boolean;
  __TAURI_INTERNALS__?: unknown;
}

const tauriGlobal = globalThis as typeof globalThis & TauriTestGlobal;

afterEach(() => {
  delete tauriGlobal.isTauri;
  delete tauriGlobal.__TAURI_INTERNALS__;
});

describe("isDesktopTauriBuild", () => {
  it("recognizes the Tauri 2 runtime signal", () => {
    tauriGlobal.isTauri = true;

    expect(isDesktopTauriBuild()).toBe(true);
  });

  it("keeps compatibility with shells exposing Tauri internals", () => {
    tauriGlobal.__TAURI_INTERNALS__ = {};

    expect(isDesktopTauriBuild()).toBe(true);
  });

  it("does not identify a regular browser as the desktop app", () => {
    expect(isDesktopTauriBuild()).toBe(false);
  });
});
