import { afterEach, describe, expect, it, vi } from "vitest";

import { setApiConfig } from "@/shared/api/core/http";

import { decodeProbImage } from "./overlayCanvas";

class RejectingImage {
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  crossOrigin: string | null = null;

  set src(value: string) {
    RejectingImage.lastSource = value;
    RejectingImage.lastCrossOrigin = this.crossOrigin;
    this.onerror?.();
  }

  static lastSource = "";
  static lastCrossOrigin: string | null = null;
}

describe("decodeProbImage", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    RejectingImage.lastSource = "";
    RejectingImage.lastCrossOrigin = null;
    setApiConfig({ apiBaseUrl: "http://127.0.0.1:8000" });
  });

  it("resolves backend-relative preview paths against the configured API", async () => {
    setApiConfig({ apiBaseUrl: "http://127.0.0.1:45174" });
    vi.stubGlobal("Image", RejectingImage);

    await expect(decodeProbImage("/api/preview.png")).rejects.toThrow(
      "Failed to decode probability image"
    );

    expect(RejectingImage.lastSource).toBe("http://127.0.0.1:45174/api/preview.png");
    expect(RejectingImage.lastCrossOrigin).toBe("anonymous");
  });

  it("leaves inline probability images unchanged", async () => {
    const dataUrl = "data:image/png;base64,AA==";
    vi.stubGlobal("Image", RejectingImage);

    await expect(decodeProbImage(dataUrl)).rejects.toThrow(
      "Failed to decode probability image"
    );

    expect(RejectingImage.lastSource).toBe(dataUrl);
    expect(RejectingImage.lastCrossOrigin).toBeNull();
  });
});
