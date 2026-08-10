import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ApiRequestError,
  apiRequest,
  apiRequestFormData,
  setApiConfig,
} from "@/shared/api/core/http";

function errorResponse(body: string, status: number): Response {
  return new Response(body, { status });
}

describe("shared/api/core/http", () => {
  beforeEach(() => {
    setApiConfig({ apiBaseUrl: "http://127.0.0.1:9000" });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    setApiConfig({ apiBaseUrl: "http://127.0.0.1:8000" });
  });

  it("carries the status code on a failed request so callers can tell 404 from 500", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(errorResponse("<html>Not Found</html>", 404))
    );

    await expect(apiRequest("/api/analysis/nope/")).rejects.toMatchObject({
      status: 404,
      isNetworkError: false,
    });
  });

  it("keeps the server's own message on a multipart failure", async () => {
    // This used to be lost: the ApiRequestError built from the parsed JSON was
    // thrown inside a `try` whose own `catch` swallowed it and re-threw the raw
    // body instead.
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          errorResponse(
            JSON.stringify({ error: "Pixel size must be greater than zero." }),
            400
          )
        )
    );

    await expect(
      apiRequestFormData("/api/assets/upload/", new FormData())
    ).rejects.toMatchObject({
      message: "Pixel size must be greater than zero.",
      status: 400,
    });
  });

  it("falls back to the raw body when a multipart failure is not JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(errorResponse("Unsupported file type.", 400))
    );

    await expect(
      apiRequestFormData("/api/assets/upload/", new FormData())
    ).rejects.toBeInstanceOf(ApiRequestError);
  });

  it("marks a transport failure as a network error, not a status", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    await expect(apiRequest("/api/assets/")).rejects.toMatchObject({
      isNetworkError: true,
      status: null,
    });
  });
});
