import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiRequestError } from "@/shared/api/core/http";
import {
  extractApiErrorMessage,
  isApiNetworkError,
  isApiNotFoundError,
} from "@/utils/apiErrors";

const DJANGO_403_PAGE = `<!DOCTYPE html>
<html lang="en">
<head><title>403 Forbidden</title><meta name="robots" content="NONE,NOARCHIVE"></head>
<body><div id="summary"><h1>Forbidden <span>(403)</span></h1>
<pre>CSRF verification failed. Request aborted.</pre></div></body>
</html>`;

afterEach(() => {
  vi.restoreAllMocks();
});

describe("extractApiErrorMessage", () => {
  it("prefers the JSON error field", () => {
    const error = new ApiRequestError(JSON.stringify({ error: "Pixel size must be greater than zero." }), {
      status: 400,
    });
    expect(extractApiErrorMessage(error, "fallback")).toBe(
      "Pixel size must be greater than zero."
    );
  });

  it("falls back to the JSON detail field", () => {
    const error = new ApiRequestError(JSON.stringify({ detail: "Not found." }), {
      status: 404,
    });
    expect(extractApiErrorMessage(error, "fallback")).toBe("Not found.");
  });

  it("replaces an HTML error document with a generic message and the status", () => {
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    const error = new ApiRequestError(DJANGO_403_PAGE, { status: 403 });

    const message = extractApiErrorMessage(error, "The request was refused.");

    expect(message).toBe("The request was refused. (HTTP 403)");
    expect(message).not.toContain("<");
    expect(logged).toHaveBeenCalledOnce();
    expect(String(logged.mock.calls[0][1])).toContain("403 Forbidden");
  });

  it("replaces markup even when it is short enough to pass the length check", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const error = new ApiRequestError("<h1>Not Found</h1>", { status: 404 });
    expect(extractApiErrorMessage(error, "Gone.")).toBe("Gone. (HTTP 404)");
  });

  it("passes through a short plain-text body", () => {
    const error = new ApiRequestError("Only queued jobs can be deleted.", {
      status: 409,
    });
    expect(extractApiErrorMessage(error, "fallback")).toBe(
      "Only queued jobs can be deleted."
    );
  });

  it("uses the bare fallback when there is no status to report", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const error = new Error(DJANGO_403_PAGE);
    expect(extractApiErrorMessage(error, "It failed.")).toBe("It failed.");
  });

  it("returns the fallback for a non-Error value", () => {
    expect(extractApiErrorMessage(null, "fallback")).toBe("fallback");
  });
});

describe("isApiNotFoundError", () => {
  it("is true only for a 404 from the API layer", () => {
    expect(isApiNotFoundError(new ApiRequestError("x", { status: 404 }))).toBe(true);
    expect(isApiNotFoundError(new ApiRequestError("x", { status: 500 }))).toBe(false);
    expect(isApiNotFoundError(new Error("404"))).toBe(false);
  });

  it("does not confuse a network failure for a missing route", () => {
    const offline = new ApiRequestError("Failed to fetch", { isNetworkError: true });
    expect(isApiNotFoundError(offline)).toBe(false);
    expect(isApiNetworkError(offline)).toBe(true);
  });
});
