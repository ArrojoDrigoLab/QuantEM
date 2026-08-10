import { ApiRequestError } from "@/shared/api/core/http";

/**
 * How long a non-JSON error body may be before it is treated as a document
 * rather than a sentence. Django's debug 403/404 pages are tens of kilobytes;
 * a legitimate plain-text error from this API is one line.
 */
const MAX_PLAIN_TEXT_ERROR_LENGTH = 200;

/** Cheap structural test for "this is a document, not a message". */
function looksLikeMarkup(message: string): boolean {
  const head = message.slice(0, 400).toLowerCase();
  return (
    head.startsWith("<!doctype") ||
    head.startsWith("<html") ||
    head.startsWith("<?xml") ||
    /<\/?(html|head|body|title|div|span|script|style|pre|h1|meta)\b/.test(head)
  );
}

function statusOf(error: unknown): number | null {
  return error instanceof ApiRequestError ? error.status : null;
}

/**
 * Turn a thrown API error into one sentence fit for a `<p>`.
 *
 * `apiRequest` throws with the raw response body as its message, so a Django
 * 403/404 HTML page used to be pasted verbatim into the UI. Anything that is
 * not JSON and does not look like a short sentence is replaced with a generic
 * message plus the status code, and the original body is logged once so it is
 * still debuggable from the console.
 */
export function extractApiErrorMessage(
  error: unknown,
  fallback: string
): string {
  if (!(error instanceof Error) || !error.message) {
    return fallback;
  }
  const message = error.message.trim();
  if (!message) return fallback;
  const status = statusOf(error);

  try {
    const parsed = JSON.parse(message) as
      | { error?: string; detail?: string; non_field_errors?: string[] }
      | null;
    if (parsed) {
      if (typeof parsed.error === "string" && parsed.error.trim()) {
        return parsed.error;
      }
      if (typeof parsed.detail === "string" && parsed.detail.trim()) {
        return parsed.detail;
      }
      if (
        Array.isArray(parsed.non_field_errors) &&
        typeof parsed.non_field_errors[0] === "string" &&
        parsed.non_field_errors[0].trim()
      ) {
        return parsed.non_field_errors[0];
      }
    }
  } catch {
    return sanitisePlainTextError(message, fallback, status);
  }
  return sanitisePlainTextError(message, fallback, status);
}

/**
 * A generic sentence for a body that is not a JSON error payload.
 *
 * Short, markup-free text is passed through -- some endpoints do answer with a
 * bare sentence. Everything else is summarised and logged.
 */
function sanitisePlainTextError(
  message: string,
  fallback: string,
  status: number | null
): string {
  if (message.length <= MAX_PLAIN_TEXT_ERROR_LENGTH && !looksLikeMarkup(message)) {
    return message;
  }
  console.error(
    `[api] Unparseable error body${status === null ? "" : ` (HTTP ${status})`}:`,
    message
  );
  return status === null ? fallback : `${fallback} (HTTP ${status})`;
}

export function isApiNetworkError(error: unknown): boolean {
  if (error instanceof ApiRequestError) {
    return error.isNetworkError;
  }
  if (!(error instanceof Error)) {
    return false;
  }
  const message = error.message.toLowerCase();
  return (
    error.name === "AbortError" ||
    message.includes("failed to fetch") ||
    message.includes("networkerror") ||
    message.includes("network request failed") ||
    message.includes("load failed")
  );
}

/**
 * True when the request failed because the endpoint is not there at all.
 *
 * A dead route and an empty collection must not render identically -- "0 runs"
 * for a 404 tells the user their work vanished.
 */
export function isApiNotFoundError(error: unknown): boolean {
  return statusOf(error) === 404;
}

export function isMicroSamUnavailableError(error: unknown): boolean {
  const message = extractApiErrorMessage(error, "").toLowerCase();
  return (
    message.includes("micro_sam is required but unavailable") ||
    message.includes("micro_sam unavailable") ||
    message.includes("no module named 'micro_sam'") ||
    message.includes('no module named "micro_sam"')
  );
}
