import { getRuntimeConfig } from "@/config";

const runtimeConfig = getRuntimeConfig();
// `window.__APP_CONFIG__.apiBaseUrl` is authoritative: the desktop shell binds the
// backend to a free ephemeral port at launch and injects the resulting origin here.
// Never rewrite it -- a hard-coded port would break every launch.
const apiBase =
  runtimeConfig.apiBaseUrl ||
  (import.meta.env.DEV
    ? "http://127.0.0.1:8000"
    : import.meta.env.VITE_API_BASE_URL) ||
  "";
// No auth token is read or sent: QuantEM is single-user and loopback-only.
// The block that used to seed one from VITE_LOCAL_AUTH_TOKEN referenced a
// variable that no longer exists, and tsc rejected the file.
let API_BASE = apiBase;

export class ApiRequestError extends Error {
  status: number | null;
  isNetworkError: boolean;

  constructor(
    message: string,
    options?: { status?: number | null; isNetworkError?: boolean }
  ) {
    super(message);
    this.name = "ApiRequestError";
    this.status = options?.status ?? null;
    this.isNetworkError = options?.isNetworkError ?? false;
  }
}

export function resolveApiUrl(path: string): string {
  if (path.startsWith("http")) {
    return path;
  }
  if (path.startsWith("/")) {
    return `${API_BASE}${path}`;
  }
  return `${API_BASE}/${path}`;
}

export function setApiConfig(config: {
  apiBaseUrl?: string;
}): void {
  if (config.apiBaseUrl) {
    API_BASE = config.apiBaseUrl;
  }
}

export function getApiAuthHeaders(): Record<string, string> {
  // QuantEM has no authentication: single-user, loopback-only.
  return {};
}

export async function apiRequest<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const url = resolveApiUrl(path);
  let response: Response;
  try {
    response = await fetch(url, {
      headers: {
        "Content-Type": "application/json",
        ...getApiAuthHeaders(),
        ...(options.headers || {}),
      },
      ...options,
    });
  } catch (error) {
    const message =
      error instanceof Error && error.message
        ? error.message
        : "Network request failed.";
    throw new ApiRequestError(message, { isNetworkError: true });
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new ApiRequestError(text || `Request failed with status ${response.status}`, {
      status: response.status,
      isNetworkError: false,
    });
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export async function apiRequestFormData<T>(
  path: string,
  formData: FormData,
  timeoutMs: number = 300000
): Promise<T> {
  const url = path.startsWith("http") ? path : `${API_BASE}${path}`;
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => {
    controller.abort();
  }, timeoutMs);

  try {
    const response = await fetch(url, {
      method: "POST",
      body: formData,
      signal: controller.signal,
      headers: Object.keys(getApiAuthHeaders()).length ? getApiAuthHeaders() : undefined,
    });

    window.clearTimeout(timeoutId);

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      // Parse first, throw after. Throwing inside the `try` used to be caught
      // by its own `catch`, which discarded the parsed `{"error": ...}` message
      // and re-threw the raw body -- so a perfectly good server sentence was
      // replaced by whatever the response happened to contain.
      let message = "";
      try {
        const errorData = JSON.parse(text) as {
          error?: string;
          message?: string;
        } | null;
        message = errorData?.error || errorData?.message || "";
      } catch {
        message = "";
      }
      throw new ApiRequestError(
        message || text || `Request failed with status ${response.status}`,
        {
          status: response.status,
          isNetworkError: false,
        }
      );
    }

    return (await response.json()) as T;
  } catch (error) {
    window.clearTimeout(timeoutId);
    if (error instanceof Error && error.name === "AbortError") {
      throw new ApiRequestError(
        `Upload timeout after ${
          timeoutMs / 1000
        } seconds. The file may be too large or the server is taking too long to process it.`,
        {
          isNetworkError: true,
        }
      );
    }
    if (error instanceof ApiRequestError) {
      throw error;
    }
    if (error instanceof Error) {
      throw new ApiRequestError(error.message, { isNetworkError: true });
    }
    throw error;
  }
}
