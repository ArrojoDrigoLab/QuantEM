import "@testing-library/jest-dom/vitest";
import { afterAll, afterEach, beforeAll } from "vitest";
import { cleanup } from "@testing-library/react";
import { server } from "./msw/server";

/**
 * In-memory `window.localStorage`.
 *
 * This jsdom build exposes no localStorage, so every `window.localStorage.x`
 * call throws a TypeError. The app's own uses are all inside try/catch (library
 * sort controls, the workflow guide's dismissal), so nothing crashed -- but the
 * persistence branches were unreachable from tests and would have hidden any
 * regression in them. A real store makes those paths testable and matches what
 * a browser does.
 */
function installLocalStorage(): void {
  if (typeof window === "undefined") return;
  if (window.localStorage) return;
  const entries = new Map<string, string>();
  const storage: Storage = {
    get length() {
      return entries.size;
    },
    clear: () => entries.clear(),
    getItem: (key) => (entries.has(key) ? (entries.get(key) as string) : null),
    key: (index) => Array.from(entries.keys())[index] ?? null,
    removeItem: (key) => {
      entries.delete(key);
    },
    setItem: (key, value) => {
      entries.set(String(key), String(value));
    },
  };
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: storage,
  });
}

installLocalStorage();

beforeAll(() => {
  server.listen({ onUnhandledRequest: "error" });
});

afterEach(() => {
  cleanup();
  server.resetHandlers();
  window.localStorage?.clear();
});

afterAll(() => {
  server.close();
});
