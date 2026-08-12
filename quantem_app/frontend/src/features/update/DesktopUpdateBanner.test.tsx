import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DesktopUpdateBanner } from "@/features/update/DesktopUpdateBanner";
import { RestartGuardProvider } from "@/features/update/restartGuard";
import { setApiConfig } from "@/shared/api/core/http";

const desktop = vi.hoisted(() => ({ enabled: true }));
const updater = vi.hoisted(() => ({ check: vi.fn(), relaunch: vi.fn() }));

vi.mock("@/features/update/desktopRuntime", () => ({
  isDesktopTauriBuild: () => desktop.enabled,
}));
vi.mock("@/features/update/desktopUpdater", () => ({
  getDesktopUpdater: async () => updater,
}));

function response(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("DesktopUpdateBanner", () => {
  beforeEach(() => {
    desktop.enabled = true;
    updater.check.mockReset();
    updater.relaunch.mockReset();
    setApiConfig({ apiBaseUrl: "http://127.0.0.1:9000" });
    window.localStorage.clear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    setApiConfig({ apiBaseUrl: "http://127.0.0.1:8000" });
  });

  it("does not check for updates outside the installed desktop build", async () => {
    desktop.enabled = false;
    render(
      <RestartGuardProvider>
        <DesktopUpdateBanner />
      </RestartGuardProvider>
    );

    await waitFor(() => expect(updater.check).not.toHaveBeenCalled());
    expect(screen.queryByText(/is available/)).not.toBeInTheDocument();
  });

  it("downloads an available update then waits for active work to finish", async () => {
    const download = vi.fn(async (callback: (event: { event: "Started" | "Progress"; data: { contentLength?: number; chunkLength?: number } }) => void) => {
      callback({ event: "Started", data: { contentLength: 100 } });
      callback({ event: "Progress", data: { chunkLength: 25 } });
    });
    updater.check.mockResolvedValue({ version: "0.2.0", body: "Safer updates", download, install: vi.fn() });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        response({
          running: [{ id: "running" }],
          queues: [{ queue_name: "default", display_name: "Default", pending: [{ id: "pending" }] }],
          failed: [],
          completed: [],
          worker: { scheduler_in_process: true },
          generated_at: "2026-08-11T00:00:00Z",
        })
      )
    );
    const user = userEvent.setup();
    render(
      <RestartGuardProvider>
        <DesktopUpdateBanner />
      </RestartGuardProvider>
    );

    await screen.findByText("QuantEM 0.2.0 is available.");
    await user.click(screen.getByRole("button", { name: "Update" }));

    await screen.findByText(/Waiting for 2 active tasks to finish/);
    expect(download).toHaveBeenCalledOnce();
  });
});
