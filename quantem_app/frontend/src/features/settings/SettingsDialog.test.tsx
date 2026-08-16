import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SettingsDialog } from "@/features/settings/SettingsDialog";
import { DesktopUpdateProvider } from "@/features/update/DesktopUpdateProvider";
import { RestartGuardProvider } from "@/features/update/restartGuard";

const desktop = vi.hoisted(() => ({ enabled: true }));
const updater = vi.hoisted(() => ({ check: vi.fn(), relaunch: vi.fn() }));
const jobs = vi.hoisted(() => ({
  getStatus: vi.fn(),
  acquireLock: vi.fn(),
  releaseLock: vi.fn(),
}));

vi.mock("@/features/update/desktopRuntime", () => ({
  isDesktopTauriBuild: () => desktop.enabled,
}));
vi.mock("@/features/update/desktopUpdater", () => ({
  getDesktopUpdater: async () => updater,
}));
vi.mock("@/shared/api/jobs", () => ({
  getJobQueueStatus: jobs.getStatus,
  acquireUpdateApplyLock: jobs.acquireLock,
  releaseUpdateApplyLock: jobs.releaseLock,
}));
vi.mock("@/features/models/ModelManagementSection", () => ({
  ModelManagementSection: () => <div>Model management</div>,
}));

describe("SettingsDialog application upgrades", () => {
  beforeEach(() => {
    desktop.enabled = true;
    updater.check.mockReset();
    updater.relaunch.mockReset();
    jobs.getStatus.mockReset();
    jobs.acquireLock.mockReset();
    jobs.releaseLock.mockReset();
    window.localStorage.clear();
    // Suppress the provider's automatic daily check so this test exercises the
    // Settings action specifically.
    window.localStorage.setItem("quantem.desktop-update.checked-at", String(Date.now()));
    jobs.getStatus.mockResolvedValue({ running: [], queues: [] });
    jobs.acquireLock.mockResolvedValue({ ready: true, reason: null, open_jobs: 0 });
    jobs.releaseLock.mockResolvedValue(undefined);
  });

  function renderSettings() {
    return render(
      <RestartGuardProvider>
        <DesktopUpdateProvider>
          <SettingsDialog
            isOpen
            status={{
              app_version: "0.1.2",
              cuda_available: false,
              supported_upload_formats: [".tif"],
            }}
            onClose={() => undefined}
          />
        </DesktopUpdateProvider>
      </RestartGuardProvider>
    );
  }

  it("checks manually, shows the available version, and starts the upgrade", async () => {
    const download = vi.fn(async () => undefined);
    const install = vi.fn(async () => undefined);
    updater.check.mockResolvedValue({
      version: "0.1.3",
      body: "Upgrade notes",
      download,
      install,
    });
    const user = userEvent.setup();
    renderSettings();

    await user.click(screen.getByRole("button", { name: "Check for upgrades" }));

    expect(await screen.findByText("New version v0.1.3 is available.")).toBeInTheDocument();
    expect(updater.check).toHaveBeenCalledOnce();

    await user.click(screen.getByRole("button", { name: "Upgrade now" }));
    await waitFor(() => expect(download).toHaveBeenCalledOnce());
    await waitFor(() => expect(install).toHaveBeenCalledOnce());
  });

  it("reports when the installed version is current and allows another check", async () => {
    updater.check.mockResolvedValue(null);
    const user = userEvent.setup();
    renderSettings();

    await user.click(screen.getByRole("button", { name: "Check for upgrades" }));

    expect(await screen.findByText("Latest version")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Check for upgrades" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Check again" }));
    await waitFor(() => expect(updater.check).toHaveBeenCalledTimes(2));
  });

  it("describes CPU and CUDA acceleration using the requested labels", () => {
    const { rerender } = renderSettings();

    expect(screen.getByText("GPU Acceleration")).toBeInTheDocument();
    expect(screen.getByText("CPU-only")).toBeInTheDocument();

    rerender(
      <RestartGuardProvider>
        <DesktopUpdateProvider>
          <SettingsDialog
            isOpen
            status={{
              app_version: "0.1.2",
              cuda_available: true,
              supported_upload_formats: [".tif"],
            }}
            onClose={() => undefined}
          />
        </DesktopUpdateProvider>
      </RestartGuardProvider>
    );

    expect(screen.getByText("CUDA-enabled")).toBeInTheDocument();
  });
});
