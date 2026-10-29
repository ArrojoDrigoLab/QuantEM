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
const systemSettings = vi.hoisted(() => ({
  get: vi.fn(),
  update: vi.fn(),
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
vi.mock("@/shared/api/systemSettings", () => ({
  getSystemSettings: systemSettings.get,
  updateSystemSettings: systemSettings.update,
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
    systemSettings.get.mockReset();
    systemSettings.update.mockReset();
    systemSettings.get.mockResolvedValue({
      pre_migration_backup_retention_count: 1,
      pre_migration_backup_retention_days: 30,
    });
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
              storage_directory: "/Users/researcher/Library/Application Support/QuantEM",
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
              storage_directory: "/Users/researcher/Library/Application Support/QuantEM",
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

  it("shows the resolved storage directory and keeps database settings below models", async () => {
    renderSettings();

    expect(
      screen.getByText("/Users/researcher/Library/Application Support/QuantEM")
    ).toBeInTheDocument();
    const headings = screen.getAllByRole("heading", { level: 2 });
    expect(headings.indexOf(screen.getByRole("heading", { name: "Models" }))).toBeLessThan(
      headings.indexOf(screen.getByRole("heading", { name: "Database upgrade backups" }))
    );
  });

  it("automatically saves the database-upgrade backup policy with stable per-field feedback", async () => {
    systemSettings.update.mockResolvedValue({
      pre_migration_backup_retention_count: 2,
      pre_migration_backup_retention_days: 45,
      removed_pre_migration_backups: 1,
    });
    const user = userEvent.setup();
    renderSettings();

    const count = await screen.findByRole("spinbutton", { name: "Maximum backups" });
    const days = screen.getByRole("spinbutton", { name: "Delete after (days)" });
    expect(count).toHaveValue(1);
    expect(days).toHaveValue(30);

    await user.clear(count);
    await user.type(count, "2");
    await user.clear(days);
    await user.type(days, "45");

    expect(screen.queryByRole("button", { name: "Save backup settings" })).not.toBeInTheDocument();
    expect(screen.getByRole("status", { name: "Saving maximum backups" })).toBeInTheDocument();
    expect(
      screen.getByRole("status", { name: "Saving backup retention days" })
    ).toBeInTheDocument();

    await waitFor(() =>
      expect(systemSettings.update).toHaveBeenCalledWith({
        pre_migration_backup_retention_count: 2,
        pre_migration_backup_retention_days: 45,
      }),
      { timeout: 2500 }
    );
    expect(
      await screen.findByRole("status", { name: "maximum backups saved" }, { timeout: 2500 })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("status", { name: "backup retention days saved" })
    ).toBeInTheDocument();
    expect(screen.getByText("1 older backup was removed.")).toBeInTheDocument();
  });

  it("clears stale validation feedback when an edit is reverted to the saved value", async () => {
    const user = userEvent.setup();
    renderSettings();

    const count = await screen.findByRole("spinbutton", { name: "Maximum backups" });
    await user.clear(count);
    expect(
      screen.getByRole("status", { name: "maximum backups could not be saved" })
    ).toBeInTheDocument();

    await user.type(count, "1");
    await waitFor(() =>
      expect(
        screen.queryByRole("status", { name: "maximum backups could not be saved" })
      ).not.toBeInTheDocument()
    );
    expect(
      screen.queryByRole("status", { name: "Saving maximum backups" })
    ).not.toBeInTheDocument();
    expect(systemSettings.update).not.toHaveBeenCalled();
  });

  it("persists a revert made while the previous autosave is still in flight", async () => {
    let finishFirstSave!: (value: {
      pre_migration_backup_retention_count: number;
      pre_migration_backup_retention_days: number;
    }) => void;
    const firstSave = new Promise<{
      pre_migration_backup_retention_count: number;
      pre_migration_backup_retention_days: number;
    }>((resolve) => {
      finishFirstSave = resolve;
    });
    systemSettings.update
      .mockImplementationOnce(() => firstSave)
      .mockResolvedValueOnce({
        pre_migration_backup_retention_count: 1,
        pre_migration_backup_retention_days: 30,
      });

    const user = userEvent.setup();
    renderSettings();
    const count = await screen.findByRole("spinbutton", { name: "Maximum backups" });

    await user.clear(count);
    await user.type(count, "2");
    await waitFor(() => expect(systemSettings.update).toHaveBeenCalledTimes(1));

    await user.clear(count);
    await user.type(count, "1");
    finishFirstSave({
      pre_migration_backup_retention_count: 2,
      pre_migration_backup_retention_days: 30,
    });

    await waitFor(
      () =>
        expect(systemSettings.update).toHaveBeenLastCalledWith({
          pre_migration_backup_retention_count: 1,
          pre_migration_backup_retention_days: 30,
        }),
      { timeout: 3000 }
    );
    expect(systemSettings.update).toHaveBeenCalledTimes(2);
    expect(count).toHaveValue(1);
  });
});
