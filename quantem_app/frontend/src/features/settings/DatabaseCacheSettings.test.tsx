import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DatabaseCacheSettings } from "./DatabaseCacheSettings";
import type { SystemSettings } from "@/shared/api/systemSettings";

const api = vi.hoisted(() => ({ update: vi.fn() }));
vi.mock("@/shared/api/systemSettings", () => ({ updateSystemSettings: api.update }));

const inherited: SystemSettings = {
  pre_migration_backup_retention_count: 1,
  pre_migration_backup_retention_days: 30,
  database_cache_mb: null,
  database_cache: {
    available: true, active_mb: 1024, recommended_mb: 512,
    min_mb: 32, max_mb: 4096, total_ram_mb: 16384, restart_required: false,
  },
};

describe("Database cache settings", () => {
  beforeEach(() => { api.update.mockReset(); });

  it("clears validation feedback when reverting to the inherited value", () => {
    render(<DatabaseCacheSettings settings={inherited} />);
    const input = screen.getByRole("spinbutton");
    fireEvent.change(input, { target: { value: "" } });
    expect(screen.getByLabelText("database cache size could not be saved")).toBeInTheDocument();
    fireEvent.change(input, { target: { value: "1024" } });
    expect(screen.queryByLabelText("database cache size could not be saved")).not.toBeInTheDocument();
    expect(api.update).not.toHaveBeenCalled();
  });

  it("persists a revert made while an earlier save is in flight", async () => {
    let finish!: (value: SystemSettings) => void;
    api.update.mockImplementationOnce(() => new Promise<SystemSettings>((resolve) => { finish = resolve; }))
      .mockResolvedValueOnce({ ...inherited, database_cache_mb: 1024 });
    render(<DatabaseCacheSettings settings={inherited} />);
    const input = screen.getByRole("spinbutton");
    fireEvent.change(input, { target: { value: "2048" } });
    await waitFor(() => expect(api.update).toHaveBeenCalledTimes(1));
    fireEvent.change(input, { target: { value: "1024" } });
    finish({ ...inherited, database_cache_mb: 2048,
      database_cache: { ...inherited.database_cache!, restart_required: true } });
    await waitFor(() => expect(api.update).toHaveBeenCalledTimes(2), { timeout: 2500 });
    expect(api.update).toHaveBeenLastCalledWith({ database_cache_mb: 1024 });
    expect(input).toHaveValue(1024);
  }, 4000);

  it("shows the inherited value without saving or recalibrating it", () => {
    render(<DatabaseCacheSettings settings={inherited} />);
    expect(screen.getByRole("spinbutton", { name: "Database cache (MiB)" })).toHaveValue(1024);
    expect(screen.queryByRole("button", { name: "Save cache size" })).not.toBeInTheDocument();
    expect(api.update).not.toHaveBeenCalled();
  });

  it("saves only the cache preference and explains that it takes effect after restart", async () => {
    api.update.mockResolvedValue({
      ...inherited, database_cache_mb: 2048,
      database_cache: { ...inherited.database_cache, restart_required: true },
    });
    render(<DatabaseCacheSettings settings={inherited} />);
    fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "2048" } });
    expect(screen.getByLabelText("Saving database cache size")).toBeInTheDocument();
    await waitFor(
      () => expect(screen.getByLabelText("database cache size saved")).toBeInTheDocument(),
      { timeout: 2000 }
    );
    expect(screen.getByText(/Takes effect after restarting/)).toBeInTheDocument();
    expect(api.update).toHaveBeenCalledExactlyOnceWith({ database_cache_mb: 2048 });
    expect(screen.getByText(/Active cache: 1,024 MiB/)).toBeInTheDocument();
  }, 3000);

  it("validates bounds and autosaves the explicit recommendation choice", async () => {
    api.update.mockResolvedValue({
      ...inherited, database_cache_mb: 512,
      database_cache: { ...inherited.database_cache, restart_required: true },
    });
    render(<DatabaseCacheSettings settings={inherited} />);
    fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "4097" } });
    expect(screen.getByText("Enter a whole number within the allowed range.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Use suggested size" }));
    expect(screen.getByRole("spinbutton")).toHaveValue(512);
    await waitFor(() => expect(api.update).toHaveBeenCalledExactlyOnceWith({ database_cache_mb: 512 }));
  }, 3000);

  it("reports failed saves without claiming the active size changed", async () => {
    api.update.mockRejectedValue(new Error("Could not write settings"));
    render(<DatabaseCacheSettings settings={inherited} />);
    fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "512" } });
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not write settings");
    expect(screen.getByText(/Active cache: 1,024 MiB/)).toBeInTheDocument();
  }, 3000);
});
