import { apiRequest } from "@/shared/api/core/http";

export interface SystemSettings {
  pre_migration_backup_retention_count: number;
  pre_migration_backup_retention_days: number;
  removed_pre_migration_backups?: number;
  database_cache_mb?: number | null;
  database_cache?: {
    available: boolean;
    active_mb: number | null;
    recommended_mb: number;
    min_mb: number;
    max_mb: number;
    total_ram_mb: number | null;
    restart_required: boolean;
  };
}

export function getSystemSettings(): Promise<SystemSettings> {
  return apiRequest<SystemSettings>("/api/system/settings/");
}

export function updateSystemSettings(
  settings: Partial<Pick<
    SystemSettings,
    | "pre_migration_backup_retention_count"
    | "pre_migration_backup_retention_days"
    | "database_cache_mb"
  >>
): Promise<SystemSettings> {
  return apiRequest<SystemSettings>("/api/system/settings/", {
    method: "PUT",
    body: JSON.stringify(settings),
  });
}
