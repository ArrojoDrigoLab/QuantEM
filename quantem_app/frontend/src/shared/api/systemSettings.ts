import { apiRequest } from "@/shared/api/core/http";

export interface SystemSettings {
  pre_migration_backup_retention_count: number;
  pre_migration_backup_retention_days: number;
  removed_pre_migration_backups?: number;
}

export function getSystemSettings(): Promise<SystemSettings> {
  return apiRequest<SystemSettings>("/api/system/settings/");
}

export function updateSystemSettings(
  settings: Pick<
    SystemSettings,
    | "pre_migration_backup_retention_count"
    | "pre_migration_backup_retention_days"
  >
): Promise<SystemSettings> {
  return apiRequest<SystemSettings>("/api/system/settings/", {
    method: "PUT",
    body: JSON.stringify(settings),
  });
}
