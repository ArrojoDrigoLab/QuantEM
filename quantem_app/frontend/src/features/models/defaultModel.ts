import type { ModelCatalogue } from "@/shared/types/finetune";
import { packRunnability } from "./runnable";

export function preferredPackId(
  catalogue: ModelCatalogue | null,
  organelleId: string
): string | null {
  const candidates = ["quantem", "omniem"].map((family) => {
    const id = `${family}:${organelleId}`;
    return { id, pack: catalogue?.packs?.find((pack) => pack.id === id) ?? null };
  });
  const downloaded = candidates.filter((choice) => choice.pack?.installed === true);
  const usableDownloaded = downloaded.filter(
    (choice) => choice.pack && packRunnability(choice.pack).state !== "blocked"
  );
  if (downloaded.length === 1) return usableDownloaded[0]?.id ?? null;
  const omniem = candidates.find((choice) => choice.id === `omniem:${organelleId}`);
  const blocked = Boolean(
    omniem?.pack?.installed === true &&
      packRunnability(omniem.pack).state === "blocked"
  );
  if (!blocked) return omniem?.id ?? null;
  return usableDownloaded[0]?.id ?? null;
}
