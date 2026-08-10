/**
 * The models screen: what QuantEM ships, what is installed, what can run here.
 *
 * This screen did not exist. `GET /api/models/` was called from exactly one
 * place — step 1 of the fine-tuning wizard — so a user who never opened Adapt
 * had no surface for models at all: no way to see that nothing was installed,
 * no way to install anything, and no explanation when a segmentation run died.
 * That is also why the clean-install failure was so opaque: the one fact that
 * explained it (`runnable: false`, "Not installed yet.") was on an endpoint
 * nothing rendered.
 *
 * The screen is deliberately blunt about the install story, because the backend
 * is: there is no remote registry and nothing here fetches anything, so the
 * only route offered is a copy already on this machine. What it must no longer
 * say is that the copy cannot be verified -- that was true of the bare
 * `checkpoint_index.json`, and stopped being true when release bundles landed.
 * A bundle's `MANIFEST.json` lists every file with a SHA-256 and
 * `install_pack_from_bundle` re-hashes each one before the pack counts as
 * installed.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  applyAdapter,
  getModelCatalogue,
  installModelPack,
} from "@/shared/api/finetune";
import { ApiRequestError } from "@/shared/api/core/http";
import { cancelJob, deleteJob } from "@/shared/api/jobs";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import { useJobProgress } from "@/shared/hooks/useJobProgress";
import { Badge, Button, PageState, Panel } from "@/shared/ui/design";
import { formatBytes, formatNumber, formatTimestamp } from "@/shared/ui/format";
import { cx } from "@/shared/ui/cx";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import {
  activeInstallPercent,
  describeActiveInstall,
} from "@/features/models/activeInstall";
import {
  describeDevice,
  noPackIsRunnable,
  packRunnability,
} from "@/features/models/runnable";
import {
  RunnabilityBadge,
  RunnabilityReason,
} from "@/features/models/components/RunnabilityBadge";
import {
  useLastFailedInstalls,
  type LastFailedInstall,
} from "@/features/models/useLastFailedInstalls";
import { SplitModeBadge } from "@/features/finetune/components/HonestScore";
import { FAMILY_LABELS, ORGANELLE_LABELS } from "@/features/finetune/models";
import type {
  AdaptedModelEntry,
  ModelCatalogue,
  ModelPack,
} from "@/shared/types/finetune";

export function ModelsScreen() {
  const {
    data: catalogue,
    error,
    loading,
    refetch,
  } = useApiQuery<ModelCatalogue | null>(() => getModelCatalogue(), []);

  const device = describeDevice(catalogue ?? null);
  const nothingRuns = noPackIsRunnable(catalogue ?? null);

  // A failed download must survive navigation. The card's own state dies with
  // the unmount, and `GET /api/models/` says nothing about past installs, so
  // the history is read from the jobs API and refreshed whenever the
  // catalogue is.
  const { failures: failedInstalls, refresh: refreshFailedInstalls } =
    useLastFailedInstalls();
  const refreshAll = useCallback(() => {
    void refetch();
    refreshFailedInstalls();
  }, [refetch, refreshFailedInstalls]);

  const byFamily = useMemo(() => {
    const groups = new Map<string, ModelPack[]>();
    for (const pack of catalogue?.packs ?? []) {
      const list = groups.get(pack.family) ?? [];
      list.push(pack);
      groups.set(pack.family, list);
    }
    return [...groups.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [catalogue]);

  return (
    <div className="min-h-screen px-5 py-5 text-slate-900 lg:px-8">
      <div className="mx-auto flex max-w-[1100px] flex-col gap-5">
        <header className="flex flex-wrap items-center justify-between gap-4 rounded-lg border border-slate-200 bg-white px-4 py-3 shadow-sm">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-cyan-700">
              QuantEM
            </p>
            <h1 className="m-0 text-2xl font-semibold tracking-normal text-slate-950">
              Models
            </h1>
            <p className="m-0 mt-1 text-sm text-slate-500">
              The released segmentation packs, what is installed, and what this
              machine can actually run.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={refreshAll}>Refresh</Button>
            <Link
              className="inline-flex h-10 items-center rounded-md border border-slate-300 bg-white px-4 text-sm font-medium text-slate-800 shadow-sm hover:bg-slate-50"
              to="/"
            >
              Back to library
            </Link>
          </div>
        </header>

        {loading && !catalogue ? <PageState title="Loading models…" /> : null}

        {error ? (
          <PageState
            tone="error"
            title="The model catalogue did not answer"
            detail={extractApiErrorMessage(error, "no response")}
          />
        ) : null}

        {catalogue ? (
          <Panel className="p-4">
            <div className="flex flex-wrap items-baseline justify-between gap-3">
              <h2 className="m-0 text-base font-semibold text-slate-950">
                This machine
              </h2>
              {device ? (
                <p className="m-0 text-sm text-slate-600">
                  Inference device: <span className="font-medium">{device}</span>
                </p>
              ) : null}
            </div>
            {nothingRuns ? (
              // The clean-install case, said once and plainly. Without this the
              // first sign of trouble is a segmentation banner stuck at 5%.
              <div className="mt-3 rounded-md border border-amber-200 bg-amber-50 p-3">
                <p className="m-0 text-sm font-semibold text-amber-900">
                  No model can run on this machine yet.
                </p>
                <p className="m-0 mt-1 text-sm text-amber-900">
                  Every pack below is unavailable for the reason it gives.
                  Creating a segmentation will still queue a run, and that run
                  will fail — install a pack first, or annotate by hand.
                </p>
              </div>
            ) : null}
            {/* Two routes to an installed pack, and both verify before the
                pack counts as installed. The previous copy ("QuantEM does not
                download weights") described the build before the registry
                download existed; a build whose backend still refuses the
                download says exactly why on the Download button's own error,
                verbatim from the server, rather than being contradicted here. */}
            <p className="m-0 mt-3 text-xs text-slate-500">
              Weights are not bundled with the application. Download fetches a
              pack from the QuantEM model registry and verifies every file
              against its published digest before the pack counts as installed;
              if this build cannot download, the button reports the server's
              reason. Installing from a local folder does the same verification
              against the release bundle's <code>MANIFEST.json</code>.
            </p>
          </Panel>
        ) : null}

        {byFamily.map(([family, packs]) => (
          <Panel key={family} className="p-4">
            <h2 className="m-0 text-base font-semibold text-slate-950">
              {FAMILY_LABELS[family] ?? family}
            </h2>
            <div className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-2">
              {packs.map((pack) => (
                <PackCard
                  key={pack.id}
                  pack={pack}
                  lastFailedInstall={failedInstalls.get(pack.id) ?? null}
                  onInstalled={refreshAll}
                />
              ))}
            </div>
          </Panel>
        ))}

        {catalogue && catalogue.adapted.length > 0 ? (
          <Panel className="p-4">
            <h2 className="m-0 text-base font-semibold text-slate-950">
              Your adapted models
            </h2>
            <p className="m-0 mt-1 text-sm text-slate-600">
              Produced by the guided fine-tuning wizard. Each one runs through
              its base pack, so it needs that pack to be runnable too. Applying
              one makes it the model used for subsequent runs on the
              segmentation it was fitted on; it changes nothing already
              produced.
            </p>
            <ul className="m-0 mt-3 flex list-none flex-col gap-2 p-0">
              {catalogue.adapted.map((entry) => (
                <AdaptedCard
                  key={entry.id}
                  entry={entry}
                  onApplied={() => void refetch()}
                />
              ))}
            </ul>
          </Panel>
        ) : null}
      </div>
    </div>
  );
}

/**
 * One adapted model, with the one action it was missing.
 *
 * This list was read-only. Combined with a wizard whose state did not survive a
 * reload, that meant a completed head training could be seen and never used:
 * the only Apply button in the product was on step 6 of a wizard you could no
 * longer reach. Apply is idempotent and reversible by applying another, so it
 * needs no confirmation -- but it does have to report what happened, because
 * nothing else on this screen changes except the badge.
 */
function AdaptedCard({
  entry,
  onApplied,
}: {
  entry: AdaptedModelEntry;
  onApplied: () => void;
}) {
  const [applying, setApplying] = useState(false);
  const [applyError, setApplyError] = useState<string | null>(null);

  const handleApply = useCallback(async () => {
    setApplying(true);
    setApplyError(null);
    try {
      // `GET/POST /api/adapters/<id>/` take the bare uuid; the catalogue
      // prefixes it with "adapted:" to keep it distinct from a pack id.
      await applyAdapter(entry.id.replace(/^adapted:/, ""));
      onApplied();
    } catch (err) {
      setApplyError(
        extractApiErrorMessage(err, "This adapter could not be applied.")
      );
    } finally {
      setApplying(false);
    }
  }, [entry.id, onApplied]);

  return (
    <li className="flex flex-col gap-2 rounded-md border border-slate-200 px-3 py-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0">
          <p className="m-0 truncate text-sm font-medium text-slate-900">
            {entry.name || entry.id}
          </p>
          <p className="m-0 text-xs text-slate-500">
            from {entry.base} · {formatTimestamp(entry.created_at)}
            {entry.calibrated_threshold !== null
              ? ` · threshold ${formatNumber(entry.calibrated_threshold, 2)}`
              : ""}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {/* Honesty rule 1: the score never travels without its split mode. */}
          {entry.heldout_dice !== null ? (
            <span className="text-sm tabular-nums text-slate-900">
              Dice {formatNumber(entry.heldout_dice, 3)}
            </span>
          ) : (
            <span className="text-sm text-slate-500">no held-out score</span>
          )}
          <SplitModeBadge mode={entry.split_mode} />
          {entry.applied_at ? <Badge tone="info">applied</Badge> : null}
          <Button
            size="sm"
            variant={entry.applied_at ? "secondary" : "primary"}
            disabled={applying || !entry.segmentation_id}
            title={
              entry.segmentation_id
                ? undefined
                : "This adapter is not attached to a segmentation, so there is nothing to apply it to."
            }
            onClick={() => void handleApply()}
          >
            {applying ? "Applying…" : entry.applied_at ? "Apply again" : "Apply"}
          </Button>
        </div>
      </div>
      {applyError ? (
        <p className="m-0 text-xs text-red-700">{applyError}</p>
      ) : null}
    </li>
  );
}

function PackCard({
  pack,
  lastFailedInstall = null,
  onInstalled,
}: {
  pack: ModelPack;
  /**
   * The most recent FAILED install job for this pack, read from the jobs API.
   *
   * This is what makes a failure survive navigation: the local download state
   * below dies with the unmount, and without this the card greeted the user
   * with "Not installed yet" and a fresh Download button however many times
   * the same install had already failed.
   */
  lastFailedInstall?: LastFailedInstall | null;
  onInstalled: () => void;
}) {
  const runnability = packRunnability(pack);
  const [sourcePath, setSourcePath] = useState("");
  const [installing, setInstalling] = useState(false);
  const [installError, setInstallError] = useState<string | null>(null);
  const [showInstall, setShowInstall] = useState(false);

  const handleInstall = useCallback(async () => {
    const trimmed = sourcePath.trim();
    if (!trimmed) return;
    setInstalling(true);
    setInstallError(null);
    try {
      await installModelPack(pack.id, trimmed);
      setShowInstall(false);
      setSourcePath("");
      onInstalled();
    } catch (err) {
      setInstallError(
        extractApiErrorMessage(err, `${pack.id} could not be installed.`)
      );
    } finally {
      setInstalling(false);
    }
  }, [onInstalled, pack.id, sourcePath]);

  // --- Download: install with no source, watched through the job it returns.
  const [downloadJobId, setDownloadJobId] = useState<string | null>(null);
  const [downloadStarting, setDownloadStarting] = useState(false);
  /** The refusal to *start* — the server's own sentence, verbatim. */
  const [downloadError, setDownloadError] = useState<string | null>(null);
  /** A local fact the job row can no longer say (it was deleted). */
  const [downloadNote, setDownloadNote] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);

  /**
   * An install already underway that this card did not start — the
   * installer's first-launch downloads, or another window's click. uat13 #1:
   * while all four installer-requested downloads were RUNNING, every card
   * still said "not installed" with a live Download button, and clicking it
   * queued a real duplicate gigabyte download. The card adopts the job the
   * catalogue names: same poll, same progress panel, same cancel — instead of
   * the Download button.
   */
  const catalogueInstall = pack.installed ? null : (pack.active_install ?? null);
  /** The job this card watches: its own attempt first, else the adopted one. */
  const watchedJobId =
    downloadJobId ?? (catalogueInstall ? String(catalogueInstall.job_id) : null);
  const { job: downloadJob } = useJobProgress(watchedJobId);
  const downloadStatus = downloadJob?.status;
  /** True until the first poll of an adopted job answers. */
  const adoptedSnapshotOnly =
    downloadJobId === null && catalogueInstall !== null && downloadJob === null;

  /**
   * Report a finished download exactly once.
   *
   * `onInstalled` is an inline arrow in the parent and changes identity on
   * every render, so an effect keyed on it alone would refetch the catalogue
   * in a loop once the status settled on SUCCESS.
   */
  const reportedInstallRef = useRef(false);
  useEffect(() => {
    if (downloadStatus === "SUCCESS" && !reportedInstallRef.current) {
      reportedInstallRef.current = true;
      onInstalled();
    }
  }, [downloadStatus, onInstalled]);

  const handleDownload = useCallback(async () => {
    setDownloadStarting(true);
    setDownloadError(null);
    setDownloadNote(null);
    setCancelError(null);
    reportedInstallRef.current = false;
    try {
      const response = await installModelPack(pack.id);
      if (response.job_id) {
        setDownloadJobId(response.job_id);
      } else {
        // The "already installed" short-circuit: nothing to poll, and the
        // catalogue is what says so.
        onInstalled();
      }
    } catch (err) {
      // Verbatim. On a backend without the registry download this is the 501
      // that names the release-bundle route — the only honest thing to show.
      setDownloadError(
        extractApiErrorMessage(err, `${pack.title} could not be downloaded.`)
      );
      // 409: an install for this pack is already active — the race where the
      // click landed before the catalogue could say so. The refetch brings
      // `active_install` in, so the card flips to the running job instead of
      // offering the button that just refused.
      if (err instanceof ApiRequestError && err.status === 409) {
        onInstalled();
      }
    } finally {
      setDownloadStarting(false);
    }
  }, [onInstalled, pack.id, pack.title]);

  /**
   * A queued job and a running one leave by different doors, same as the
   * Analysis screen's Cancel: `POST /cancel/` refuses anything not RUNNING
   * with a 409, and `DELETE` is the only exit a queued job has. A deleted row
   * cannot be polled, so the note about it is kept locally.
   */
  const handleCancelDownload = useCallback(async () => {
    if (!watchedJobId) return;
    setCancelling(true);
    setCancelError(null);
    // Before the first poll of an adopted job answers, the catalogue snapshot
    // is the only status there is; a RUNNING job must go through /cancel/,
    // and DELETE is the only exit a queued one has.
    const running =
      downloadStatus === "RUNNING" ||
      (downloadStatus === undefined && catalogueInstall?.status === "RUNNING");
    try {
      if (running) {
        await cancelJob(watchedJobId);
      } else {
        await deleteJob(watchedJobId);
        setDownloadNote("Download removed from the queue before it started.");
        if (downloadJobId !== null) {
          setDownloadJobId(null);
        } else {
          // An adopted job: only a catalogue refetch clears `active_install`,
          // and polling the deleted row would 404 forever.
          onInstalled();
        }
      }
    } catch (err) {
      setCancelError(
        extractApiErrorMessage(err, "The download could not be cancelled.")
      );
    } finally {
      setCancelling(false);
    }
  }, [catalogueInstall, downloadJobId, downloadStatus, onInstalled, watchedJobId]);

  const downloadActive =
    watchedJobId !== null &&
    (downloadJob === null ||
      downloadStatus === "PENDING" ||
      downloadStatus === "RUNNING" ||
      downloadStatus === "RETRY");

  /**
   * Show the persisted failure only when no live attempt owns the card.
   *
   * The moment an attempt exists here — started by the user, or adopted from
   * the catalogue — that attempt's own live states (progress, verbatim FAILED
   * message, cancelled note) are fresher than the jobs-API snapshot, and
   * rendering both would print the same failure twice.
   */
  const showLastFailedInstall = lastFailedInstall !== null && watchedJobId === null;

  return (
    <div
      className={cx(
        "rounded-md border p-3",
        runnability.state === "blocked"
          ? "border-slate-200 bg-slate-50"
          : "border-slate-200 bg-white"
      )}
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <p
            className={cx(
              "m-0 text-sm font-semibold",
              runnability.state === "blocked" ? "text-slate-600" : "text-slate-900"
            )}
          >
            {pack.title}
          </p>
          <p className="m-0 font-mono text-xs text-slate-500">{pack.id}</p>
        </div>
        <RunnabilityBadge runnability={runnability} />
      </div>

      <RunnabilityReason runnability={runnability} className="m-0 mt-2 text-xs text-amber-800" />

      <p className="m-0 mt-2 text-xs text-slate-500">
        {ORGANELLE_LABELS[pack.organelle] ?? pack.organelle} · {pack.neck} ·{" "}
        {pack.decoder} · tile {pack.tile_size} ·{" "}
        {pack.canonical_nm === null
          ? "native resolution"
          : `${formatNumber(pack.canonical_nm, 1)} nm/px`}
      </p>
      <p className="m-0 mt-1 text-xs text-slate-500">
        {pack.installed
          ? "Installed"
          : `${formatBytes(pack.download_bytes)} to install`}
        {pack.encoder_tier ? ` · encoder: ${pack.encoder_tier}` : ""}
        {pack.licence ? ` · ${pack.licence}` : ""}
      </p>

      {!pack.installed ? (
        <div className="mt-2 flex flex-col gap-2">
          {showLastFailedInstall ? (
            // The install history this card used to forget on unmount: the
            // job's message verbatim (it is the only text separating a dead
            // network from a digest mismatch from a full disk), when it
            // failed, and a button that admits it is a retry.
            <p
              className="m-0 text-xs text-red-700"
              role="alert"
              data-testid={`last-failed-install-${pack.id}`}
            >
              <strong>
                The last download of this pack failed
                {lastFailedInstall.finishedAt
                  ? ` (${formatTimestamp(lastFailedInstall.finishedAt)})`
                  : ""}
                .
              </strong>{" "}
              {lastFailedInstall.message ||
                "The job recorded no reason for it."}
            </p>
          ) : null}
          {downloadActive ? (
            <div
              className="rounded-md border border-slate-200 bg-slate-50 p-2"
              data-testid={`download-progress-${pack.id}`}
            >
              <div className="flex items-center justify-between gap-2">
                <p className="m-0 text-xs font-medium text-slate-700">
                  {downloadStatus === "RUNNING"
                    ? "Downloading…"
                    : adoptedSnapshotOnly && catalogueInstall
                      ? // The catalogue's own words about a job this card did
                        // not start, until the first poll answers: "Queued",
                        // or "Installing — 214.0 MB of 1.2 GB".
                        describeActiveInstall(catalogueInstall)
                      : "Waiting in the queue…"}
                </p>
                <Button
                  size="sm"
                  disabled={cancelling || downloadJob?.cancel_requested}
                  onClick={() => void handleCancelDownload()}
                >
                  {cancelling ? "Cancelling…" : "Cancel"}
                </Button>
              </div>
              <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-slate-200">
                <div
                  className="h-full rounded-full bg-cyan-600 transition-[width]"
                  style={{
                    width: `${Math.min(
                      100,
                      Math.max(
                        0,
                        downloadJob?.progress ??
                          (catalogueInstall
                            ? (activeInstallPercent(catalogueInstall) ?? 0)
                            : 0)
                      )
                    )}%`,
                  }}
                />
              </div>
              {/* The job's own words — the worker reports bytes there. */}
              {downloadJob?.message ? (
                <p className="m-0 mt-1 text-xs text-slate-600">
                  {downloadJob.message}
                </p>
              ) : null}
              {downloadJob?.cancel_requested ? (
                <p className="m-0 mt-1 text-xs text-amber-800">
                  Cancellation requested. The worker stops at its next
                  checkpoint.
                </p>
              ) : null}
              {cancelError ? (
                <p className="m-0 mt-1 text-xs text-red-700" role="alert">
                  {cancelError}
                </p>
              ) : null}
            </div>
          ) : (
            <div className="flex flex-wrap items-center gap-2">
              <Button
                size="sm"
                variant="primary"
                disabled={downloadStarting}
                onClick={() => void handleDownload()}
              >
                {downloadStarting
                  ? "Requesting…"
                  : `${
                      // After a recorded failure the same action must call
                      // itself what it is. "Download" beside a failure notice
                      // reads as a different, untried route; it is not.
                      showLastFailedInstall ? "Retry download" : "Download"
                    } (${formatBytes(pack.download_bytes)})`}
              </Button>
            </div>
          )}
          {downloadStatus === "FAILED" && !downloadActive ? (
            // Verbatim from the job row: `Job.message` is the only text that
            // says whether the network died, a digest mismatched, or the disk
            // filled, and each of those wants a different response.
            <p className="m-0 text-xs text-red-700" role="alert">
              <strong>The download failed.</strong>{" "}
              {downloadJob?.message?.trim() ||
                "The job recorded no reason for it."}
            </p>
          ) : null}
          {downloadStatus === "CANCELLED" ? (
            <p className="m-0 text-xs text-amber-800" role="status">
              Download cancelled. Nothing was installed; download again to
              restart it.
            </p>
          ) : null}
          {downloadNote ? (
            <p className="m-0 text-xs text-amber-800" role="status">
              {downloadNote}
            </p>
          ) : null}
          {downloadError ? (
            <p className="m-0 text-xs text-red-700" role="alert">
              {downloadError}
            </p>
          ) : null}
          {showInstall ? (
            <div className="flex flex-col gap-2">
              <label
                className="text-xs font-semibold uppercase tracking-wide text-slate-500"
                htmlFor={`install-${pack.id}`}
              >
                Directory on this machine
              </label>
              <input
                id={`install-${pack.id}`}
                className="h-9 w-full rounded-md border border-slate-300 bg-white px-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                value={sourcePath}
                placeholder="e.g. D:\quantem-models-0.1.0"
                disabled={installing}
                onChange={(event) => setSourcePath(event.target.value)}
              />
              {/* The help used to describe only the maintainer's layout -- "the
                  folder holding head.pt and resolved_config.yaml", with
                  D:\models\mito_quantem as the example -- which is not a shape
                  anyone who downloaded a release has. A release unzips to
                  MANIFEST.json beside packs/<family>__<organelle>/, and the
                  endpoint accepts any level of that. The shapes below are the
                  ones `_resolve_local_source` names in its own error. */}
              <div className="m-0 text-xs text-slate-500">
                Any of:
                <ul className="m-0 mt-1 list-disc pl-4">
                  <li>
                    the folder you unzipped a QuantEM model release into (it
                    holds <code>MANIFEST.json</code> and <code>packs/</code>);
                  </li>
                  <li>
                    that release's <code>packs/</code> folder;
                  </li>
                  <li>
                    just this pack's folder inside it,{" "}
                    <code>packs/{pack.id.replace(":", "__")}/</code>;
                  </li>
                  <li>
                    or a folder holding <code>head.pt</code> and{" "}
                    <code>resolved_config.yaml</code> (training output, not a
                    release).
                  </li>
                </ul>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button
                  size="sm"
                  variant="primary"
                  disabled={installing || !sourcePath.trim()}
                  onClick={() => void handleInstall()}
                >
                  {installing ? "Installing…" : "Install from this folder"}
                </Button>
                <Button
                  size="sm"
                  disabled={installing}
                  onClick={() => {
                    setShowInstall(false);
                    setInstallError(null);
                  }}
                >
                  Cancel
                </Button>
              </div>
              {installError ? (
                <p className="m-0 text-xs text-red-700">{installError}</p>
              ) : null}
            </div>
          ) : (
            <div>
              <Button
                size="sm"
                disabled={downloadActive}
                onClick={() => setShowInstall(true)}
              >
                Install from a local folder
              </Button>
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
