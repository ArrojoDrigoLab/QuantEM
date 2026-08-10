"""Scan completed SSL experiments and train + test the fixed decoder probes on each.

For every experiment under ``runs/`` it finds the final-stage encoder, takes the last N (default 3)
checkpoints, and for each (checkpoint x organelle x decoder) trains the frozen-encoder decoder head
and evaluates it on the held-out test split, writing per-test-image metrics to a clean CSV.

It is idempotent (skips heads already done), runs the heads in parallel across GPUs (one head
per GPU by default; ``--per-gpu 2`` packs two small heads on each GPU), and shows a single-line,
width-independent progress bar with overall ETA.

    python -m encoder_evaluation.train_decoders_and_test \
        --runs-root <encoder runs> --derived-root <ground-truth tiles> --out <results dir> \
        --decoders linear light_conv --organelles mito er --n-checkpoints 3 [--per-gpu 2]

Module top-level stays torch-free so each spawned worker can pin ``CUDA_VISIBLE_DEVICES`` before
importing torch.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from em_ssl.utils.checkpoint_index import CheckpointIndex  # torch-free
from encoder_evaluation.constants import DEFAULT_DERIVED_ROOT, VALID_ORGANELLES  # torch-free

_STAGE_RE = re.compile(r"stage(\d+)")
_SHORT_DEC = {"linear": "lin", "light_conv": "lc"}

def _native_tile_size(manifest, exp_name: str, fallback: int) -> int:
    """The encoder's final-stage SSL crop (what it was actually trained at), for native-res probing.

    Primary source is the manifest ``crop_schedule`` (last stage's global crop). Falls back to parsing
    the experiment name (…_768, …_1024, …512to768to1024 -> the LAST 512/768/1024 token = the final
    stage), then to ``fallback`` (the global --tile-size). Keeps the probe honest: a 768/1024-trained
    encoder is fed 768/1024 windows, not the 512 that under-feeds its receptive field.
    """
    sched = list(getattr(manifest, "crop_schedule", None) or [])
    for stage in reversed(sched):
        if isinstance(stage, dict):
            for k in ("global_crops_size", "global_crop_size", "crop_size", "size"):
                v = stage.get(k)
                if v:
                    return int(v)
    nums = [int(n) for n in re.findall(r"\d+", exp_name or "") if int(n) in (512, 768, 1024)]
    if nums:
        return nums[-1]
    return int(fallback)

# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    experiment: str
    run_dir: str
    step: int
    ckpt_path: str
    organelle: str
    combos: list  # [(decoder, fraction), ...] still to do for this (ckpt, organelle)
    manifest: object  # EncoderManifest (picklable, torch-free)
    phase: int = 1  # 1 = final checkpoints, 2 = the earlier ones
    priority: int = 1  # queue order: 0 = priority-experiment finals, 1 = other finals, 2 = phase-2

def _final_stage_index(exp_dir: Path) -> Path | None:
    """Pick the most-continued stage dir (highest ``stageN``) that has a checkpoint_index.json."""
    candidates = []
    for d in sorted(exp_dir.iterdir()):
        if d.is_dir() and (d / CheckpointIndex.FILENAME).exists():
            m = _STAGE_RE.search(d.name)
            candidates.append((int(m.group(1)) if m else -1, d))
    if not candidates:
        # some single-stage runs may put the index directly in the experiment dir
        return exp_dir if (exp_dir / CheckpointIndex.FILENAME).exists() else None
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]

def _head_dir(out_root: Path, exp: str, step: int, organelle: str, decoder: str, frac: float) -> Path:
    return out_root / exp / f"s{step}" / f"{organelle}__{decoder}__f{int(round(frac * 100))}"

def _head_done(out_root: Path, exp: str, step: int, organelle: str, decoder: str, frac: float) -> bool:
    return (_head_dir(out_root, exp, step, organelle, decoder, frac) / "done.json").exists()

def discover_checkpoints_on_disk(run_dir: Path, manifest) -> list:
    """Find exported encoder checkpoints on disk when the index's ``checkpoints`` list is empty.

    Robust to indices that were reset or re-prepared — manifest intact, records wiped — but that still
    have the real teacher files on disk at ``eval/<step>/teacher_checkpoint.pth``. The step is parsed
    from the path.
    """
    from em_ssl.utils.checkpoint_index import CheckpointRecord

    run_dir = Path(run_dir)
    recs = []
    for p in sorted((run_dir / "eval").glob("*/teacher_checkpoint.pth")):
        digits = re.findall(r"\d+", p.parent.name)
        recs.append(CheckpointRecord(step=int(digits[-1]) if digits else -1, kind="teacher",
                                     path=str(p)))
    return recs

def scan_jobs(runs_root: Path, out_root: Path, organelles, decoders, n_ckpts: int,
              steps: list[int] | None, all_stages: bool, log, fractions=(1.0,),
              priority_experiments=()) -> list[Job]:
    """Build the job list, ordered into two phases:

    Phase 1 = each experiment's final checkpoint (any ``priority_experiments`` first,
    then all the others). Phase 2 = the remaining (earlier) checkpoints for every experiment.
    """
    jobs: list[Job] = []
    prio = {a.upper() for a in priority_experiments}
    if not runs_root.exists():
        log(f"[scan] runs root not found: {runs_root}")
        return jobs
    for exp_dir in sorted(p for p in runs_root.iterdir() if p.is_dir() and p.name != "reports"):
        stage_dirs = ([d for d in sorted(exp_dir.iterdir())
                       if d.is_dir() and (d / CheckpointIndex.FILENAME).exists()]
                      if all_stages else [_final_stage_index(exp_dir)])
        for run_dir in filter(None, stage_dirs):
            try:
                idx = CheckpointIndex.load(run_dir)
            except Exception as exc:
                log(f"[scan] {run_dir.name}: bad index ({exc!r})")
                continue
            recs = idx.teacher_checkpoints()
            if not recs:  # index records wiped/never written -> recover the real files from disk
                recs = discover_checkpoints_on_disk(run_dir, idx.manifest)
                if recs:
                    log(f"[scan] {exp_dir.name}: index lists 0 checkpoints; recovered {len(recs)} "
                        f"teacher ckpt(s) from disk (eval/*/teacher_checkpoint.pth)")
            recs = sorted(recs, key=lambda r: r.step)
            if not recs:
                log(f"[scan] {exp_dir.name}/{run_dir.name}: no checkpoints (index empty + none on "
                    f"disk) — skip")
                continue
            chosen = sorted(([r for r in recs if r.step in set(steps)] if steps else recs[-n_ckpts:]),
                            key=lambda r: r.step)
            if not chosen:
                continue
            final_step = chosen[-1].step  # highest step = the "final" checkpoint
            arm = exp_dir.name.split("_")[0].upper()   # the run-name prefix, used for ordering
            for rec in chosen:
                if not Path(rec.path).exists():
                    log(f"[scan] {exp_dir.name} s{rec.step}: weights missing ({rec.path}) — skip")
                    continue
                if rec.step == final_step:
                    phase, priority = 1, (0 if arm in prio else 1)
                else:
                    phase, priority = 2, 2
                for organelle in organelles:
                    combos = [(d, fr) for d in decoders for fr in fractions
                              if not _head_done(out_root, exp_dir.name, rec.step, organelle, d, fr)]
                    if combos:
                        jobs.append(Job(exp_dir.name, str(run_dir), rec.step, rec.path,
                                        organelle, combos, idx.manifest, phase=phase, priority=priority))
    # queue order: priority-experiment finals -> other finals -> phase 2, deterministic within a bucket
    jobs.sort(key=lambda j: (j.priority, j.experiment, j.step, j.organelle))
    return jobs

# --------------------------------------------------------------------------- #
# Worker (runs in a spawned process pinned to one GPU)
# --------------------------------------------------------------------------- #
def _worker(wid: int, gpu_id, jobs, progress_q, cfg_dict, derived_root, out_root, save,
            native_tile: bool = False):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # torch + harness imported here so CUDA_VISIBLE_DEVICES is already set
    import dataclasses

    import torch

    from encoder_evaluation.harness.config import ProbeConfig
    from encoder_evaluation.harness.dataset import load_manifest, subset_fraction
    from encoder_evaluation.harness.encoders import FrozenEncoder
    from encoder_evaluation.harness.evaluate import evaluate_heads, shutdown_eval_pool
    from encoder_evaluation.harness.feature_cache import build_train_cache, cache_subset_indices, train_head_cached
    from encoder_evaluation.harness.train import train_head

    base_cfg = ProbeConfig(**cfg_dict)
    device = "cuda" if (gpu_id is not None and torch.cuda.is_available()) else "cpu"

    def send(*msg):
        progress_q.put(msg)

    # Announce which physical GPU this worker landed on.
    if gpu_id is not None and device == "cpu":
        send("log", f"[worker {wid}] WARNING: gpu {gpu_id} requested but CUDA is unavailable — running on CPU (very slow)")
    elif device == "cuda":
        try:
            send("log", f"[worker {wid}] gpu {gpu_id} -> {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    def _free_cuda():
        try:
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass

    # The whole loop is wrapped so `worker_done` always fires (in finally); otherwise a crash mid-job
    # would leave the orchestrator waiting on a worker that never reports, holding its GPU idle.
    try:
        for job in iter(jobs.get, None):
            man = job.manifest
            if native_tile:
                # Probe this encoder at its own native SSL crop (512/768/1024), rebuilding the cfg from
                # the pristine cfg_dict so nothing drifts across jobs. Shrink the sliding-window batch at
                # large tiles to stay within VRAM (2.25x/4x the tokens at 768/1024).
                _t = _native_tile_size(man, job.experiment, cfg_dict.get("tile_size") or 512)
                _eb = cfg_dict.get("eval_batch_windows", 32)
                _eb = min(_eb, 8) if _t >= 1024 else (min(_eb, 16) if _t >= 768 else _eb)
                base_cfg = ProbeConfig(**{**cfg_dict, "tile_size": _t, "eval_batch_windows": _eb})
            layers = base_cfg.resolved_layers(man.depth)
            desc = (f"{job.experiment}/s{job.step} {job.organelle}"
                    + (f" @{base_cfg.tile_size}px" if native_tile else ""))
            send("start", wid, desc)
            n_combos = len(job.combos)
            ticked = 0
            encoder = None
            cache = None

            def tick():  # count each head exactly once so the P1/P2 total can never stall
                nonlocal ticked
                if ticked < n_combos:
                    ticked += 1
                    send("decoder_done", wid, job.phase)

            try:
                try:
                    train_all = load_manifest(derived_root, job.organelle, "train")
                    test_recs = load_manifest(derived_root, job.organelle, "test")
                except FileNotFoundError as exc:
                    send("log", f"[{desc}] no derived data ({exc}) — skipping")
                    continue
                if not train_all or not test_recs:
                    send("log", f"[{desc}] empty train/test for {job.organelle} — skipping")
                    continue
                try:
                    encoder = FrozenEncoder.from_manifest(job.ckpt_path, man, base_cfg.tile_size,
                                                          apply_encoder_norm=base_cfg.apply_encoder_norm)
                except Exception as exc:
                    send("log", f"[{desc}] encoder load failed ({exc!r}) — skipping")
                    continue
                # Common compare region: the encoder reads the full tile and crops its central tokens
                # to this before the decoder, so the decoder output and the cropped labels agree.
                encoder.compare_tile = base_cfg.compare_tile

                # Encoder adaptation: train a subset of the encoder alongside the head — LoRA adapters,
                # the LayerNorms, the last N blocks, or the whole backbone, per ``adapt`` (see
                # harness/encoder_adaptation.py). The same mechanism as the encoder-adaptation experiment
                # in segmentation training. Requires uncached features; cached features are static and
                # would defeat adaptation.
                adapted = str(getattr(base_cfg, "adapt", "frozen") or "frozen").lower() != "frozen"
                if adapted:
                    from encoder_evaluation.harness.encoder_adaptation import apply_adaptation
                    apply_adaptation(encoder, base_cfg.adapt, base_cfg.adapt_params or {})
                    send("log", f"[{desc}] encoder adapt={base_cfg.adapt} params={base_cfg.adapt_params} "
                                f"adapter_lr={base_cfg.adapter_lr}")

                # optional: forward the frozen encoder over the train set once, reuse for every decoder+fraction
                if base_cfg.cache_train_features and not adapted:
                    send("phase", wid, "cache", "cache", 0.0)
                    try:
                        cache = build_train_cache(encoder, train_all, base_cfg, derived_root, layers, device)
                    except Exception as exc:
                        send("log", f"[{desc}] feature cache failed ({exc!r}); falling back to per-step forward")
                        _free_cuda()

                # --- train every (decoder, fraction) head ---
                # An adapted encoder trains its own params, so heads cannot share one: each would start
                # from the previous head's encoder and all would be scored against the last one. Rebuild
                # and re-adapt per head in that case. A frozen encoder is unchanged by training and is
                # shared, which is what makes the single shared evaluation pass below valid.
                trained = {}  # (mode, frac) -> (decoder, n_train)
                results = {}  # key -> metrics, filled per head when adapted
                n_heads_done = 0  # heads attempted, so a failed head still forces a fresh encoder
                for mode, frac in job.combos:
                    if adapted and n_heads_done:
                        encoder = FrozenEncoder.from_manifest(
                            job.ckpt_path, man, base_cfg.tile_size,
                            apply_encoder_norm=base_cfg.apply_encoder_norm)
                        encoder.compare_tile = base_cfg.compare_tile
                        apply_adaptation(encoder, base_cfg.adapt, base_cfg.adapt_params or {})
                    cfg = dataclasses.replace(base_cfg, decoder=mode)
                    short = f"{_SHORT_DEC.get(mode, mode)}@{int(round(frac * 100))}"
                    send("phase", wid, short, "train", 0.0)

                    def tlog(step, _m, _max=cfg.max_steps, _s=short):
                        send("phase", wid, _s, "train", min((step + 1) / max(_max, 1), 1.0))

                    try:
                        if cache is not None:
                            sub = cache_subset_indices(cache, frac, cfg.seed)
                            decoder = train_head_cached(cache, sub, cfg, mode, man.embedding_dim, len(layers),
                                                        man.patch_size, device, logger=tlog, tag=f"{desc} {mode}")
                            n_train = len(sub)
                        else:
                            train_recs = subset_fraction(train_all, frac, seed=base_cfg.seed)
                            decoder = train_head(encoder, train_recs, cfg, derived_root, layers, device,
                                                 logger=tlog, tag=f"{desc} {mode} f{frac}")
                            n_train = len(train_recs)
                        trained[(mode, frac)] = (decoder, n_train)
                        if adapted:
                            # This head owns its encoder, so score it now, before the next head replaces it.
                            key = f"{mode}__f{frac}"
                            try:
                                results.update(evaluate_heads(
                                    encoder, {key: decoder}, test_recs, base_cfg, derived_root, layers,
                                    device, on_crop=lambda i, n: send("phase", wid, "eval", "eval", i / n)))
                            except Exception as exc:
                                send("log", f"[{desc} {mode} f{frac}] EVAL FAILED: {exc!r}")
                                _free_cuda()
                        tick()  # tick per decoder as it finishes training (smooth counter)
                    except Exception as exc:
                        send("log", f"[{desc} {mode} f{frac}] TRAIN FAILED: {exc!r}")
                        _free_cuda()  # a caught OOM otherwise leaves the allocator wedged for the next combo
                        tick()
                    finally:
                        n_heads_done += 1

                # training done: drop the (up to 40 GB) feature cache before eval so the metric pool
                # has RAM headroom — the cache is never needed past training.
                cache = None
                _free_cuda()

                # --- frozen encoder: score every head in one shared pass over the test set ---
                if trained:
                    if not adapted:
                        dmap = {f"{m}__f{fr}": d for (m, fr), (d, _n) in trained.items()}
                        try:
                            results = evaluate_heads(encoder, dmap, test_recs, base_cfg, derived_root, layers, device,
                                                     on_crop=lambda i, n: send("phase", wid, "eval", "eval", i / n))
                        except Exception as exc:
                            send("log", f"[{desc}] EVAL FAILED: {exc!r}")
                            _free_cuda()
                            results = {}
                    for (m, fr), (d, n_train) in trained.items():
                        key = f"{m}__f{fr}"
                        if key in results:  # counter already ticked at train; this only writes metrics
                            _write_head_outputs(out_root, job, m, fr, man, base_cfg, results[key],
                                                save_decoder=save, decoder=d if save else None, torch=torch,
                                                n_train=n_train)
            except Exception as exc:
                # OOM or native fault at an unguarded line: log, free the GPU, and keep the worker
                # alive for the next job instead of exiting and leaving this GPU idle.
                send("log", f"[{desc}] JOB CRASHED ({type(exc).__name__}: {exc}) — worker recovering "
                            f"(re-run later fills the gap; done.json is idempotent)")
                _free_cuda()
            finally:
                while ticked < n_combos:  # never leave heads uncounted, else the counter stalls forever
                    tick()
                encoder = None
                cache = None
                _free_cuda()
    finally:
        shutdown_eval_pool()  # tear down the metric pool so atexit cleanup does not hang
        send("worker_done", wid)

_PER_IMAGE_COLS = [
    "experiment", "run_id", "framework", "arch", "embedding_dim", "step", "organelle", "decoder",
    "label_fraction", "sample_id", "dataset", "collection", "crop_id", "subgroup", "modality",
    "scale_band", "tissue_context", "species_group", "orientation", "plane_k", "gt_is_instance",
    "dice", "iou", "precision", "recall", "boundary_f1", "boundary_iou", "hd95", "cldice", "auprc",
    "pq", "sq", "rq", "ap", "vi", "n_gt_inst", "n_pred_inst",
    "gt_fg", "pred_fg", "valid_px", "excluded", "theta_px", "dilation_px",
]

def _write_head_outputs(out_root, job, mode, frac, man, cfg, result, save_decoder, decoder, torch,
                        n_train):
    d = _head_dir(Path(out_root), job.experiment, job.step, job.organelle, mode, frac)
    d.mkdir(parents=True, exist_ok=True)
    base = {"experiment": job.experiment, "run_id": man.run_id, "framework": man.framework,
            "arch": man.arch, "embedding_dim": man.embedding_dim, "step": job.step,
            "organelle": job.organelle, "decoder": mode, "label_fraction": frac}
    with open(d / "per_image_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_PER_IMAGE_COLS)
        w.writeheader()
        for m in result["per_crop"]:
            w.writerow({**base, **{k: m.get(k) for k in _PER_IMAGE_COLS if k not in base}})
    (d / "summary.json").write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    if save_decoder and decoder is not None:
        torch.save(decoder.state_dict(), d / "decoder.pt")
    s = result["summary"]
    (d / "done.json").write_text(json.dumps({
        "ckpt_path": job.ckpt_path, "step": job.step, "organelle": job.organelle, "decoder": mode,
        "label_fraction": frac, "n_train": n_train, "macro": s["macro"],
        "worst_subgroup": s.get("worst_subgroup"), "macro_ci": s.get("macro_ci"),
        "n_test": s["n_crops"], "n_evaluated": s["n_evaluated"],
    }, indent=2), encoding="utf-8")

# --------------------------------------------------------------------------- #
# Progress bar (single line, width-independent) + main
# --------------------------------------------------------------------------- #
def _fmt_t(s: float) -> str:
    s = int(s)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def _detect_gpus() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True, timeout=10)
        n = sum(1 for ln in out.stdout.splitlines() if ln.strip()) if out.returncode == 0 else 0
        if n > 0:
            return n
    except Exception:
        pass
    # nvidia-smi missing or unusable in this subprocess: ask torch in an isolated child so as not to
    # initialize CUDA in this (spawn) parent. The fallback keeps the runner from dropping to a single
    # worker when nvidia-smi is unavailable.
    try:
        out = subprocess.run([sys.executable, "-c", "import torch;print(torch.cuda.device_count())"],
                             capture_output=True, text=True, timeout=120)
        return int((out.stdout or "0").strip() or 0)
    except Exception:
        return 0

def run(args) -> None:
    try:  # tolerate non-UTF-8 consoles (Windows cp1252) for logs + progress
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    runs_root, out_root = Path(args.runs_root), Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []

    def log(m):  # buffered so it doesn't fight the progress line
        logs.append(m)

    from encoder_evaluation.harness.config import load_probe_config
    cfg = load_probe_config(args.config)
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.tile_size:
        cfg.tile_size = args.tile_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.eval_batch:
        cfg.eval_batch_windows = args.eval_batch
    if args.cache_train_features:
        cfg.cache_train_features = True
    if args.cache_max_tiles is not None:
        cfg.cache_max_tiles = args.cache_max_tiles
    if args.bootstrap_n is not None:
        cfg.bootstrap_n = args.bootstrap_n
    if args.eval_workers is not None:
        cfg.eval_workers = args.eval_workers
    if getattr(args, "cache_on_gpu", None) is not None:
        cfg.cache_on_gpu = args.cache_on_gpu
    cfg_dict = cfg.to_dict()
    fractions = args.fractions if args.fractions else cfg.label_fractions

    organelles = [o for o in args.organelles if o in VALID_ORGANELLES]
    prio = tuple(a.strip().upper() for a in (args.priority or "").split(",") if a.strip())
    jobs = scan_jobs(runs_root, out_root, organelles, args.decoders, args.n_checkpoints,
                     args.steps, args.all_stages, log, fractions=fractions, priority_experiments=prio)
    total = sum(len(j.combos) for j in jobs)
    total_phase = {1: sum(len(j.combos) for j in jobs if j.phase == 1),
                   2: sum(len(j.combos) for j in jobs if j.phase == 2)}
    for m in logs:
        print(m)
    if total == 0:
        print("Nothing to do — all heads already trained (or no completed experiments found).")
        _write_combined(out_root)
        return
    print(f"Phase 1 (final ckpts; {'/'.join(prio) or 'none'} first): {total_phase[1]} decoder(s)  |  "
          f"Phase 2 (earlier ckpts): {total_phase[2]} decoder(s)")

    n_gpus = args.gpus if args.gpus is not None else _detect_gpus()
    if n_gpus <= 0:
        workers, gpu_for = 1, [None]
        print(f"No GPU detected -> 1 CPU worker. {total} decoder(s) across {len(jobs)} job(s).")
    else:
        workers = n_gpus * max(1, args.per_gpu)
        gpu_for = [g for g in range(n_gpus) for _ in range(max(1, args.per_gpu))]
        print(f"{n_gpus} GPU(s) x {args.per_gpu} per GPU = {workers} workers; "
              f"{total} decoder(s) across {len(jobs)} job(s).")

    ctx = mp.get_context("spawn")
    job_q, prog_q = ctx.Queue(), ctx.Queue()
    for j in jobs:
        job_q.put(j)
    for _ in range(workers):
        job_q.put(None)
    procs = [ctx.Process(target=_worker, args=(w, gpu_for[w], job_q, prog_q, cfg_dict,
                                               str(Path(args.derived_root)), str(out_root),
                                               bool(args.save_decoders), bool(args.native_tile_size)))
             for w in range(workers)]
    for p in procs:
        p.start()

    _progress_loop(prog_q, workers, total, total_phase, procs)
    for p in procs:
        p.join()
    dead = [(i, gpu_for[i], p.exitcode) for i, p in enumerate(procs)
            if p.exitcode not in (0, None)]
    for i, g, code in dead:
        print(f"  ! worker {i} (gpu {g}) exited abnormally (exit {code}) — likely an OOM/native crash; "
              f"its heads are unfinished. Re-run the same command to fill them in (idempotent via done.json).")
    _write_combined(out_root)
    print(f"\nDone. Per-image metrics + summaries under {out_root}  "
          f"(combined: all_per_image_metrics.csv, all_summary.csv)")

def _progress_loop(prog_q, workers, total, total_phase=None, procs=None) -> None:
    total_phase = total_phase or {1: total, 2: 0}
    start = time.time()
    done = 0
    done_phase = {1: 0, 2: 0}
    state: dict[int, dict] = {}
    alive = workers
    is_tty = sys.stdout.isatty()
    last_render = 0.0
    last_livecheck = 0.0

    def render(final=False):
        elapsed = time.time() - start
        partial = sum(s.get("frac", 0.0) for s in state.values() if s.get("phase") in ("train", "eval"))
        eff = done + min(partial, max(total - done, 0))
        rate = eff / elapsed if elapsed > 0 else 0
        eta = (total - eff) / rate if rate > 0 and eff < total else 0
        pct = int(100 * done / total) if total else 100
        inprog = 0 if final else max(alive, 0)  # heads actively training/evaluating right now
        line = (f"[P1 {done_phase[1]}/{total_phase[1]}  P2 {done_phase[2]}/{total_phase[2]}] "
                f"{pct}% | {inprog} in progress | "
                f"{_fmt_t(elapsed)} elapsed, ETA {_fmt_t(eta) if eta else '--:--'}")
        cols = shutil.get_terminal_size((100, 20)).columns
        if is_tty:
            sys.stdout.write("\r" + line[:cols - 1].ljust(cols - 1))
            sys.stdout.flush()
        elif final or (time.time() - last_render) > 15:
            print(line)

    while alive > 0:
        try:
            msg = prog_q.get(timeout=0.25)
        except Exception:
            # No message. If every worker process has exited (some may have been OS-killed / segfaulted
            # without sending 'worker_done'), stop instead of waiting on them forever.
            if procs is not None and (time.time() - last_livecheck) > 2.0:
                last_livecheck = time.time()
                n_live = sum(1 for p in procs if p.is_alive())
                if n_live == 0:
                    if alive > 0:
                        print(f"\n  ! {alive} worker(s) exited without reporting done "
                              f"(crash/OOM-kill); stopping. Re-run to finish remaining heads.")
                    break
            render()
            continue
        kind = msg[0]
        if kind == "start":
            _, wid, desc = msg
            state[wid] = {"desc": desc, "phase": "", "frac": 0.0, "mode": ""}
        elif kind == "phase":
            _, wid, mode, phase, frac = msg
            state.setdefault(wid, {"desc": "?"}).update(mode=mode, phase=phase, frac=frac)
        elif kind == "decoder_done":
            done += 1
            job_phase = msg[2] if len(msg) > 2 else 1
            done_phase[job_phase] = done_phase.get(job_phase, 0) + 1
            state.get(msg[1], {})["frac"] = 0.0
        elif kind == "log":
            if is_tty:
                sys.stdout.write("\r" + " " * (shutil.get_terminal_size((100, 20)).columns - 1) + "\r")
            print(msg[1])
        elif kind == "worker_done":
            alive -= 1
        if is_tty and (time.time() - last_render) > 0.15:
            render()
            last_render = time.time()
    render(final=True)

def _write_combined(out_root: Path) -> None:
    """Concatenate every head's per-image CSV + summary into two clean top-level CSVs."""
    per_rows, per_header = [], None
    sum_rows = []
    for done_file in out_root.glob("*/s*/*/done.json"):
        head_dir = done_file.parent
        pcsv = head_dir / "per_image_metrics.csv"
        if pcsv.exists():
            with open(pcsv, encoding="utf-8") as fh:
                rdr = csv.reader(fh)
                hdr = next(rdr, None)
                if hdr and per_header is None:
                    per_header = hdr
                per_rows.extend(row for row in rdr)
        meta = json.loads(done_file.read_text(encoding="utf-8"))
        mac = meta.get("macro", {}) or {}
        worst = meta.get("worst_subgroup", {}) or {}
        ci = meta.get("macro_ci", {}) or {}
        parts = head_dir.parts
        row = {
            "experiment": parts[-3], "step": head_dir.parent.name.lstrip("s"),
            "organelle": meta.get("organelle"), "decoder": meta.get("decoder"),
            "label_fraction": meta.get("label_fraction"), "n_train": meta.get("n_train"),
            "n_test": meta.get("n_test"), "n_evaluated": meta.get("n_evaluated"),
        }
        for k in ("dice", "iou", "precision", "recall", "boundary_f1", "boundary_iou",
                  "hd95", "cldice", "auprc", "pq", "sq", "rq", "ap", "vi"):
            if mac.get(k) is not None:
                row[f"macro_{k}"] = mac.get(k)
                if worst.get(k):
                    row[f"worst_{k}"] = worst[k]["value"]
                if ci.get(k):
                    row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = ci[k]["lo"], ci[k]["hi"]
        sum_rows.append(row)
    if per_header:
        with open(out_root / "all_per_image_metrics.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(per_header)
            w.writerows(per_rows)
    if sum_rows:
        lead = ["experiment", "step", "organelle", "decoder", "label_fraction", "n_train",
                "n_test", "n_evaluated"]
        rest = sorted({k for r in sum_rows for k in r} - set(lead))
        fields = lead + rest
        with open(out_root / "all_summary.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(sorted(sum_rows, key=lambda r: (r["experiment"], r["step"], r["organelle"],
                                                        r["decoder"], r.get("label_fraction") or 0)))

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train + test fixed decoder probes across SSL experiments.")
    p.add_argument("--runs-root", default="runs", help="SSL output root (contains <experiment>/stageN/).")
    p.add_argument("--derived-root", default=DEFAULT_DERIVED_ROOT, required=True,
                   help="Derived ground-truth dataset, from encoder_evaluation.dataprep.build_dataset.")
    p.add_argument("--out", required=True,
                   help="Output root for the per-head records and the rolled-up metric CSVs/JSONs.")
    p.add_argument("--organelles", nargs="+", default=list(VALID_ORGANELLES))
    p.add_argument("--decoders", nargs="+", default=["linear", "light_conv"])
    p.add_argument("--n-checkpoints", type=int, default=3, help="Last N checkpoints per experiment.")
    p.add_argument("--priority", default="",
                   help="Comma-separated experiment arms whose final checkpoint runs first in Phase 1 "
                        "(matched on the run-name prefix). Phase 1 = all final checkpoints, "
                        "Phase 2 = the earlier ones.")
    p.add_argument("--steps", type=int, nargs="*", default=None, help="Explicit checkpoint steps.")
    p.add_argument("--all-stages", action="store_true", help="Include every stage, not just the final.")
    p.add_argument("--gpus", type=int, default=None, help="GPU count (default: auto-detect).")
    p.add_argument("--per-gpu", type=int, default=1, help="Heads packed per GPU (2 ok for ViT-B).")
    p.add_argument("--config", default=None, help="Probe YAML (defaults if omitted).")
    p.add_argument("--max-steps", type=int, default=0, help="Override decoder training steps.")
    p.add_argument("--tile-size", type=int, default=0, help="Override probe tile size (global).")
    p.add_argument("--native-tile-size", action="store_true",
                   help="Probe each encoder at its own native SSL crop (512/768/1024 from the manifest "
                        "crop_schedule) instead of the global --tile-size. Use with a derived dataset "
                        "whose crops are large enough that large-context encoders see real tissue, not "
                        "padding. Auto-shrinks the eval window batch at 768/1024 to stay within VRAM.")
    p.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers per head.")
    p.add_argument("--eval-batch", type=int, default=0, help="Sliding-window tiles per encoder forward (default 32).")
    p.add_argument("--cache-train-features", action="store_true",
                   help="Forward the frozen encoder over the train set once per (ckpt,organelle) and "
                        "train all heads on cached features (much faster; drops per-step augmentation).")
    p.add_argument("--cache-max-tiles", type=int, default=None,
                   help="Cap cached train tiles per (ckpt,organelle) via reservoir subsampling to bound "
                        "RAM (default 6000 ~= 38 GB/ViT-B cache; 0 = uncapped). Two caches must fit in RAM.")
    p.add_argument("--bootstrap-n", type=int, default=None,
                   help="Bootstrap resamples for macro CIs (default 1000). ~300 is plenty and cuts the "
                        "per-head eval CPU cost ~3x; 0 disables CIs.")
    p.add_argument("--eval-workers", type=int, default=None,
                   help="Parallel processes for per-crop eval metrics (default 8; identical results). "
                        "1 = serial. The mito distance-transforms/instance metrics are the GPU-idle cost.")
    p.add_argument("--cache-on-gpu", action=argparse.BooleanOptionalAction, default=None,
                   help="Keep the feature cache in VRAM (default on; auto-falls back to CPU if it won't "
                        "fit) so cached training has no per-step CPU->GPU copy. --no-cache-on-gpu disables.")
    p.add_argument("--fractions", type=float, nargs="*", default=None,
                   help="Label-efficiency fractions (default from config, usually [1.0]).")
    p.add_argument("--save-decoders", action="store_true", help="Also save trained decoder weights.")
    run(p.parse_args(argv))

if __name__ == "__main__":
    main()
