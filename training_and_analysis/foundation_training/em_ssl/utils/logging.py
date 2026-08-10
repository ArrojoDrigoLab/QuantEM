"""Metric logging fan-out: JSONL + CSV always; TensorBoard and WandB optional.

Neither TensorBoard nor WandB is required — if unavailable or disabled they are silently
skipped so a run on a minimal host still gets durable JSONL/CSV logs. Everything a caller passes
to ``log(step, metrics)`` reaches the JSONL and CSV records; TensorBoard receives the scalar
entries only. The decoder-probe harness logs per-step loss components and learning rate this way.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

class MetricLogger:
    def __init__(
        self,
        run_dir: str | os.PathLike,
        tensorboard: bool = True,
        wandb: bool = False,
        wandb_project: str | None = None,
        run_name: str | None = None,
        jsonl_name: str = "training_log.jsonl",
        csv_name: str = "training_log.csv",
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / jsonl_name
        self.csv_path = self.run_dir / csv_name
        self._csv_fields: list[str] = []
        self._jsonl_fp = open(self.jsonl_path, "a", encoding="utf-8")
        self._t0 = _now()

        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))
            except Exception:
                self.tb = None

        self.wandb = None
        if wandb:
            try:
                import wandb as _wandb

                _wandb.init(project=wandb_project or "em-dino-ssl", name=run_name, dir=str(self.run_dir))
                self.wandb = _wandb
            except Exception:
                self.wandb = None

    def log(self, step: int, metrics: dict[str, Any], prefix: str = "") -> None:
        flat = {f"{prefix}{k}": v for k, v in metrics.items()}
        record = {"step": int(step), "wall_time": round(_now() - self._t0, 3), **flat}

        # JSONL
        self._jsonl_fp.write(json.dumps(record, default=_jsonable) + "\n")
        self._jsonl_fp.flush()

        # CSV (rewrite header if new fields appear)
        self._write_csv(record)

        # TensorBoard (scalars only)
        if self.tb is not None:
            for k, v in flat.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    try:
                        self.tb.add_scalar(k, v, step)
                    except Exception:
                        pass

        # WandB
        if self.wandb is not None:
            try:
                self.wandb.log(flat, step=int(step))
            except Exception:
                pass

    def _write_csv(self, record: dict[str, Any]) -> None:
        new_fields = [k for k in record if k not in self._csv_fields]
        if new_fields:
            # Header changed: rewrite the whole CSV with the union of fields.
            self._csv_fields.extend(new_fields)
            rows: list[dict[str, Any]] = []
            if self.csv_path.exists():
                with open(self.csv_path, newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
            rows.append({k: record.get(k) for k in self._csv_fields})
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._csv_fields)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k) for k in self._csv_fields})
        else:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._csv_fields)
                w.writerow({k: record.get(k) for k in self._csv_fields})

    def close(self) -> None:
        try:
            self._jsonl_fp.close()
        except Exception:
            pass
        if self.tb is not None:
            try:
                self.tb.flush()
                self.tb.close()
            except Exception:
                pass
        if self.wandb is not None:
            try:
                self.wandb.finish()
            except Exception:
                pass

def _now() -> float:
    return time.time()

def _jsonable(v):
    try:
        import torch

        if isinstance(v, torch.Tensor):
            return v.item() if v.numel() == 1 else v.tolist()
    except Exception:
        pass
    return str(v)
