"""Lightweight experiment tracking.

Every run writes two artefacts:

* ``experiments/runs/<run_id>/run.json`` -- the full record (config, metrics,
  environment) needed to reproduce it;
* one appended row in ``experiments/experiment_log.csv`` -- the flat, committed
  table used for the report.

No external tracking service, so the log stays readable in the repository.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bug2code.logging_utils import get_logger
from bug2code.paths import PROJECT_ROOT, ensure_dir

logger = get_logger(__name__)

LOG_COLUMNS = [
    "run_id",
    "timestamp",
    "task",
    "model",
    "input_variant",
    "dataset",
    "split",
    "seed",
    "hyperparameters",
    "val_metrics",
    "test_metrics",
    "git_commit",
    "notes",
]


def _git_commit() -> str:
    """Return the current commit sha, or ``unknown`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass
class ExperimentRecord:
    """One experiment run.

    ``test_metrics`` must stay empty until a configuration has been selected on
    validation only; test is touched exactly once.
    """

    run_id: str
    task: str
    model: str
    dataset: str
    split: str
    seed: int
    input_variant: str = "n/a"
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_row(self) -> dict[str, str]:
        """Flatten to the CSV schema, JSON-encoding the nested fields."""
        return {
            "run_id": self.run_id,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "task": self.task,
            "model": self.model,
            "input_variant": self.input_variant,
            "dataset": self.dataset,
            "split": self.split,
            "seed": str(self.seed),
            "hyperparameters": json.dumps(self.hyperparameters, sort_keys=True),
            "val_metrics": json.dumps(self.val_metrics, sort_keys=True),
            "test_metrics": json.dumps(self.test_metrics, sort_keys=True),
            "git_commit": _git_commit(),
            "notes": self.notes,
        }


def make_run_id(task: str, model: str, suffix: str = "") -> str:
    """Build a sortable, human-readable run id."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    parts = [stamp, task, model]
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


def log_experiment(
    record: ExperimentRecord,
    experiments_dir: Path,
    artefacts: dict[str, Any] | None = None,
) -> Path:
    """Append ``record`` to the CSV log and write its per-run JSON.

    Args:
        record: The run to persist.
        experiments_dir: Root experiments directory from the config.
        artefacts: Extra JSON-serialisable detail (per-class scores, curves, ...).

    Returns:
        The per-run directory.
    """
    run_dir = ensure_dir(experiments_dir / "runs" / record.run_id)

    payload: dict[str, Any] = {
        **record.to_row(),
        "hyperparameters": record.hyperparameters,
        "val_metrics": record.val_metrics,
        "test_metrics": record.test_metrics,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "artefacts": artefacts or {},
    }
    (run_dir / "run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = ensure_dir(experiments_dir) / "experiment_log.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(record.to_row())

    logger.info("logged run %s -> %s", record.run_id, run_dir)
    return run_dir
