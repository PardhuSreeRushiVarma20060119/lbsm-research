"""
experiment_tracking.py
=======================
LBSM — Lightweight experiment run tracking.

Records what an experiment script (``experiments/*/run_*.py``) did: when it
ran, against which config, which artefacts it produced, and whether it
succeeded — as a single JSON record per run. Deliberately minimal (no
database, no server): this is a research pipeline with a handful of
experiment stages, not an MLOps deployment.
"""

from __future__ import annotations

import subprocess
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from .io import PathLike, save_json


def _git_commit() -> Optional[str]:
    """Best-effort current git commit hash; None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return None


@dataclass
class ExperimentRecord:
    """Metadata for a single experiment run.

    Attributes
    ----------
    name        : experiment identifier, e.g. ``"rl_adaptive"``.
    config      : the configuration dict the run was executed with.
    started_at  : ISO-8601 UTC timestamp, set on construction.
    finished_at : ISO-8601 UTC timestamp, set by :meth:`finish`.
    status      : ``"running"`` | ``"completed"`` | ``"failed"``.
    git_commit  : current HEAD hash, if run inside a git checkout.
    outputs     : paths to artefacts the run produced.
    error       : traceback string, populated only if the run failed.
    """
    name: str
    config: Dict[str, Any]
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = None
    status: str = "running"
    git_commit: Optional[str] = field(default_factory=_git_commit)
    outputs: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def add_output(self, path: PathLike) -> None:
        """Register an output artefact path produced by this run."""
        self.outputs.append(str(path))

    def finish(self, status: str = "completed") -> None:
        """Mark the run finished with the given status."""
        self.status = status
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "config": self.config,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "git_commit": self.git_commit,
            "outputs": self.outputs,
            "error": self.error,
        }

    def save(self, path: PathLike) -> None:
        """Persist this record as JSON."""
        save_json(self.to_dict(), path)


@contextmanager
def track_run(
    name: str,
    config: Dict[str, Any],
    record_path: Optional[PathLike] = None,
) -> Iterator[ExperimentRecord]:
    """Context manager that tracks one experiment run end-to-end.

    On a clean exit, marks the record ``completed``; on an exception, marks
    it ``failed`` with the traceback attached, then re-raises. If
    ``record_path`` is given, the record is saved as JSON on either path.

    Example
    -------
    >>> with track_run("rl_adaptive", cfg, "outputs/logs/rl_adaptive_run.json") as run:
    ...     run.add_output("data/raw/nb05/q_tables.npy")
    ...     # ... do the work ...
    """
    record = ExperimentRecord(name=name, config=config)
    try:
        yield record
        record.finish("completed")
    except Exception:
        record.error = traceback.format_exc()
        record.finish("failed")
        raise
    finally:
        if record_path is not None:
            record.save(record_path)
