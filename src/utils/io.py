"""
io.py
=====
LBSM — File I/O helpers.

Centralises the ``DIRS`` / ``proc(f)`` / ``raw(f)`` / ``figfile(f)`` path
idiom that every notebook (see NB05 cell 2) currently redefines locally, plus
thin save/load wrappers for the three formats used throughout the pipeline
(CSV via pandas, ``.npy`` via numpy, JSON for experiment metadata).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Create ``path`` (and parents) if missing; return it as a :class:`Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class NotebookPaths:
    """Bundles the raw/processed/figures directory trio for one notebook.

    Reproduces the ``DIRS`` dict + ``raw()``/``proc()``/``figfile()`` helper
    pattern used inline in every notebook, as a single reusable object.

    Parameters
    ----------
    nb_id : notebook identifier, e.g. ``"nb05"``.
    root  : repository root (defaults to the current working directory,
            which is correct when notebooks are run from ``notebooks/`` with
            ``sys.path.insert(0, "..")`` as they already do).
    """

    def __init__(self, nb_id: str, root: PathLike = ".") -> None:
        self.nb_id = nb_id
        self.root = Path(root)
        self.raw_dir = ensure_dir(self.root / "data" / "raw" / nb_id)
        self.proc_dir = ensure_dir(self.root / "data" / "processed" / nb_id)
        self.fig_dir = ensure_dir(self.root / "outputs" / "figures" / nb_id)
        self.report_dir = ensure_dir(self.root / "outputs" / "reports" / nb_id)

    def raw(self, filename: str) -> str:
        return str(self.raw_dir / filename)

    def proc(self, filename: str) -> str:
        return str(self.proc_dir / filename)

    def figfile(self, filename: str) -> str:
        return str(self.fig_dir / filename)

    def report(self, filename: str) -> str:
        return str(self.report_dir / filename)


def save_dataframe(df: pd.DataFrame, path: PathLike, index: bool = False) -> None:
    """Save a DataFrame to CSV, creating parent directories as needed."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=index)


def load_dataframe(path: PathLike) -> pd.DataFrame:
    """Load a CSV into a DataFrame."""
    return pd.read_csv(path)


def save_array(arr: np.ndarray, path: PathLike) -> None:
    """Save a NumPy array to ``.npy``, creating parent directories as needed."""
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, arr)


def load_array(path: PathLike) -> np.ndarray:
    """Load a ``.npy`` array."""
    return np.load(path)


def save_json(obj: Dict[str, Any], path: PathLike, indent: int = 2) -> None:
    """Save a JSON-serialisable dict, creating parent directories as needed."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=str)


def load_json(path: PathLike) -> Dict[str, Any]:
    """Load a JSON file into a dict."""
    with open(path) as f:
        return json.load(f)
