"""Read-only access layer for the Streamlit demo (etap 3).

The demo NEVER trains or recomputes: it only reads ``runs/`` produced by etap 2.
``manifest.json`` is the source of truth for metrics; the ``graph_material_*`` files
hold the arrays/series to (re)draw figures. Everything here is plain file reading.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
FIGURES_DIR = REPO_ROOT / "figures"

# task_name -> (Tx code, human title). Fallback below handles anything unknown.
_TASK_LABELS = {
    "Dota2WinPrediction": ("T4", "Dota 2 win prediction"),
    "LoLWinPrediction": ("T4", "LoL win prediction (перенос)"),
    "CookieCatsChurn": ("T2", "Cookie Cats churn"),
    "OnlineGamingSegmentation": ("T1", "Online-gaming segmentation"),
    "SessionRetention": ("T12", "Session retention"),
}

_CONTRACT = {
    "supervised": "SupervisedTask",
    "unsupervised": "UnsupervisedTask",
    "deterministic": "DeterministicTask",
}

# Confusion-matrix class names per supervised run (positive class = index 1).
_CM_LABELS = {
    "Dota2WinPrediction": ["team2 win", "team1 win"],
    "LoLWinPrediction": ["blue loss", "blue win"],
    "CookieCatsChurn": ["retained", "churned"],
}


@dataclass(frozen=True)
class RunInfo:
    run_dir: Path
    run_id: str
    task_name: str
    category: str
    dataset: str
    timestamp: str
    metrics: dict
    code: str
    title: str
    is_synthetic: bool

    @property
    def label(self) -> str:
        return f"{self.code} · {self.title}" if self.code else self.title

    @property
    def contract(self) -> str:
        return _CONTRACT.get(self.category, self.category)

    @property
    def when(self) -> str:
        try:
            return datetime.fromisoformat(self.timestamp).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return self.timestamp or "—"


def discover_runs(runs_dir: Path = RUNS_DIR) -> List[RunInfo]:
    """Scan ``runs/`` for run directories that carry a ``manifest.json``."""
    runs: List[RunInfo] = []
    if not runs_dir.exists():
        return runs
    for d in sorted(runs_dir.iterdir()):
        manifest = d / "manifest.json"
        if not d.is_dir() or not manifest.exists():
            continue  # skips non-dirs like taskwright.jsonl, and incomplete dirs
        try:
            m = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        task_name = m.get("task_name", d.name)
        dataset = m.get("dataset", "") or ""
        code, title = _TASK_LABELS.get(task_name, ("", task_name))
        runs.append(
            RunInfo(
                run_dir=d,
                run_id=d.name,
                task_name=task_name,
                category=m.get("category", ""),
                dataset=dataset,
                timestamp=m.get("timestamp", ""),
                metrics=m.get("metrics", {}) or {},
                code=code,
                title=title,
                is_synthetic="synth" in dataset.lower(),
            )
        )
    return runs


def cm_display_labels(task_name: str) -> List[str]:
    return _CM_LABELS.get(task_name, ["class 0", "class 1"])


# --- material loaders (return None when the file is absent) --------------------
def supervised_material(run: RunInfo) -> Optional[dict]:
    path = run.run_dir / "graph_material_supervised.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return {"y_true": d["y_true"], "y_pred": d["y_pred"], "y_score": d["y_score"]}


def unsupervised_material(run: RunInfo) -> Optional[dict]:
    path = run.run_dir / "graph_material_unsupervised.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return {"labels": d["labels"], "projection": d["projection"]}


def retention_material(run: RunInfo) -> Optional[pd.DataFrame]:
    path = run.run_dir / "graph_material_retention.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# --- overview table ------------------------------------------------------------
def key_metrics_str(run: RunInfo) -> str:
    m = run.metrics

    def f(key: str) -> str:
        v = m.get(key)
        return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"

    if run.category == "supervised":
        return f"f1={f('f1')} · ROC AUC={f('roc_auc')}"
    if run.category == "unsupervised":
        return f"silhouette={f('silhouette')} · DB={f('davies_bouldin')} · k={m.get('n_clusters', '?')}"
    if run.category == "deterministic":
        return f"σκ={m.get('sigma_kappa', '?')} (KPI valid in [0,1])"
    return ", ".join(f"{k}={v}" for k, v in m.items())


def overview_table(runs: List[RunInfo]) -> pd.DataFrame:
    rows = [
        {
            "Задача": run.label + (" · синтетика" if run.is_synthetic else ""),
            "Контракт": run.contract,
            "Датасет": run.dataset,
            "Ключевые метрики": key_metrics_str(run),
            "Дата прогона": run.when,
        }
        for run in runs
    ]
    return pd.DataFrame(rows)
