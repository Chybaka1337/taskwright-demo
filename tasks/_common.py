"""Shared apробация utilities: paths, reproducible supervised test fold, figures.

The split constants below mirror ``taskwright.runtime.runner`` (the library owns the
split; it is NOT modified here). We re-derive the Runtime's test fold deterministically
so the saved ROC / confusion-matrix material corresponds *exactly* to the metrics in
the returned ``TaskResult`` — and :func:`reproduce_supervised_test` asserts that
correspondence, failing loudly if the split, seed, or persisted model ever diverge.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
RUNS_DIR = REPO_ROOT / "runs"
FIGURES_DIR = REPO_ROOT / "figures"

# Mirror of taskwright.runtime.runner._DEFAULT_TEST_SIZE / _DEFAULT_RANDOM_STATE
# (random train/test split used when time_aware=False). Source of truth = the library.
TEST_SIZE = 0.25
RANDOM_STATE = 0


def reproduce_supervised_test(task: Any, events: Sequence[Any], result: Any):
    """Return ``(y_true, y_pred, y_score)`` for the Runtime's test fold.

    Rebuilds X/y from the same task+events, repeats the Runtime's split, loads the
    persisted fitted model, and asserts the recomputed metric panel equals
    ``result.metrics`` — guaranteeing the figures match the TaskResult.
    """
    X = task.build_features(events)
    y = task.build_labels(events)

    if bool(getattr(task, "time_aware", False)):
        n = len(X)
        n_test = min(max(1, round(n * TEST_SIZE)), n - 1)
        X_test, y_test = X.iloc[-n_test:], y.iloc[-n_test:]
    else:
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
        )

    model = joblib.load(result.artifact_path)
    y_true = y_test.to_numpy()
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    recomputed = {
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }
    for key, val in recomputed.items():
        reported = result.metrics.get(key)
        assert reported is not None and abs(reported - val) < 1e-9, (
            f"Repro mismatch for {key}: TaskResult={reported} vs recomputed={val}. "
            "The reproduced split/seed/model diverged from the Runtime."
        )
    return y_true, y_pred, y_score


def save_supervised_material(run_dir: Path, y_true, y_pred, y_score) -> Path:
    """Co-locate ROC/confusion source arrays next to the run's joblib + manifest."""
    path = Path(run_dir) / "graph_material_supervised.npz"
    np.savez(path, y_true=y_true, y_pred=y_pred, y_score=y_score)
    return path


def plot_roc(y_true, y_score, auc: float, title: str, out_path: Path) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return _finish(fig, out_path)


def plot_confusion(
    y_true,
    y_pred,
    title: str,
    out_path: Path,
    labels: Optional[Sequence[int]] = None,
    display_labels: Optional[Sequence[str]] = None,
) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.set_title(title)
    return _finish(fig, out_path)


def _finish(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
