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
from sklearn.decomposition import PCA  # noqa: E402
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


def reproduce_unsupervised_projection(task: Any, events: Sequence[Any], result: Any):
    """Return ``(labels, projection_2d)`` consistent with the Runtime's clustering.

    Uses the Runtime's exact cluster assignments (from the persisted payload) and
    re-applies the same normalization (``task.scaler``, mirroring
    ``taskwright.runtime.runner._normalize``) before a 2D PCA, so the scatter shows
    precisely what the Runtime clustered / scored.
    """
    X = task.build_features(events)
    payload = joblib.load(result.artifact_path)
    labels = np.asarray(payload["labels"])

    scaler_factory = getattr(task, "scaler", None)
    if scaler_factory is None:
        X_norm = X.to_numpy()
    else:
        scaler = scaler_factory() if isinstance(scaler_factory, type) else scaler_factory
        X_norm = scaler.fit_transform(X)

    projection = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_norm)
    return labels, projection


def save_unsupervised_material(run_dir: Path, labels, projection) -> Path:
    """Co-locate cluster assignments + 2D projection next to the run's joblib payload."""
    path = Path(run_dir) / "graph_material_unsupervised.npz"
    np.savez(path, labels=np.asarray(labels), projection=np.asarray(projection))
    return path


def plot_clusters_2d(
    projection, labels, title: str, out_path: Path, xlabel: str = "PC1", ylabel: str = "PC2"
) -> Path:
    projection = np.asarray(projection)
    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in sorted(set(labels.tolist())):
        mask = labels == c
        ax.scatter(
            projection[mask, 0], projection[mask, 1], s=5, alpha=0.35,
            label=f"cluster {c} (n={int(mask.sum())})",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", markerscale=2, framealpha=0.9)
    return _finish(fig, out_path)


def plot_retention_curves(curves: dict, title: str, out_path: Path) -> Path:
    """Plot one or more day-N retention curves (label -> pandas Series indexed by day)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label, series in curves.items():
        ax.plot(
            series.index.to_numpy(), series.to_numpy(),
            marker="o", ms=3, lw=1.5, label=label,
        )
    ax.set_xlabel("days since first appearance")
    ax.set_ylabel("retention (share of eligible players)")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return _finish(fig, out_path)


def _finish(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
