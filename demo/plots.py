"""Matplotlib figure builders for the demo — rebuild figures from saved material.

Each function returns a ``Figure`` for ``st.pyplot``. Reported metrics come from
``manifest.json`` (source of truth); these helpers only turn saved arrays/series into
plots (``roc_curve`` / ``confusion_matrix`` tabulate the saved test predictions — they
do not re-run any model).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
)


def roc_fig(y_true, y_score, auc: float, title: str):
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
    fig.tight_layout()
    return fig


def confusion_fig(y_true, y_pred, title: str, display_labels):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def clusters_fig(projection, labels, title: str):
    projection = np.asarray(projection)
    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in sorted(set(labels.tolist())):
        mask = labels == c
        ax.scatter(
            projection[mask, 0], projection[mask, 1], s=5, alpha=0.35,
            label=f"cluster {c} (n={int(mask.sum())})",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(loc="best", markerscale=2, framealpha=0.9)
    fig.tight_layout()
    return fig


def retention_fig(df, title: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = df["day_offset"].to_numpy()
    for col in [c for c in df.columns if c != "day_offset"]:
        ax.plot(x, df[col].to_numpy(), marker="o", ms=3, lw=1.5, label=col)
    ax.set_xlabel("days since first appearance")
    ax.set_ylabel("retention (share of eligible players)")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
