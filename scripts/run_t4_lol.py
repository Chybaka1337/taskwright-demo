"""Run the T4 transfer (LoL win prediction) through the Runtime and export figures.

    poetry run python -m scripts.run_t4_lol
"""
from __future__ import annotations

import taskwright as tw

from tasks._common import (
    FIGURES_DIR,
    RUNS_DIR,
    plot_confusion,
    plot_roc,
    reproduce_supervised_test,
    save_supervised_material,
)
from tasks.t4_win_prediction.lol import LoLWinPrediction, load_events


def main() -> None:
    events = load_events()
    task = LoLWinPrediction()
    result = tw.run(task, events, dataset="lol-diamond-10min", output_dir=RUNS_DIR)

    print(f"events     : {len(events)}")
    print(f"category   : {result.category.value}")
    print(f"metrics    : {result.metrics}")
    print(f"artifact   : {result.artifact_path}")

    y_true, y_pred, y_score = reproduce_supervised_test(task, events, result)
    run_dir = result.artifact_path.parent
    material = save_supervised_material(run_dir, y_true, y_pred, y_score)

    roc_png = plot_roc(
        y_true, y_score, result.metrics["roc_auc"],
        "T4 transfer — LoL win prediction (ROC)", FIGURES_DIR / "t4_lol_roc.png",
    )
    cm_png = plot_confusion(
        y_true, y_pred, "T4 transfer — LoL win prediction (confusion)",
        FIGURES_DIR / "t4_lol_confusion.png",
        labels=[0, 1], display_labels=["blue loss", "blue win"],
    )

    acc = float((y_true == y_pred).mean())
    print(f"test fold  : n={len(y_true)} | accuracy={acc:.4f} (derived from reproduced split)")
    print(f"material   : {material}")
    print(f"figures    : {roc_png} | {cm_png}")


if __name__ == "__main__":
    main()
