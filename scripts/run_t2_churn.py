"""Run T2 (Cookie Cats churn) through the Runtime and export figures.

    poetry run python -m scripts.run_t2_churn
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
from tasks.t2_churn.cookie_cats import CookieCatsChurn, load_events


def main() -> None:
    events = load_events()
    task = CookieCatsChurn()
    result = tw.run(task, events, dataset="cookie-cats", output_dir=RUNS_DIR)

    print(f"events     : {len(events)}")
    print(f"category   : {result.category.value}")
    print(f"metrics    : {result.metrics}")
    print(f"artifact   : {result.artifact_path}")

    y_true, y_pred, y_score = reproduce_supervised_test(task, events, result)
    run_dir = result.artifact_path.parent
    material = save_supervised_material(run_dir, y_true, y_pred, y_score)

    roc_png = plot_roc(
        y_true, y_score, result.metrics["roc_auc"],
        "T2 — Cookie Cats churn (ROC)", FIGURES_DIR / "t2_churn_roc.png",
    )
    cm_png = plot_confusion(
        y_true, y_pred, "T2 — Cookie Cats churn (confusion)",
        FIGURES_DIR / "t2_churn_confusion.png",
        labels=[0, 1], display_labels=["retained", "churned"],
    )

    acc = float((y_true == y_pred).mean())
    base = float(y_true.mean())
    print(f"test fold  : n={len(y_true)} | accuracy={acc:.4f} | churn base rate={base:.4f}")
    print(f"material   : {material}")
    print(f"figures    : {roc_png} | {cm_png}")


if __name__ == "__main__":
    main()
