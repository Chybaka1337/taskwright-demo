"""Run T1 (online-gaming player segmentation) through the Runtime and export a figure.

    poetry run python -m scripts.run_t1_segmentation
"""
from __future__ import annotations

import joblib
import pandas as pd

import taskwright as tw

from tasks._common import (
    FIGURES_DIR,
    RUNS_DIR,
    plot_clusters_2d,
    reproduce_unsupervised_projection,
    save_unsupervised_material,
)
from tasks.t1_segmentation.online_gaming import OnlineGamingSegmentation, load_events


def main() -> None:
    events = load_events()
    task = OnlineGamingSegmentation()
    result = tw.run(
        task, events, dataset="online-gaming-behavior-synthetic", output_dir=RUNS_DIR
    )

    print(f"events     : {len(events)}  (SYNTHETIC dataset)")
    print(f"category   : {result.category.value}")
    print(f"metrics    : {result.metrics}")
    print(f"artifact   : {result.artifact_path}")

    profile = joblib.load(result.artifact_path)["profile"]
    print("per-cluster feature means:")
    print(pd.DataFrame(profile["feature_means"]).T.to_string())

    labels, projection = reproduce_unsupervised_projection(task, events, result)
    run_dir = result.artifact_path.parent
    material = save_unsupervised_material(run_dir, labels, projection)
    fig = plot_clusters_2d(
        projection, labels,
        "T1 — online-gaming segmentation (k-means, PCA 2D) [synthetic]",
        FIGURES_DIR / "t1_segmentation_clusters.png",
    )

    # Post-hoc sanity: do behavioral clusters line up with the *ignored* target?
    xtab = pd.crosstab(
        pd.Series(labels, name="cluster"),
        pd.Series([e.engagement_level for e in events], name="EngagementLevel(ignored)"),
    )
    print("cluster x EngagementLevel (sanity, target was NOT used to fit):")
    print(xtab.to_string())
    print(f"material   : {material}")
    print(f"figure     : {fig}")


if __name__ == "__main__":
    main()
