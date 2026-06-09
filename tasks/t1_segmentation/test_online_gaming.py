"""Light smoke test for the T1 segmentation contract (small slice for speed)."""
from __future__ import annotations

import taskwright as tw

from tasks.t1_segmentation.online_gaming import (
    FEATURE_COLS,
    N_CLUSTERS,
    OnlineGamingSegmentation,
    load_events,
)


def test_features_exclude_target():
    events = load_events(nrows=300)
    task = OnlineGamingSegmentation()
    X = task.build_features(events)
    assert list(X.columns) == FEATURE_COLS
    assert "engagement_level" not in X.columns  # target must not leak into features
    assert len(X) == 300


def test_runtime_end_to_end(tmp_path):
    events = load_events(nrows=1500)
    result = tw.run(
        OnlineGamingSegmentation(), events, dataset="og-smoke", output_dir=tmp_path
    )
    assert result.category is tw.TaskCategory.UNSUPERVISED
    assert result.metrics["n_clusters"] == N_CLUSTERS
    assert {"silhouette", "davies_bouldin"} <= set(result.metrics)
    payload = __import__("joblib").load(result.artifact_path)
    assert set(payload) >= {"model", "labels", "profile"}
    assert len(payload["labels"]) == 1500
