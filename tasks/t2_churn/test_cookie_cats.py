"""Light smoke test for the T2 Cookie Cats churn contract (small slice for speed)."""
from __future__ import annotations

import taskwright as tw

from tasks.t2_churn.cookie_cats import FEATURE_COLS, CookieCatsChurn, load_events


def test_load_and_label():
    events = load_events(nrows=100)
    assert len(events) == 100
    task = CookieCatsChurn()
    X = task.build_features(events)
    y = task.build_labels(events)
    assert list(X.columns) == FEATURE_COLS
    assert len(X) == len(y) == 100 and X.index.equals(y.index)
    assert set(y.unique()) <= {0, 1}
    # churn == NOT retention_7
    assert y.iloc[0] == int(not events[0].retention_7)


def test_runtime_end_to_end(tmp_path):
    events = load_events(nrows=3000)
    result = tw.run(CookieCatsChurn(), events, dataset="cc-smoke", output_dir=tmp_path)
    assert result.category is tw.TaskCategory.SUPERVISED
    assert {"precision", "recall", "f1", "roc_auc"} <= set(result.metrics)
    assert result.artifact_path.exists()
