"""Light smoke test for the T4 Dota 2 contract (runs on a small slice for speed)."""
from __future__ import annotations

import taskwright as tw

from tasks.t4_win_prediction.dota2 import (
    FEATURE_COLS,
    Dota2MatchResult,
    Dota2WinPrediction,
    load_events,
)


def test_load_events_schema():
    events = load_events(nrows=50)
    assert len(events) == 50
    e = events[0]
    assert isinstance(e, Dota2MatchResult)
    assert len(e.picks) == 113
    assert set(e.picks) <= {-1, 0, 1}
    assert e.picks.count(1) == 5 and e.picks.count(-1) == 5  # classic 5v5 draft
    assert isinstance(e.team1_won, bool)


def test_features_labels_aligned():
    events = load_events(nrows=200)
    task = Dota2WinPrediction()
    X = task.build_features(events)
    y = task.build_labels(events)
    assert list(X.columns) == FEATURE_COLS
    assert len(X) == len(y) == 200
    assert X.index.equals(y.index)
    assert set(y.unique()) <= {0, 1}


def test_runtime_end_to_end(tmp_path):
    events = load_events(nrows=2000)
    result = tw.run(
        Dota2WinPrediction(), events, dataset="dota2-smoke", output_dir=tmp_path
    )
    assert result.category is tw.TaskCategory.SUPERVISED
    assert {"precision", "recall", "f1", "roc_auc"} <= set(result.metrics)
    assert result.artifact_path.exists()
