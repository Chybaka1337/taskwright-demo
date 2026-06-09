"""Light smoke test for the T4 LoL transfer contract (small slice for speed)."""
from __future__ import annotations

import taskwright as tw

from tasks.t4_win_prediction.lol import LoLMatch10min, LoLWinPrediction, load_events


def test_load_events_schema():
    events = load_events(nrows=50)
    assert len(events) == 50
    e = events[0]
    assert isinstance(e, LoLMatch10min)
    assert len(e.features) == 38  # 40 cols - gameId - blueWins
    assert e.blue_wins in (0, 1)
    assert "blueGoldDiff" in e.features


def test_runtime_end_to_end(tmp_path):
    events = load_events(nrows=2000)
    task = LoLWinPrediction()
    X, y = task.build_features(events), task.build_labels(events)
    assert len(X) == len(y) == 2000 and X.index.equals(y.index)

    result = tw.run(task, events, dataset="lol-smoke", output_dir=tmp_path)
    assert result.category is tw.TaskCategory.SUPERVISED
    assert {"precision", "recall", "f1", "roc_auc"} <= set(result.metrics)
    assert result.artifact_path.exists()
