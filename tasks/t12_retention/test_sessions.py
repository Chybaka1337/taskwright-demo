"""Light smoke test for the T12 retention contract."""
from __future__ import annotations

import pandas as pd

import taskwright as tw

from tasks.t12_retention.sessions import SessionRetention, day_n_retention, load_events


def test_load_and_mapping():
    events = load_events(nrows=200)
    assert len(events) == 200
    e = events[0]
    assert e.event_name in ("BEGIN_SESSION", "END_SESSION")
    assert e.player_type in ("casual", "churner", "hardcore")


def test_retention_day0_is_one():
    events = load_events(nrows=2000)
    max_day = max(e.event_timestamp.date() for e in events)
    s = day_n_retention(events, max_day)
    assert s.loc[0] == 1.0  # everyone present on their own first day
    assert ((s >= 0.0) & (s <= 1.0)).all()


def test_runtime_end_to_end(tmp_path):
    events = load_events()
    window = (
        min(e.event_timestamp for e in events),
        max(e.event_timestamp for e in events),
    )
    result = tw.run(
        SessionRetention(), events, dataset="t12-smoke", window=window, output_dir=tmp_path
    )
    assert result.category is tw.TaskCategory.DETERMINISTIC
    assert result.metrics["sigma_kappa"] == 1
    kpi = __import__("joblib").load(result.artifact_path)
    assert isinstance(kpi, pd.Series) and kpi.loc[0] == 1.0
