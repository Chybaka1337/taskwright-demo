"""T12 — retention / KPI on the synthetic session stream.

Reference ``taskwright.DeterministicTask``: no model, no split — the Runtime dedups by
``event_id``, filters to the time window, calls :meth:`aggregate`, then checks the KPI
with :meth:`is_valid`. Each CSV row is already an event (session begin/end).

Real schema (read from the file; 36630 rows × 7 cols, all str, no NaN; 3916 players;
span 2024-01-01 → 2024-02-18). Field -> BaseEvent mapping:
  player_id -> user_id,  timestamp -> event_timestamp,  id -> event_id,
  event_type -> event_name (BEGIN_SESSION | END_SESSION).
``player_type`` (casual | churner | hardcore) is the generator's ground-truth label,
kept on the event so retention can be validated per segment (churners drop fast).

KPI: classic day-N retention with a censoring-aware denominator — for offset N,
retention[N] = (#players active on first_day+N) / (#players whose first_day+N is still
within the observation window). Day 0 is 1.0 by construction.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Iterable, List, Literal, Optional

import pandas as pd

import taskwright as tw

from .._common import RAW_DIR

SESSIONS_CSV = RAW_DIR / "sessions_synth.csv"
HORIZON_DAYS = 30


class GameSessionEvent(tw.BaseEvent):
    """One session begin/end event for a synthetic player."""

    event_name: Literal["BEGIN_SESSION", "END_SESSION"]
    player_type: str  # generator ground truth: casual | churner | hardcore
    cohort_id: str
    session_id: str


def load_events(path=SESSIONS_CSV, nrows: int | None = None) -> List[GameSessionEvent]:
    """Map each CSV row to one :class:`GameSessionEvent` (timestamps parsed explicitly)."""
    df = pd.read_csv(path, nrows=nrows)
    ts = pd.to_datetime(df["timestamp"])
    records = df.to_dict("records")
    events: List[GameSessionEvent] = []
    for i, rec in enumerate(records):
        events.append(
            GameSessionEvent(
                user_id=str(rec["player_id"]),
                event_timestamp=ts.iloc[i].to_pydatetime(),
                event_id=str(rec["id"]),
                event_name=str(rec["event_type"]),
                player_type=str(rec["player_type"]),
                cohort_id=str(rec["cohort_id"]),
                session_id=str(rec["session_id"]),
            )
        )
    return events


def day_n_retention(events: Iterable[tw.BaseEvent], max_day: date, horizon: int = HORIZON_DAYS) -> pd.Series:
    """Censoring-aware day-N retention over ``events``, for N in ``0..horizon``.

    A player is counted at offset N only if ``first_day + N`` is within the observation
    window (``<= max_day``); they are 'retained' there if they have any event that day.
    """
    active: dict[str, set] = defaultdict(set)
    first: dict[str, date] = {}
    for e in events:
        day = e.event_timestamp.date()
        player = e.user_id
        active[player].add(day)
        if player not in first or day < first[player]:
            first[player] = day

    data: dict[int, float] = {}
    for n in range(horizon + 1):
        eligible = retained = 0
        for player, first_day in first.items():
            target = first_day + timedelta(days=n)
            if target <= max_day:
                eligible += 1
                if target in active[player]:
                    retained += 1
        if eligible > 0:
            data[n] = retained / eligible
    return pd.Series(data, name="retention").rename_axis("day_offset")


class SessionRetention(tw.DeterministicTask):
    """Day-N retention curve over the session stream (the KPI is a pandas Series)."""

    def aggregate(self, events: Iterable[tw.BaseEvent], window: Optional[tuple]) -> pd.Series:
        events = list(events)
        if window is not None:
            max_day = window[1].date()
        else:
            max_day = max(e.event_timestamp.date() for e in events)
        return day_n_retention(events, max_day, HORIZON_DAYS)

    def is_valid(self, kpi) -> bool:
        """Narrow the admissible region to a finite retention series in [0, 1]."""
        if not isinstance(kpi, pd.Series) or kpi.empty:
            return False
        return bool(kpi.notna().all()) and bool(((kpi >= 0.0) & (kpi <= 1.0)).all())
