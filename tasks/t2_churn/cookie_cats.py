"""T2 — churn prediction on the Cookie Cats A/B dataset.

Reference ``taskwright.SupervisedTask``. Each row is one player's first-14-day summary
(real ``userid``), so unlike the Dota2/LoL snapshots ``user_id`` is meaningful here;
there is still no event time, so the task is ``time_aware=False`` (cross-sectional
random split).

Real schema (read from the file; 90189 rows × 5 cols, no NaN):
  userid          = player id (-> user_id / event_id)
  version         = A/B group, 'gate_30' | 'gate_40'
  sum_gamerounds  = rounds played in the first 14 days
  retention_1     = returned 1 day after install (bool)
  retention_7     = returned 7 days after install (bool)  -> churn label source

Churn label := NOT retention_7 (did not come back on day 7).

Note: ``sum_gamerounds`` is measured over 14 days, which overlaps the 7-day label
horizon, so it carries mild label leakage by construction. We keep it because it is the
canonical Cookie Cats feature and this is a framework apробация, not a production churn
model — the caveat is disclosed rather than hidden.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, List, Literal

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import taskwright as tw

from .._common import RAW_DIR

COOKIE_CATS_CSV = RAW_DIR / "cookie_cats.csv"
_SNAPSHOT_T0 = datetime(2018, 1, 1)  # synthetic base; data has no real install timestamp

FEATURE_COLS = ["is_gate_40", "sum_gamerounds", "retention_1"]


class CookieCatsPlayer(tw.BaseEvent):
    """One Cookie Cats player's first-14-day summary."""

    event_name: Literal["cookie_cats_player"] = "cookie_cats_player"
    version: str
    sum_gamerounds: int
    retention_1: bool
    retention_7: bool  # label source; mapped to churn in build_labels


def load_events(path=COOKIE_CATS_CSV, nrows: int | None = None) -> List[CookieCatsPlayer]:
    """Map each CSV row to one :class:`CookieCatsPlayer` event (row order preserved)."""
    df = pd.read_csv(path, nrows=nrows)
    events: List[CookieCatsPlayer] = []
    for i, row in enumerate(df.itertuples(index=False)):
        events.append(
            CookieCatsPlayer(
                user_id=f"cc-{row.userid}",
                event_timestamp=_SNAPSHOT_T0 + timedelta(seconds=i),
                event_id=f"cc-{row.userid}",
                version=str(row.version),
                sum_gamerounds=int(row.sum_gamerounds),
                retention_1=bool(row.retention_1),
                retention_7=bool(row.retention_7),
            )
        )
    return events


class CookieCatsChurn(tw.SupervisedTask):
    """Predict 7-day churn (did not return on day 7) from early-engagement features."""

    time_aware = False  # cross-sectional snapshot, no temporal order -> random split

    def build_features(self, events: Iterable[tw.BaseEvent]) -> pd.DataFrame:
        events = list(events)
        return pd.DataFrame(
            {
                "is_gate_40": [1 if e.version == "gate_40" else 0 for e in events],
                "sum_gamerounds": [float(e.sum_gamerounds) for e in events],
                "retention_1": [int(e.retention_1) for e in events],
            }
        )

    def build_labels(self, events: Iterable[tw.BaseEvent]) -> pd.Series:
        return pd.Series([int(not e.retention_7) for e in events], name="churn")

    def build_model(self):
        return HistGradientBoostingClassifier(random_state=0)
