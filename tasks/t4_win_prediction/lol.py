"""T4 transfer — win prediction on 'League of Legends Diamond Ranked 10min' (2nd game).

Same ``taskwright.SupervisedTask`` contract as the Dota 2 reference, pointed at a
different game's snapshot — this is the portability evidence (the ВКР goal): the
framework runs unchanged across games, only the task implementation differs.

Real schema (read from the file; 9879 rows × 40 cols, no NaN):
  gameId   = unique match id (dropped, not a feature)
  blueWins = target (1 if the blue team won), balanced ~50/50
  38 numeric blue*/red* match-state stats at the 10-minute mark = features
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Literal

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import taskwright as tw

from .._common import RAW_DIR

LOL_CSV = RAW_DIR / "high_diamond_ranked_10min.csv"
_ID_COL = "gameId"
_LABEL_COL = "blueWins"
_SNAPSHOT_T0 = datetime(2020, 1, 1)  # synthetic base; the snapshot has no real time axis


class LoLMatch10min(tw.BaseEvent):
    """One ranked LoL match, summarized at the 10-minute mark.

    Static snapshot: ``user_id``/``event_timestamp`` are synthesized (no player, no
    time axis) and unused as features (``time_aware=False``).
    """

    event_name: Literal["lol_match_10min"] = "lol_match_10min"
    blue_wins: int
    features: Dict[str, float]


def load_events(path=LOL_CSV, nrows: int | None = None) -> List[LoLMatch10min]:
    """Map each CSV row to one :class:`LoLMatch10min` event (row order preserved)."""
    df = pd.read_csv(path, nrows=nrows)
    feature_cols = [c for c in df.columns if c not in (_ID_COL, _LABEL_COL)]
    ids = df[_ID_COL].to_numpy()
    labels = df[_LABEL_COL].to_numpy()
    records = df[feature_cols].to_dict("records")  # one dict per row, column order kept
    events: List[LoLMatch10min] = []
    for i, rec in enumerate(records):
        events.append(
            LoLMatch10min(
                user_id=f"lol-{ids[i]}",
                event_timestamp=_SNAPSHOT_T0 + timedelta(seconds=i),
                event_id=f"lol-{ids[i]}",
                blue_wins=int(labels[i]),
                features={k: float(v) for k, v in rec.items()},
            )
        )
    return events


class LoLWinPrediction(tw.SupervisedTask):
    """Predict whether the blue team wins from the 10-minute match state."""

    time_aware = False  # static snapshot, no temporal order -> random split

    def build_features(self, events: Iterable[tw.BaseEvent]) -> pd.DataFrame:
        return pd.DataFrame([e.features for e in events])

    def build_labels(self, events: Iterable[tw.BaseEvent]) -> pd.Series:
        return pd.Series([e.blue_wins for e in events], name="blue_wins")

    def build_model(self):
        return HistGradientBoostingClassifier(random_state=0)
