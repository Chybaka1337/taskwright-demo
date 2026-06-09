"""T4 — win prediction on the UCI 'Dota 2 Games Results' snapshot (main reference).

Reference implementation of the ``taskwright.SupervisedTask`` contract. The dataset
ships pre-split (dota2Train.csv / dota2Test.csv); per the apробация plan we feed the
TRAIN file as the event stream and let the Runtime do its own random train/test split
(time_aware=False) — dota2Test.csv is not used (no double split).

Real schema (read from the file, headerless, all int64, no NaN):
  col0  = label: 1 if team "1" won, -1 otherwise
  col1  = cluster id (region), col2 = game mode, col3 = game type
  col4..col116 = 113 hero indicators in {-1, 0, 1} (1=on team1, -1=on team2, 0=absent)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, List, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import taskwright as tw

from .._common import RAW_DIR

DOTA2_TRAIN_CSV = RAW_DIR / "dota2Train.csv"
N_HEROES = 113
_SNAPSHOT_T0 = datetime(2016, 1, 1)  # synthetic base; the snapshot has no real time axis

HERO_COLS = [f"hero_{k}" for k in range(N_HEROES)]
META_COLS = ["cluster_id", "game_mode", "game_type"]
FEATURE_COLS = META_COLS + HERO_COLS


class Dota2MatchResult(tw.BaseEvent):
    """One finished Dota 2 match.

    Static snapshot: there is no player identity and no event time. ``user_id`` and
    ``event_timestamp`` are synthesized from the row index solely to satisfy the
    telemetry base — they carry no signal and are not used as features (the task is
    ``time_aware=False``, so the Runtime never touches the time axis).
    """

    event_name: Literal["dota2_match"] = "dota2_match"
    cluster_id: int
    game_mode: int
    game_type: int
    picks: List[int]  # 113 hero indicators in {-1, 0, 1}, in column order
    team1_won: bool  # raw UCI label (column 0) == 1


def load_events(path=DOTA2_TRAIN_CSV, nrows: int | None = None) -> List[Dota2MatchResult]:
    """Map each CSV row to one :class:`Dota2MatchResult` event (row order preserved).

    ``nrows`` caps how many rows are read (used by the fast smoke test); ``None``
    reads the whole file.
    """
    df = pd.read_csv(path, header=None, nrows=nrows)
    label = df.iloc[:, 0].to_numpy()
    cluster = df.iloc[:, 1].to_numpy()
    mode = df.iloc[:, 2].to_numpy()
    gtype = df.iloc[:, 3].to_numpy()
    picks = df.iloc[:, 4:].to_numpy()
    events: List[Dota2MatchResult] = []
    for i in range(len(df)):
        events.append(
            Dota2MatchResult(
                user_id=f"dota2-train-{i}",
                event_timestamp=_SNAPSHOT_T0 + timedelta(seconds=i),
                event_id=f"dota2-train-{i}",
                cluster_id=int(cluster[i]),
                game_mode=int(mode[i]),
                game_type=int(gtype[i]),
                picks=[int(v) for v in picks[i]],
                team1_won=bool(label[i] == 1),
            )
        )
    return events


class Dota2WinPrediction(tw.SupervisedTask):
    """Predict which team wins from the draft (hero indicators + match metadata)."""

    time_aware = False  # static snapshot, no temporal order -> random split

    def build_features(self, events: Iterable[tw.BaseEvent]) -> pd.DataFrame:
        events = list(events)
        meta = np.array(
            [[e.cluster_id, e.game_mode, e.game_type] for e in events], dtype=float
        )
        picks = np.array([e.picks for e in events], dtype=float)
        return pd.DataFrame(np.hstack([meta, picks]), columns=FEATURE_COLS)

    def build_labels(self, events: Iterable[tw.BaseEvent]) -> pd.Series:
        return pd.Series([int(e.team1_won) for e in events], name="team1_won")

    def build_model(self):
        return HistGradientBoostingClassifier(random_state=0)
