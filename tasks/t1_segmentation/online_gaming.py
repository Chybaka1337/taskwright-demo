"""T1 — player segmentation on the 'Predict Online Gaming Behavior' dataset.

Reference ``taskwright.UnsupervisedTask`` (k-means). **SYNTHETIC dataset**: the rows
are generator output, so any segments found are structure of the generator, not real
player behavior — reported as such, not as a substantive finding.

Real schema (read from the file; 40034 rows × 13 cols, no NaN). We cluster on the five
continuous *behavioral* features only; the provided ``EngagementLevel`` label is
**ignored** for fitting (kept on the event purely as a post-hoc sanity reference):
  behavioral -> PlayTimeHours, SessionsPerWeek, AvgSessionDurationMinutes,
                PlayerLevel, AchievementsUnlocked
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, List, Literal

import pandas as pd
from sklearn.cluster import KMeans

import taskwright as tw

from .._common import RAW_DIR

ONLINE_GAMING_CSV = RAW_DIR / "online_gaming_behavior_dataset.csv"
_SNAPSHOT_T0 = datetime(2024, 1, 1)  # synthetic base; data has no real time axis
N_CLUSTERS = 3  # matches the 3 EngagementLevel tiers, enabling a post-hoc cross-tab

FEATURE_COLS = [
    "play_time_hours",
    "sessions_per_week",
    "avg_session_duration_min",
    "player_level",
    "achievements_unlocked",
]


class OnlineGamingPlayer(tw.BaseEvent):
    """One synthetic player profile."""

    event_name: Literal["online_gaming_profile"] = "online_gaming_profile"
    age: int
    gender: str
    location: str
    game_genre: str
    game_difficulty: str
    play_time_hours: float
    in_game_purchases: int
    sessions_per_week: int
    avg_session_duration_min: int
    player_level: int
    achievements_unlocked: int
    engagement_level: str  # synthetic target; NOT used for clustering


def load_events(path=ONLINE_GAMING_CSV, nrows: int | None = None) -> List[OnlineGamingPlayer]:
    """Map each CSV row to one :class:`OnlineGamingPlayer` event (row order preserved)."""
    df = pd.read_csv(path, nrows=nrows)
    events: List[OnlineGamingPlayer] = []
    for i, r in enumerate(df.itertuples(index=False)):
        events.append(
            OnlineGamingPlayer(
                user_id=f"og-{r.PlayerID}",
                event_timestamp=_SNAPSHOT_T0 + timedelta(seconds=i),
                event_id=f"og-{r.PlayerID}",
                age=int(r.Age),
                gender=str(r.Gender),
                location=str(r.Location),
                game_genre=str(r.GameGenre),
                game_difficulty=str(r.GameDifficulty),
                play_time_hours=float(r.PlayTimeHours),
                in_game_purchases=int(r.InGamePurchases),
                sessions_per_week=int(r.SessionsPerWeek),
                avg_session_duration_min=int(r.AvgSessionDurationMinutes),
                player_level=int(r.PlayerLevel),
                achievements_unlocked=int(r.AchievementsUnlocked),
                engagement_level=str(r.EngagementLevel),
            )
        )
    return events


class OnlineGamingSegmentation(tw.UnsupervisedTask):
    """Segment players by behavioral engagement/progression features (k-means)."""

    # scaler left at the contract default (StandardScaler) -> Runtime normalizes.

    def build_features(self, events: Iterable[tw.BaseEvent]) -> pd.DataFrame:
        events = list(events)
        return pd.DataFrame(
            {
                "play_time_hours": [e.play_time_hours for e in events],
                "sessions_per_week": [e.sessions_per_week for e in events],
                "avg_session_duration_min": [e.avg_session_duration_min for e in events],
                "player_level": [e.player_level for e in events],
                "achievements_unlocked": [e.achievements_unlocked for e in events],
            }
        )

    def build_model(self):
        return KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=0)

    def interpret(self, clusters, X: pd.DataFrame) -> dict:
        """Per-cluster sizes + mean of each (original, unscaled) behavioral feature."""
        df = X.copy()
        df["cluster"] = clusters
        sizes = df["cluster"].value_counts().sort_index()
        means = df.groupby("cluster").mean(numeric_only=True)
        return {
            "sizes": {int(k): int(v) for k, v in sizes.items()},
            "feature_means": means.round(3).to_dict(orient="index"),
        }
