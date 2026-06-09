"""Run T12 (session retention) through the Runtime and export the retention curve.

    poetry run python -m scripts.run_t12_retention
"""
from __future__ import annotations

import joblib
import pandas as pd

import taskwright as tw

from tasks._common import FIGURES_DIR, RUNS_DIR, plot_retention_curves
from tasks.t12_retention.sessions import SessionRetention, day_n_retention, load_events

_PLAYER_TYPES = ["churner", "casual", "hardcore"]


def main() -> None:
    events = load_events()
    task = SessionRetention()
    window = (
        min(e.event_timestamp for e in events),
        max(e.event_timestamp for e in events),
    )
    result = tw.run(
        task, events, dataset="sessions-synth", window=window, output_dir=RUNS_DIR
    )

    print(f"events     : {len(events)}  | players: {len({e.user_id for e in events})}")
    print(f"category   : {result.category.value}")
    print(f"metrics    : {result.metrics}  (sigma_kappa=1 => KPI passed is_valid in [0,1])")
    print(f"artifact   : {result.artifact_path}")

    overall = joblib.load(result.artifact_path)  # the persisted KPI (retention Series)
    max_day = window[1].date()

    # Per-segment validation against the generator's ground-truth player_type.
    curves = {"overall": overall}
    for t in _PLAYER_TYPES:
        subset = [e for e in events if e.player_type == t]
        curves[t] = day_n_retention(subset, max_day)

    material = result.artifact_path.parent / "graph_material_retention.csv"
    pd.DataFrame(curves).to_csv(material, index_label="day_offset")

    fig = plot_retention_curves(
        curves, "T12 — day-N retention (synthetic sessions)",
        FIGURES_DIR / "t12_retention_curve.png",
    )

    def at(series, n):
        return f"{series.get(n, float('nan')):.3f}" if n in series.index else "n/a"

    print("retention @ day 1 / 7 / 30:")
    print(f"  overall : {at(overall,1)} / {at(overall,7)} / {at(overall,30)}")
    for t in _PLAYER_TYPES:
        s = curves[t]
        print(f"  {t:8s}: {at(s,1)} / {at(s,7)} / {at(s,30)}")
    print(f"material   : {material}")
    print(f"figure     : {fig}")


if __name__ == "__main__":
    main()
