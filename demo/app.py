"""Streamlit demo over the taskwright apробация runs (etap 3) — VISUALIZATION ONLY.

    poetry run streamlit run demo/app.py

Reads ``runs/`` (read-only): metrics come from each run's ``manifest.json`` (= the
TaskResult), figures are rebuilt from the saved ``graph_material_*`` files. Nothing is
trained or recomputed; taskwright and the task implementations are not touched.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

import plots  # noqa: E402  (sibling module; demo/ is on sys.path under `streamlit run`)
import runs_loader as rl  # noqa: E402

st.set_page_config(page_title="taskwright — апробация", layout="wide")


def _fmt(value, nd: int = 3) -> str:
    return f"{value:.{nd}f}" if isinstance(value, (int, float)) else "n/a"


def _show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def overview_screen(runs) -> None:
    st.subheader("Обзор прогонов")
    st.caption(
        "Один контракт taskwright = одна строка. Метрики — из `manifest.json` "
        "(совпадают с `TaskResult`); демка ничего не обучает и не пересчитывает."
    )

    t4 = [r for r in runs if r.code == "T4"]
    if len(t4) >= 2:
        games = ", ".join(r.dataset for r in t4)
        st.info(
            "**Переносимость (главный тезис ВКР):** T4 — *один* контракт "
            f"`SupervisedTask` на *двух* разных играх ({games}). "
            "Меняется только реализация задачи, не фреймворк."
        )

    st.dataframe(rl.overview_table(runs), hide_index=True, width="stretch")

    counts = {}
    for r in runs:
        counts[r.contract] = counts.get(r.contract, 0) + 1
    st.caption("Контракты в прогонах: " + " · ".join(f"{k}: {v}" for k, v in counts.items()))


def _metric_row(pairs) -> None:
    cols = st.columns(len(pairs))
    for col, (name, value) in zip(cols, pairs):
        col.metric(name, value)


def drilldown_supervised(run) -> None:
    m = run.metrics
    _metric_row(
        [
            ("precision", _fmt(m.get("precision"))),
            ("recall", _fmt(m.get("recall"))),
            ("f1", _fmt(m.get("f1"))),
            ("ROC AUC", _fmt(m.get("roc_auc"))),
        ]
    )
    if run.code == "T4":
        st.caption("Один контракт `SupervisedTask`, две игры — см. вкладку «Обзор».")

    mat = rl.supervised_material(run)
    if mat is None:
        st.info("Материал для графиков (`graph_material_supervised.npz`) не найден для этого прогона.")
        return
    left, right = st.columns(2)
    with left:
        _show(plots.roc_fig(mat["y_true"], mat["y_score"], m.get("roc_auc", float("nan")), f"{run.label} — ROC"))
    with right:
        _show(
            plots.confusion_fig(
                mat["y_true"], mat["y_pred"], f"{run.label} — матрица ошибок",
                rl.cm_display_labels(run.task_name),
            )
        )


def drilldown_unsupervised(run) -> None:
    if run.is_synthetic:
        st.warning("Датасет **синтетический** — найденные сегменты суть структура генератора, не реальные игроки.")
    m = run.metrics
    _metric_row(
        [
            ("n_clusters", str(m.get("n_clusters", "—"))),
            ("silhouette", _fmt(m.get("silhouette"))),
            ("davies_bouldin", _fmt(m.get("davies_bouldin"))),
        ]
    )
    mat = rl.unsupervised_material(run)
    if mat is None:
        st.info("Материал для графиков (`graph_material_unsupervised.npz`) не найден.")
        return
    _show(plots.clusters_fig(mat["projection"], mat["labels"], f"{run.label} — кластеры (PCA 2D)"))
    st.caption("Кластеризация по поведенческим признакам; целевой `EngagementLevel` в обучении не использовался.")


def drilldown_deterministic(run) -> None:
    df = rl.retention_material(run)
    overall = None
    if df is not None and "overall" in df.columns:
        overall = df.set_index("day_offset")["overall"]

    def ret_at(n: int) -> str:
        if overall is not None and n in overall.index:
            return _fmt(overall.loc[n])
        return "n/a"

    _metric_row(
        [
            ("σκ (sigma_kappa)", str(run.metrics.get("sigma_kappa", "—"))),
            ("retention day 1", ret_at(1)),
            ("retention day 7", ret_at(7)),
            ("retention day 30", ret_at(30)),
        ]
    )
    if df is None:
        st.info("Материал для графиков (`graph_material_retention.csv`) не найден.")
        return
    _show(plots.retention_fig(df, f"{run.label} — кривая retention по дням"))
    st.caption(
        "День-0 = 1.0 по построению. Оверлеи по сегментам `player_type` — сверка с "
        "ground-truth генератора (churner падает быстрее всех, hardcore держится дольше)."
    )


def drilldown_screen(runs) -> None:
    st.subheader("Детально по прогону")
    labels = [f"{r.label}  ·  {r.dataset}" for r in runs]
    idx = st.selectbox("Выберите прогон", range(len(runs)), format_func=lambda i: labels[i])
    run = runs[idx]

    st.markdown(f"### {run.label}")
    st.caption(
        f"Контракт: `{run.contract}`  ·  датасет: `{run.dataset}`  ·  "
        f"прогон: `{run.run_id}`  ·  {run.when}"
    )

    if run.category == "supervised":
        drilldown_supervised(run)
    elif run.category == "unsupervised":
        drilldown_unsupervised(run)
    elif run.category == "deterministic":
        drilldown_deterministic(run)
    else:
        st.info(f"Неизвестная категория контракта: {run.category}")


def main() -> None:
    st.title("taskwright — апробация на открытых игровых данных")
    st.caption("Этап 3 · визуализация готовых прогонов из `runs/` (только чтение).")

    runs = rl.discover_runs()
    if not runs:
        st.warning(
            "В `runs/` нет прогонов. Сначала запустите апробацию (этап 2), например:\n\n"
            "```\n"
            "poetry run python -m scripts.run_t4_dota2\n"
            "poetry run python -m scripts.run_t4_lol\n"
            "poetry run python -m scripts.run_t2_churn\n"
            "poetry run python -m scripts.run_t1_segmentation\n"
            "poetry run python -m scripts.run_t12_retention\n"
            "```"
        )
        st.stop()

    overview_tab, detail_tab = st.tabs(["Обзор всех прогонов", "Детально по прогону"])
    with overview_tab:
        overview_screen(runs)
    with detail_tab:
        drilldown_screen(runs)


if __name__ == "__main__":
    main()
