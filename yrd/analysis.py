from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def classify_pollution_events(
    frame: pd.DataFrame,
    *,
    o3_threshold: float,
    pm25_threshold: float,
) -> pd.Series:
    labels = []
    for _, row in frame.iterrows():
        if row["O3"] >= o3_threshold and row["PM2.5"] >= pm25_threshold:
            labels.append("compound")
        elif row["O3"] >= o3_threshold:
            labels.append("o3")
        elif row["PM2.5"] >= pm25_threshold:
            labels.append("pm25")
        else:
            labels.append("normal")
    return frame.assign(event=labels)["event"]


def write_markdown_summary(
    path: Path,
    *,
    title: str,
    bullets: list[str],
    intro: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if intro:
        lines.extend([intro, ""])
    for bullet in bullets:
        lines.append(f"- {bullet}")
    path.write_text("\n".join(lines) + "\n")


def metrics_bullets(metrics: dict[str, object]) -> list[str]:
    bullets: list[str] = []
    model_metrics = metrics.get("joint_test", {})
    baseline_metrics = metrics.get("baseline_test", {})
    for horizon, summary in sorted(model_metrics.items(), key=lambda item: int(item[0])):
        baseline = baseline_metrics.get(horizon, {})
        model_rmse = summary.get("rmse")
        base_rmse = baseline.get("rmse")
        if model_rmse is not None and base_rmse is not None:
            bullets.append(
                f"{horizon}h 预测上，联合 MLP 的 RMSE 为 {model_rmse:.4f}，持续性基线为 {base_rmse:.4f}。"
            )
    return bullets


def coupling_bullets(summary: dict[str, object]) -> list[str]:
    bullets = [
        f"整体 `EI^{{nis}}` 为 {summary['ei_nis']:.4f}，`Syn_p^{{nis}}` 为 {summary['syn_nis']:.4f}。",
    ]
    group_terms = summary.get("group_ei_nis", {})
    if group_terms:
        ranked = sorted(group_terms.items(), key=lambda item: item[1], reverse=True)
        top_name, top_value = ranked[0]
        bullets.append(f"源组中贡献最大的部分是 `{top_name}`，其对应的 `EI^{{nis}}` 为 {top_value:.4f}。")
    return bullets


def coupling_by_horizon_bullets(summary_by_horizon: dict[str, dict[str, object]]) -> list[str]:
    bullets: list[str] = []
    for horizon, summary in sorted(summary_by_horizon.items(), key=lambda item: int(item[0])):
        bullets.append(
            f"{horizon}h 目标上，整体 `EI^{{nis}}` 为 {summary['ei_nis']:.4f}，`Syn_p^{{nis}}` 为 {summary['syn_nis']:.4f}。"
        )
    if "1" in summary_by_horizon and "24" in summary_by_horizon:
        delta = summary_by_horizon["24"]["syn_nis"] - summary_by_horizon["1"]["syn_nis"]
        bullets.append(f"`24h - 1h` 的协同差值为 {delta:.4f}。")
    return bullets


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
