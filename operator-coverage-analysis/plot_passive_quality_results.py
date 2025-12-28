#!/usr/bin/env python3
"""
Generate matplotlib figures from the CSV outputs produced by analyze_passive_quality.py.
This script reads summaries under results/ and creates diagnostic plots (bar charts,
boxplots, violin plots, ECDF curves, scatter diagnostics, KPI availability) without
recomputing upstream statistics. Axes and labels adapt automatically depending on
whether the inputs are normalized (0–1) or raw KPI averages.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from scipy import stats
except ImportError:  # pragma: no cover - optional dependency
    stats = None


SUMMARY_FILES = {
    "overall": "summary_overall.csv",
    "by_rat": "summary_by_rat.csv",
    "by_speed": "summary_by_speed.csv",
    "by_operator": "summary_by_operator.csv",
    "by_rat_operator": "summary_by_rat_operator.csv",
    "by_speed_operator": "summary_by_speed_operator.csv",
}

OPTIONAL_FILES = {
    "per_file": "per_file_stats.csv",
    "per_file_speed": "per_file_stats_by_speed.csv",
    "kpi_presence": "kpi_presence.csv",
    "operator_metadata": "operator_names.csv",
}

ENV_DISPLAY = {
    "inside": "Inside",
    "outdoor_driving": "Outdoor driving",
    "outside_walking": "Outside walking",
    "od_unknown": "Outdoor (unknown)",
}

SPEED_BUCKET_DISPLAY = {
    "inside": "Inside (indoor)",
    "od_quasi_static": "Outdoor <=0.5 m/s",
    "od_slow": "Outdoor 0.5-5 m/s",
    "od_fast": "Outdoor >5 m/s",
    "od_unknown": "Outdoor (unknown speed)",
}

SPEED_BUCKET_SHORT = {
    "inside": "Indoor",
    "od_quasi_static": "Outdoor <=0.5",
    "od_slow": "Outdoor 0.5-5",
    "od_fast": "Outdoor >5",
    "od_unknown": "Outdoor ?",
}

ENV_COLORS = {
    "inside": "#1f77b4",  # blue
    "outdoor_driving": "#ff7f0e",  # orange
    "outside_walking": "#2ca02c",  # green
    "od_unknown": "#d62728",  # red
}

SPEED_BUCKET_COLORS = {
    "inside": "#1f77b4",
    "od_quasi_static": "#ff7f0e",
    "od_slow": "#2ca02c",
    "od_fast": "#d62728",
    "od_unknown": "#9467bd",
}

VIOLIN_COLOR_SEQUENCE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
]

def _build_operator_mapping(metadata: Optional[pd.DataFrame]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if metadata is None:
        return mapping
    if not {"operator", "name"}.issubset(metadata.columns):
        return mapping
    for _, row in metadata.dropna(subset=["operator", "name"]).iterrows():
        key = str(row["operator"]).strip()
        value = str(row["name"]).strip()
        if key and value:
            mapping[key] = value
    return mapping


def _operator_env_color(operator_name: str, env: str) -> str:
    operator_name = operator_name.strip()
    env = env.strip() if isinstance(env, str) else env
    return OPERATOR_ENV_COLORS.get(operator_name, {}).get(env, _env_color(env))

OPERATOR_ENV_COLORS = {
    "TIM": {"inside": "#1f77b4", "outdoor_driving": "#6f2dbd"},
    "Vodafone": {"inside": "#1f77b4", "outdoor_driving": "#6f2dbd"},
    "Iliad": {"inside": "#ffd60a", "outdoor_driving": "#ff7f0e"},
    "Wind": {"inside": "#ffd60a", "outdoor_driving": "#ff7f0e"},
    "WindTre": {"inside": "#ffd60a", "outdoor_driving": "#ff7f0e"},
    "TIM/Vodafone": {"inside": "#1f77b4", "outdoor_driving": "#6f2dbd"},
    "Iliad/Wind": {"inside": "#ffd60a", "outdoor_driving": "#ff7f0e"},
}

OPERATOR_GROUP_LABELS = {
    "TIM": "TIM/Vodafone",
    "Vodafone": "TIM/Vodafone",
    "Iliad": "Iliad/Wind",
    "Wind": "Iliad/Wind",
    "WindTre": "Iliad/Wind",
}


def _slug_to_title(value: str) -> str:
    cleaned = value.replace("_", " ").strip()
    if not cleaned:
        return value
    return cleaned[0].upper() + cleaned[1:]


def _pretty_env(env: str) -> str:
    return ENV_DISPLAY.get(env, _slug_to_title(env))


def _pretty_speed_bucket(bucket: str) -> str:
    return SPEED_BUCKET_DISPLAY.get(bucket, _slug_to_title(bucket))


def _short_speed_bucket(bucket: str) -> str:
    return SPEED_BUCKET_SHORT.get(bucket, _pretty_speed_bucket(bucket))


def _add_mean_median_legend(ax: plt.Axes, loc: str = "upper left") -> None:
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    labels_set = set(existing_labels)
    new_handles: List[Line2D] = []
    new_labels: List[str] = []

    if "Median" not in labels_set:
        new_handles.append(Line2D([0], [0], color="red", linewidth=2.0))
        new_labels.append("Median")
    if "Mean" not in labels_set:
        new_handles.append(Line2D([0], [0], color="black", linewidth=1.5))
        new_labels.append("Mean")

    if not new_handles:
        return

    ax.legend(
        existing_handles + new_handles,
        existing_labels + new_labels,
        loc=loc,
        frameon=False,
    )


def _env_color(key: str, default: str = "#7f7f7f") -> str:
    return ENV_COLORS.get(key, default)


def _speed_color(key: str, default: str = "#7f7f7f") -> str:
    return SPEED_BUCKET_COLORS.get(key, default)


def _apply_violin_colors(parts: dict, colors: List[str]) -> None:
    for pc, color in zip(parts.get("bodies", []), colors):
        pc.set_facecolor(color)
        pc.set_edgecolor("black")
        pc.set_alpha(0.65)


def _normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "mean_quality": "mean",
        "median_quality": "median",
        "std_quality": "std",
        "file_count": "count",
    }
    return df.rename(columns=rename_map)


def _normalize_per_file(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "mean_quality": "mean",
        "median_quality": "median",
        "std_quality": "std",
    }
    return df.rename(columns=rename_map)


def load_data(results_dir: Path) -> Dict[str, Optional[pd.DataFrame]]:
    data: Dict[str, Optional[pd.DataFrame]] = {}
    required_keys = {"overall", "by_rat", "by_speed"}
    fallback_frames = {
        "by_operator": pd.DataFrame(columns=["operator", "env", "count", "mean", "median", "std"]),
        "by_rat_operator": pd.DataFrame(columns=["rat", "operator", "env", "count", "mean", "median", "std"]),
        "by_speed_operator": pd.DataFrame(columns=["operator", "speed_bucket", "count", "mean", "median", "std"]),
    }
    for key, filename in SUMMARY_FILES.items():
        path = results_dir / filename
        if not path.exists():
            if key in required_keys:
                raise FileNotFoundError(f"Required summary file missing: {path}")
            data[key] = fallback_frames.get(key, pd.DataFrame())
            continue
        df = pd.read_csv(path)
        data[key] = _normalize_summary(df)

    per_file_path = results_dir / OPTIONAL_FILES["per_file"]
    data["per_file"] = (
        _normalize_per_file(pd.read_csv(per_file_path)) if per_file_path.exists() else None
    )
    per_file_speed_path = results_dir / OPTIONAL_FILES["per_file_speed"]
    data["per_file_speed"] = (
        _normalize_per_file(pd.read_csv(per_file_speed_path)) if per_file_speed_path.exists() else None
    )
    kpi_path = results_dir / OPTIONAL_FILES["kpi_presence"]
    data["kpi_presence"] = pd.read_csv(kpi_path) if kpi_path.exists() else None
    operator_meta_path = Path(OPTIONAL_FILES["operator_metadata"])
    if not operator_meta_path.exists():
        operator_meta_path = results_dir / OPTIONAL_FILES["operator_metadata"]
    if operator_meta_path.exists():
        try:
            data["operator_metadata"] = pd.read_csv(operator_meta_path)
        except Exception:
            data["operator_metadata"] = None
    else:
        data["operator_metadata"] = None
    return data


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _save_text_panel(path: Path, title: str, lines: List[str]) -> None:
    if not lines:
        return
    fig, ax = plt.subplots(figsize=(7, max(2.0, 0.4 * len(lines) + 1.0)))
    ax.axis("off")
    ax.set_title(title, loc="left")
    ax.text(
        0.0,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
    )
    _save_figure(path)


def _value_limits(values: Iterable[float], errors: Optional[Iterable[float]] = None) -> Optional[Tuple[float, float]]:
    values_series = pd.Series(pd.to_numeric(values, errors="coerce"), dtype=float).dropna()
    if values_series.empty:
        return None
    if errors is not None:
        errors_series = pd.Series(pd.to_numeric(errors, errors="coerce"), dtype=float).reindex(values_series.index, fill_value=0.0)
        lower = (values_series - errors_series).dropna()
        upper = (values_series + errors_series).dropna()
        if lower.empty or upper.empty:
            finite = values_series
        else:
            finite = pd.concat([lower, upper], ignore_index=True)
    else:
        finite = values_series
    finite = finite.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return None
    return float(finite.min()), float(finite.max())


def _is_normalized(values: Iterable[float]) -> bool:
    finite = pd.Series(pd.to_numeric(values, errors="coerce"), dtype=float).dropna()
    if finite.empty:
        return True
    return finite.min() >= -0.05 and finite.max() <= 1.05


def _range_text(limits: Optional[Tuple[float, float]]) -> str:
    if not limits:
        return ""
    return f"{limits[0]:.1f}..{limits[1]:.1f}"


def _axis_label(base: str, normalized: bool, limits: Optional[Tuple[float, float]]) -> str:
    if normalized:
        return f"{base} (0-1)"
    if limits:
        return f"{base} (raw range {_range_text(limits)})"
    return f"{base} (raw units)"


def _expanded_limits(limits: Optional[Tuple[float, float]], normalized: bool) -> Optional[Tuple[float, float]]:
    if limits is None:
        return None
    low, high = limits
    if normalized:
        low_lim = min(0.0, low - 0.05)
        high_lim = max(1.0, high + 0.05)
        return (low_lim, high_lim)
    span = high - low
    if span == 0:
        padding = max(1.0, max(abs(high), abs(low), 1.0) * 0.1)
    else:
        padding = max(1.0, abs(span) * 0.1)
    return (low - padding, high + padding)


def _apply_axis_meta(
    ax: plt.Axes,
    values: Iterable[float],
    errors: Optional[Iterable[float]] = None,
    base_label: str = "NetworkQuality",
    axis: str = "y",
) -> Tuple[bool, Optional[Tuple[float, float]]]:
    limits = _value_limits(values, errors)
    normalized = _is_normalized(values)
    label = _axis_label(base_label, normalized, limits)
    expanded = _expanded_limits(limits, normalized)

    if axis == "y":
        ax.set_ylabel(label)
        if expanded:
            ax.set_ylim(expanded)
    else:
        ax.set_xlabel(label)
        if expanded:
            ax.set_xlim(expanded)

    return normalized, limits


def _welch_pvalue(samples_a: pd.Series, samples_b: pd.Series) -> Optional[float]:
    if stats is None:
        return None
    a = pd.to_numeric(samples_a, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(samples_b, errors="coerce").dropna().to_numpy()
    if len(a) < 2 or len(b) < 2:
        return None
    _, p_value = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(p_value)


def plot_overall_mean_bar(
    summary_overall: pd.DataFrame,
    per_file: Optional[pd.DataFrame],
    output_path: Path,
) -> bool:
    subset = summary_overall[summary_overall["env"].isin(["inside", "outdoor_driving"])]
    if subset.empty:
        return False

    inside_mean = subset.loc[subset["env"] == "inside", "mean"].iloc[0]
    outdoor_mean = subset.loc[subset["env"] == "outdoor_driving", "mean"].iloc[0]
    delta = inside_mean - outdoor_mean
    pct = np.nan
    if np.isfinite(outdoor_mean) and outdoor_mean != 0:
        pct = delta / outdoor_mean

    p_value = None
    if per_file is not None:
        inside_samples = per_file.loc[per_file["env"] == "inside", "mean"]
        outdoor_samples = per_file.loc[per_file["env"] == "outdoor_driving", "mean"]
        p_value = _welch_pvalue(inside_samples, outdoor_samples)

    plt.figure(figsize=(6, 4))
    positions = np.arange(len(subset))
    colors = [_env_color(env) for env in subset["env"]]
    ax = plt.gca()
    plt.subplots_adjust(bottom=0.28)
    bars = ax.bar(
        positions,
        subset["mean"],
        yerr=subset["std"],
        capsize=6,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    median_label_added = False
    bar_width = bars.patches[0].get_width() if bars.patches else 0.6
    for xpos, median, env_color in zip(positions, subset["median"], colors):
        if np.isfinite(median):
            ax.hlines(
                median,
                xpos - bar_width / 2,
                xpos + bar_width / 2,
                colors="red",
                linewidth=2.0,
                label="Median" if not median_label_added else None,
            )
            median_label_added = True

    display_envs = [_pretty_env(env) if isinstance(env, str) else "" for env in subset["env"]]
    ax.set_xticks(positions, display_envs)
    ax.set_xlabel("Environment")

    normalized, limits = _apply_axis_meta(ax, subset["mean"], subset["std"])
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_title(f"Mean NetworkQuality by Environment {scale_suffix}")

    ymin, ymax = ax.get_ylim()
    ax.text(
        0.5,
        ymax - (ymax - ymin) * 0.05,
        f"Δ={delta:.3f} ({_format_percent(pct)}) • p={_format_pvalue(p_value)}",
        ha="center",
        va="top",
    )

    env_handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=0.6, label=_pretty_env(env))
        for env, color in zip(subset["env"], colors)
    ]
    unique_handles = {}
    for handle in env_handles:
        unique_handles[handle.get_label()] = handle
    legend_handles = list(unique_handles.values())
    legend_labels = [handle.get_label() for handle in legend_handles]
    if median_label_added:
        legend_handles.append(Line2D([0], [0], color="red", linewidth=2.0, label="Median"))
        legend_labels.append("Median")
    ax.legend(legend_handles, legend_labels, frameon=False, loc="best")

    lines = []
    for env, mean_val, median_val in zip(subset["env"], subset["mean"], subset["median"]):
        lines.append(f"{_pretty_env(env)}: mean={_format_stat(mean_val)}, median={_format_stat(median_val)}")
    if lines:
        ax.text(
            1.02,
            0.95,
            "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )

    _save_figure(output_path)
    return True


def _format_percent(delta_pct: float) -> str:
    if not np.isfinite(delta_pct):
        return "n/a"
    return f"{delta_pct*100:.1f}%"


def _format_pvalue(p_value: Optional[float]) -> str:
    if p_value is None or not np.isfinite(p_value):
        return "n/a"
    if p_value < 1e-4:
        return "<1e-4"
    return f"{p_value:.4f}"


def _format_stat(value: float) -> str:
    return f"{value:.3f}" if np.isfinite(value) else "n/a"


def plot_by_rat_mean_bar(
    summary_by_rat: pd.DataFrame,
    per_file: Optional[pd.DataFrame],
    output_path: Path,
) -> bool:
    if summary_by_rat.empty:
        return False

    rats = sorted(summary_by_rat["rat"].dropna().unique())
    envs = ["inside", "outdoor_driving"]
    positions = np.arange(len(rats))
    width = 0.35

    plt.figure(figsize=(max(7, len(rats) * 1.5), 4))
    ax = plt.gca()
    median_label_added = False
    info_lines: List[str] = []

    for idx, env in enumerate(envs):
        env_data = summary_by_rat[summary_by_rat["env"] == env]
        means = []
        medians = []
        stds = []
        counts = []
        for rat in rats:
            row = env_data[env_data["rat"] == rat]
            if row.empty:
                means.append(np.nan)
                medians.append(np.nan)
                stds.append(0.0)
                counts.append(np.nan)
            else:
                means.append(row.iloc[0]["mean"])
                medians.append(row.iloc[0]["median"])
                stds.append(row.iloc[0]["std"])
                counts.append(row.iloc[0].get("count", np.nan))
        bar_positions = positions + (idx - 0.5) * width
        bars = ax.bar(
            bar_positions,
            means,
            width=width,
            yerr=stds,
            capsize=5,
            color=_env_color(env),
            edgecolor="black",
            linewidth=0.6,
        )
        bar_width = bars.patches[0].get_width() if bars.patches else width
        for xpos, median in zip(bar_positions, medians):
            if np.isfinite(median):
                ax.hlines(
                    median,
                    xpos - bar_width / 2,
                    xpos + bar_width / 2,
                    colors="red",
                    linewidth=2.0,
                    label="Median" if not median_label_added else None,
                )
                median_label_added = True
        for rat, mean_val, median_val, count in zip(rats, means, medians, counts):
            if not np.isfinite(mean_val) and not np.isfinite(median_val):
                continue
            count_txt = f", n={int(count)}" if np.isfinite(count) else ""
            info_lines.append(
                f"{rat} - {_pretty_env(env)}: mean={_format_stat(mean_val)} | median={_format_stat(median_val)}{count_txt}"
            )

    ax.set_xticks(positions, rats)
    ax.set_xlabel("RAT")

    normalized, limits = _apply_axis_meta(ax, summary_by_rat["mean"], summary_by_rat["std"])
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_title(f"Mean NetworkQuality by RAT and Environment {scale_suffix}")
    env_handles = [
        Patch(facecolor=_env_color(env), edgecolor="black", linewidth=0.6, label=_pretty_env(env))
        for env in envs
    ]
    legend_handles = env_handles.copy()
    if median_label_added:
        legend_handles.append(Line2D([0], [0], color="red", linewidth=2.0, label="Median"))
    ax.legend(legend_handles, [handle.get_label() for handle in legend_handles], frameon=False, loc="best")

    annotation_lines: List[str] = []
    if per_file is not None:
        for rat in rats:
            inside_mean = summary_by_rat.loc[
                (summary_by_rat["rat"] == rat) & (summary_by_rat["env"] == "inside"),
                "mean",
            ]
            outdoor_mean = summary_by_rat.loc[
                (summary_by_rat["rat"] == rat) & (summary_by_rat["env"] == "outdoor_driving"),
                "mean",
            ]
            if inside_mean.empty or outdoor_mean.empty:
                continue
            delta = inside_mean.iloc[0] - outdoor_mean.iloc[0]
            pct = np.nan
            if np.isfinite(outdoor_mean.iloc[0]) and outdoor_mean.iloc[0] != 0:
                pct = delta / outdoor_mean.iloc[0]
            inside_samples = per_file.loc[
                (per_file["rat"] == rat) & (per_file["env"] == "inside"), "mean"
            ]
            outdoor_samples = per_file.loc[
                (per_file["rat"] == rat) & (per_file["env"] == "outdoor_driving"), "mean"
            ]
            p_value = _welch_pvalue(inside_samples, outdoor_samples)
            annotation_lines.append(
                f"{rat}: Δ={delta:.3f} ({_format_percent(pct)}), p={_format_pvalue(p_value)}"
            )

    text_y = 0.95
    if annotation_lines:
        ax.text(
            1.02,
            text_y,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
        text_y -= 0.25

    if info_lines:
        ax.text(
            1.02,
            text_y,
            "\n".join(info_lines),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )

    _save_figure(output_path)
    return True


def plot_by_operator_mean_bar(
    summary_by_operator: pd.DataFrame,
    per_file: Optional[pd.DataFrame],
    operator_metadata: Optional[pd.DataFrame],
    output_path: Path,
) -> bool:
    if summary_by_operator.empty:
        return False

    operator_mapping = _build_operator_mapping(operator_metadata)
    operators = sorted(summary_by_operator["operator"].dropna().unique())
    envs = ["inside", "outdoor_driving"]
    positions = np.arange(len(operators))
    width = 0.35

    plt.figure(figsize=(max(7, len(operators) * 1.5), 4))
    ax = plt.gca()
    median_label_added = False
    info_lines: List[str] = []

    legend_entries: Dict[Tuple[str, str], Patch] = {}

    for idx, env in enumerate(envs):
        env_data = summary_by_operator[summary_by_operator["env"] == env]
        means = []
        medians = []
        stds = []
        counts = []
        for operator in operators:
            row = env_data[env_data["operator"] == operator]
            if row.empty:
                means.append(np.nan)
                medians.append(np.nan)
                stds.append(0.0)
                counts.append(np.nan)
            else:
                means.append(row.iloc[0]["mean"])
                medians.append(row.iloc[0]["median"])
                stds.append(row.iloc[0]["std"])
                counts.append(row.iloc[0].get("count", np.nan))
        bar_positions = positions + (idx - 0.5) * width
        display_names = [operator_mapping.get(op, op) for op in operators]
        bar_colors = [_operator_env_color(name, env) for name in display_names]
        bars = ax.bar(
            bar_positions,
            means,
            width=width,
            yerr=stds,
            capsize=5,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.6,
        )
        bar_width = bars.patches[0].get_width() if bars.patches else width
        for xpos, median in zip(bar_positions, medians):
            if np.isfinite(median):
                ax.hlines(
                    median,
                    xpos - bar_width / 2,
                    xpos + bar_width / 2,
                    colors="red",
                    linewidth=2.0,
                    label="Median" if not median_label_added else None,
                )
                median_label_added = True
        for operator, display_name, mean_val, median_val, count in zip(
            operators, display_names, means, medians, counts
        ):
            if not np.isfinite(mean_val) and not np.isfinite(median_val):
                continue
            count_txt = f", n={int(count)}" if np.isfinite(count) else ""
            info_lines.append(
                f"{display_name} - {_pretty_env(env)}: "
                f"mean={_format_stat(mean_val)} | median={_format_stat(median_val)}{count_txt}"
            )
            color_key = (
                OPERATOR_GROUP_LABELS.get(display_name, display_name),
                _pretty_env(env),
            )
            if color_key not in legend_entries:
                legend_entries[color_key] = Patch(
                    facecolor=_operator_env_color(display_name, env),
                    edgecolor="black",
                    linewidth=0.6,
                    label=f"{color_key[0]} {color_key[1].lower()}",
                )

    display_names = [operator_mapping.get(operator, operator) for operator in operators]
    ax.set_xticks(positions, display_names, rotation=20)
    ax.set_xlabel("Carrier")

    normalized, limits = _apply_axis_meta(ax, summary_by_operator["mean"], summary_by_operator["std"])
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_title(f"Mean NetworkQuality by Carrier and Environment {scale_suffix}")
    legend_handles = list(legend_entries.values())
    if median_label_added:
        legend_handles.append(Line2D([0], [0], color="red", linewidth=2.0, label="Median"))
    if legend_handles:
        ax.legend(legend_handles, [handle.get_label() for handle in legend_handles], frameon=False, loc="best")

    annotation_lines: List[str] = []
    if per_file is not None and "operator" in per_file.columns:
        for operator in operators:
            inside_samples = per_file.loc[
                (per_file["operator"] == operator) & (per_file.get("env") == "inside"),
                "mean",
            ]
            outdoor_samples = per_file.loc[
                (per_file["operator"] == operator) & (per_file.get("env") == "outdoor_driving"),
                "mean",
            ]
            if inside_samples.empty or outdoor_samples.empty:
                continue
            delta = inside_samples.mean() - outdoor_samples.mean()
            pct = np.nan
            if np.isfinite(outdoor_samples.mean()) and outdoor_samples.mean() != 0:
                pct = delta / outdoor_samples.mean()
            p_value = _welch_pvalue(inside_samples, outdoor_samples)
            annotation_lines.append(
                f"{operator_mapping.get(operator, operator)}: Δ={delta:.3f} ({_format_percent(pct)}), p={_format_pvalue(p_value)}"
            )

    text_y = 0.95
    if annotation_lines:
        ax.text(
            1.02,
            text_y,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
        text_y -= 0.25

    if info_lines:
        ax.text(
            1.02,
            text_y,
            "\n".join(info_lines),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )

    _save_figure(output_path)
    return True


def plot_by_rat_operator_mean_bar(
    summary_by_rat_operator: pd.DataFrame,
    per_file: Optional[pd.DataFrame],
    operator_metadata: Optional[pd.DataFrame],
    output_path: Path,
) -> bool:
    if summary_by_rat_operator.empty:
        return False

    operator_mapping = _build_operator_mapping(operator_metadata)

    combos = (
        summary_by_rat_operator[["rat", "operator"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["rat", "operator"])
    )
    if combos.empty:
        return False

    envs = ["inside", "outdoor_driving"]
    display_info = [
        (row["rat"], row["operator"], operator_mapping.get(row["operator"], row["operator"]))
        for _, row in combos.iterrows()
    ]
    labels = [f"{rat}\n{display}" for rat, _, display in display_info]
    positions = np.arange(len(labels))
    width = 0.35

    fig_width = max(8, len(labels) * 2.0)
    plt.figure(figsize=(fig_width, 4.2))
    ax = plt.gca()
    median_label_added = False
    info_lines: List[str] = []
    legend_entries: Dict[Tuple[str, str], Patch] = {}

    for idx, env in enumerate(envs):
        env_data = summary_by_rat_operator[summary_by_rat_operator["env"] == env]
        means = []
        medians = []
        stds = []
        counts = []
        for (_, row), (_, op_code, display_name) in zip(combos.iterrows(), display_info):
            row_data = env_data[
                (env_data["rat"] == row["rat"]) & (env_data["operator"] == row["operator"])
            ]
            if row_data.empty:
                means.append(np.nan)
                medians.append(np.nan)
                stds.append(0.0)
                counts.append(np.nan)
            else:
                means.append(row_data.iloc[0]["mean"])
                medians.append(row_data.iloc[0]["median"])
                stds.append(row_data.iloc[0]["std"])
                counts.append(row_data.iloc[0].get("count", np.nan))
        bar_positions = positions + (idx - 0.5) * width
        bar_colors = [_operator_env_color(display_name, env) for _, _, display_name in display_info]
        bars = ax.bar(
            bar_positions,
            means,
            width=width,
            yerr=stds,
            capsize=5,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.6,
        )
        bar_width = bars.patches[0].get_width() if bars.patches else width
        for xpos, median in zip(bar_positions, medians):
            if np.isfinite(median):
                ax.hlines(
                    median,
                    xpos - bar_width / 2,
                    xpos + bar_width / 2,
                    colors="red",
                    linewidth=2.0,
                    label="Median" if not median_label_added else None,
                )
                median_label_added = True
        for (rat, op_code, display_name), label, mean_val, median_val, count in zip(
            display_info, labels, means, medians, counts
        ):
            if not np.isfinite(mean_val) and not np.isfinite(median_val):
                continue
            count_txt = f", n={int(count)}" if np.isfinite(count) else ""
            info_lines.append(
                f"{label.replace(chr(10), ' / ')} - {_pretty_env(env)}: "
                f"mean={_format_stat(mean_val)} | median={_format_stat(median_val)}{count_txt}"
            )
            color_key = (
                OPERATOR_GROUP_LABELS.get(display_name, display_name),
                _pretty_env(env),
            )
            if color_key not in legend_entries:
                legend_entries[color_key] = Patch(
                    facecolor=_operator_env_color(display_name, env),
                    edgecolor="black",
                    linewidth=0.6,
                    label=f"{color_key[0]} {color_key[1].lower()}",
                )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("RAT / Carrier")

    normalized, limits = _apply_axis_meta(ax, summary_by_rat_operator["mean"], summary_by_rat_operator["std"])
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_title(f"Mean NetworkQuality by RAT, Carrier, and Environment {scale_suffix}")
    legend_handles = list(legend_entries.values())
    if median_label_added:
        legend_handles.append(Line2D([0], [0], color="red", linewidth=2.0, label="Median"))
    if legend_handles:
        ax.legend(legend_handles, [handle.get_label() for handle in legend_handles], frameon=False, loc="best")

    stats_lines: List[str] = []
    if per_file is not None and {"rat", "operator", "env"}.issubset(per_file.columns):
        annotation_lines: List[str] = []
        for (_, row), (_, op_code, display_name), label in zip(combos.iterrows(), display_info, labels):
            inside_samples = per_file.loc[
                (per_file["rat"] == row["rat"])
                & (per_file["operator"] == row["operator"])
                & (per_file["env"] == "inside"),
                "mean",
            ]
            outdoor_samples = per_file.loc[
                (per_file["rat"] == row["rat"])
                & (per_file["operator"] == row["operator"])
                & (per_file["env"] == "outdoor_driving"),
                "mean",
            ]
            if inside_samples.empty or outdoor_samples.empty:
                continue
            delta = inside_samples.mean() - outdoor_samples.mean()
            pct = np.nan
            if np.isfinite(outdoor_samples.mean()) and outdoor_samples.mean() != 0:
                pct = delta / outdoor_samples.mean()
            p_value = _welch_pvalue(inside_samples, outdoor_samples)
            annotation_lines.append(
                f"{label.replace(chr(10), ' / ')}: Δ={delta:.3f} ({_format_percent(pct)}), p={_format_pvalue(p_value)}"
            )
        if annotation_lines:
            stats_lines.append("Inside vs Outdoor driving (Welch t-test):")
            stats_lines.extend(annotation_lines)

    if info_lines:
        if stats_lines:
            stats_lines.append("")
        stats_lines.append("Environment stats per RAT / Carrier:")
        stats_lines.extend(info_lines)

    if stats_lines:
        stats_path = output_path.with_name(f"{output_path.stem}_stats{output_path.suffix}")
        _save_text_panel(stats_path, "RAT / Carrier Environment Statistics", stats_lines)

    _save_figure(output_path)
    return True


def plot_speed_buckets_bar(
    summary_by_speed: pd.DataFrame,
    per_file_speed: Optional[pd.DataFrame],
    output_path: Path,
) -> bool:
    if summary_by_speed.empty:
        return False

    order = ["inside", "od_quasi_static", "od_slow", "od_fast"]
    df = summary_by_speed.copy()
    df["order"] = df["speed_bucket"].apply(lambda x: order.index(x) if x in order else len(order))
    df.sort_values("order", inplace=True)

    plt.figure(figsize=(max(7.5, len(df) * 1.6), 4.2))
    ax = plt.gca()
    positions = np.arange(len(df))
    colors = [_speed_color(bucket) for bucket in df["speed_bucket"]]
    bars = ax.bar(
        positions,
        df["mean"],
        yerr=df["std"],
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    bar_width = bars.patches[0].get_width() if bars.patches else 0.6
    median_label_added = False
    medians = df.get("median", pd.Series([np.nan] * len(df)))
    for xpos, median in zip(positions, medians):
        if np.isfinite(median):
            ax.hlines(
                median,
                xpos - bar_width / 2,
                xpos + bar_width / 2,
                colors="red",
                linewidth=2.0,
                label="Median" if not median_label_added else None,
            )
            median_label_added = True

    display_buckets = [_short_speed_bucket(bucket) for bucket in df["speed_bucket"]]
    ax.set_xticks(positions, display_buckets)
    ax.set_xlabel("Speed Bucket")

    normalized, limits = _apply_axis_meta(ax, df["mean"], df["std"])
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_title(
        f"Mean NetworkQuality by Speed Bucket {scale_suffix}\n"
        "Inside (indoor); Outdoor <=0.5 m/s; Outdoor 0.5-5 m/s; Outdoor >5 m/s"
    )

    legend_handles: List[Line2D | Patch] = []
    seen_labels: set[str] = set()
    for bucket, color in zip(df["speed_bucket"], colors):
        label = _short_speed_bucket(bucket)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        legend_handles.append(Patch(facecolor=color, edgecolor="black", linewidth=0.6, label=label))
    if median_label_added:
        legend_handles.append(Line2D([0], [0], color="red", linewidth=2.0, label="Median"))
    if legend_handles:
        ax.legend(legend_handles, [handle.get_label() for handle in legend_handles], frameon=False, loc="best")

    stats_lines = [
        f"{_pretty_speed_bucket(bucket)}: mean={_format_stat(mean)} | median={_format_stat(median)}"
        for bucket, mean, median in zip(df["speed_bucket"], df["mean"], medians)
    ]

    annotation_lines: List[str] = []
    if per_file_speed is not None:
        inside_samples = per_file_speed.loc[per_file_speed["speed_bucket"] == "inside", "mean"]
        for bucket in df["speed_bucket"]:
            if bucket == "inside":
                continue
            label = _pretty_speed_bucket(bucket)
            bucket_mean = df.loc[df["speed_bucket"] == bucket, "mean"].iloc[0]
            inside_mean = df.loc[df["speed_bucket"] == "inside", "mean"].iloc[0]
            delta = inside_mean - bucket_mean
            pct = np.nan
            if np.isfinite(bucket_mean) and bucket_mean != 0:
                pct = delta / bucket_mean
            bucket_samples = per_file_speed.loc[per_file_speed["speed_bucket"] == bucket, "mean"]
            p_value = _welch_pvalue(inside_samples, bucket_samples)
            annotation_lines.append(
                f"Inside vs {label}: Δ={delta:.3f} ({_format_percent(pct)}), p={_format_pvalue(p_value)}"
            )

    text_y = 0.95
    if annotation_lines:
        ax.text(
            1.02,
            text_y,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        text_y -= 0.3

    if stats_lines:
        ax.text(
            1.02,
            text_y,
            "\n".join(stats_lines),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        text_y -= 0.2

    explanation = [
        "Labels:",
        "Indoor = inside measurements",
        "Outdoor <=0.5 = quasi-static driving", 
        "Outdoor 0.5-5 = slow driving",
        "Outdoor >5 = fast driving",
    ]
    ax.text(
        -0.02,
        -0.28,
        "\n".join(explanation),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
    )

    _save_figure(output_path)

    if annotation_lines:
        stats_path = output_path.with_name(f"{output_path.stem}_stats{output_path.suffix}")
        _save_text_panel(stats_path, "Speed Bucket Comparison (Inside baseline)", annotation_lines)

    return True


def plot_speed_buckets_operator_bar(
    summary_by_speed_operator: pd.DataFrame,
    per_file_speed: Optional[pd.DataFrame],
    operator_metadata: Optional[pd.DataFrame],
    output_path: Path,
) -> bool:
    if summary_by_speed_operator.empty:
        return False

    operator_mapping = _build_operator_mapping(operator_metadata)

    operators = sorted(summary_by_speed_operator["operator"].dropna().unique())
    if not operators:
        return False

    bucket_order = ["inside", "od_quasi_static", "od_slow", "od_fast"]
    plt.figure(figsize=(max(8, len(operators) * 1.7), 4.2))
    ax = plt.gca()
    width = 0.18
    positions = np.arange(len(operators))
    median_label_added = False
    info_lines: List[str] = []

    display_names = [operator_mapping.get(op, op) for op in operators]
    env_colors = {
        "Inside": "#1f77b4",  # blue
        "Outdoor": "#6f2dbd",
    }
    if all(name in {"Iliad", "Wind", "WindTre"} for name in display_names):
        env_colors = {
            "Inside": "#ffd60a",
            "Outdoor": "#ff7f0e",
        }

    for idx, bucket in enumerate(bucket_order):
        bucket_data = summary_by_speed_operator[summary_by_speed_operator["speed_bucket"] == bucket]
        means = []
        medians = []
        stds = []
        counts = []
        for operator in operators:
            row = bucket_data[bucket_data["operator"] == operator]
            if row.empty:
                means.append(np.nan)
                medians.append(np.nan)
                stds.append(0.0)
                counts.append(np.nan)
            else:
                means.append(row.iloc[0]["mean"])
                medians.append(row.iloc[0]["median"])
                stds.append(row.iloc[0]["std"])
                counts.append(row.iloc[0].get("count", np.nan))
        bar_positions = positions + (idx - (len(bucket_order) - 1) / 2) * width
        bars = ax.bar(
            bar_positions,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=_speed_color(bucket),
            edgecolor="black",
            linewidth=0.6,
        )
        bar_width = bars.patches[0].get_width() if bars.patches else width
        for xpos, median in zip(bar_positions, medians):
            if np.isfinite(median):
                ax.hlines(
                    median,
                    xpos - bar_width / 2,
                    xpos + bar_width / 2,
                    colors="red",
                    linewidth=2.0,
                    label="Median" if not median_label_added else None,
                )
                median_label_added = True
        for operator, display_name, mean_val, median_val, count in zip(
            operators, display_names, means, medians, counts
        ):
            if not np.isfinite(mean_val) and not np.isfinite(median_val):
                continue
            count_txt = f", n={int(count)}" if np.isfinite(count) else ""
            info_lines.append(
                f"{display_name} - {_short_speed_bucket(bucket)}: mean={_format_stat(mean_val)} | median={_format_stat(median_val)}{count_txt}"
            )

    ax.set_xticks(positions, display_names, rotation=20)
    ax.set_xlabel("Carrier")

    normalized, limits = _apply_axis_meta(ax, summary_by_speed_operator["mean"], summary_by_speed_operator["std"])
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_title(
        f"Mean NetworkQuality by Speed Bucket and Carrier {scale_suffix}\n"
        "Inside (indoor) | Outdoor <=0.5 m/s | Outdoor 0.5-5 m/s | Outdoor >5 m/s"
    )

    legend_handles: List[Line2D | Patch] = []
    seen = set()
    for bucket in bucket_order:
        label = _short_speed_bucket(bucket)
        if label in seen:
            continue
        seen.add(label)
        legend_handles.append(
            Patch(facecolor=_speed_color(bucket), edgecolor="black", linewidth=0.6, label=label)
        )
    if median_label_added:
        legend_handles.append(Line2D([0], [0], color="red", linewidth=2.0, label="Median"))
    if legend_handles:
        ax.legend(legend_handles, [h.get_label() for h in legend_handles], frameon=False, loc="best")

    annotation_lines: List[str] = []
    if per_file_speed is not None and {"operator", "speed_bucket"}.issubset(per_file_speed.columns):
        for operator, display_name in zip(operators, display_names):
            inside_samples = per_file_speed.loc[
                (per_file_speed["operator"] == operator) & (per_file_speed["speed_bucket"] == "inside"),
                "mean",
            ]
            for bucket in bucket_order:
                if bucket == "inside":
                    continue
                bucket_samples = per_file_speed.loc[
                    (per_file_speed["operator"] == operator) & (per_file_speed["speed_bucket"] == bucket),
                    "mean",
                ]
                if inside_samples.empty or bucket_samples.empty:
                    continue
                delta = inside_samples.mean() - bucket_samples.mean()
                pct = np.nan
                if np.isfinite(bucket_samples.mean()) and bucket_samples.mean() != 0:
                    pct = delta / bucket_samples.mean()
                p_value = _welch_pvalue(inside_samples, bucket_samples)
                annotation_lines.append(
                    f"{display_name} Inside vs {_pretty_speed_bucket(bucket)}: Δ={delta:.3f} ({_format_percent(pct)}), p={_format_pvalue(p_value)}"
                )

    text_y = 0.95
    if annotation_lines:
        ax.text(
            1.02,
            text_y,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        text_y -= 0.3

    if info_lines:
        ax.text(
            1.02,
            text_y,
            "\n".join(info_lines),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )

    _save_figure(output_path)
    return True


def plot_per_file_box_env(per_file: pd.DataFrame, output_path: Path) -> bool:
    if per_file is None or "env" not in per_file.columns:
        return False
    subset = per_file[per_file["env"].isin(["inside", "outdoor_driving"])]
    if subset.empty:
        return False

    data = [subset[subset["env"] == env]["mean"].dropna() for env in ["inside", "outdoor_driving"]]
    if any(series.empty for series in data):
        return False

    plt.figure(figsize=(6, 4))
    plt.boxplot(
        data,
        labels=[_pretty_env("inside"), _pretty_env("outdoor_driving")],
        showmeans=True,
        meanline=True,
        meanprops={"color": "black", "linewidth": 1.5, "linestyle": "-", "marker": "None"},
        medianprops={"color": "red", "linewidth": 2.0},
    )
    plt.xlabel("Environment")

    ax = plt.gca()
    combined = pd.concat(data, ignore_index=True)
    normalized, limits = _apply_axis_meta(
        ax,
        combined,
        base_label="Per-file Mean NetworkQuality",
    )
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_title(f"Per-file NetworkQuality Distribution by Environment {scale_suffix}")

    inside_samples, outdoor_samples = data
    p_value = _welch_pvalue(inside_samples, outdoor_samples)
    delta = inside_samples.mean() - outdoor_samples.mean()
    pct = np.nan
    if np.isfinite(outdoor_samples.mean()) and outdoor_samples.mean() != 0:
        pct = delta / outdoor_samples.mean()
    ymin, ymax = ax.get_ylim()
    ax.text(
        0.5,
        ymax - (ymax - ymin) * 0.05,
        f"Δ={delta:.3f} ({_format_percent(pct)}) • p={_format_pvalue(p_value)}",
        ha="center",
        va="top",
    )

    stats_lines = [
        f"{label}: mean={_format_stat(series.mean())}, median={_format_stat(series.median())}, n={len(series)}"
        for label, series in zip([_pretty_env("inside"), _pretty_env("outdoor_driving")], data)
    ]
    ax.text(
        1.02,
        0.95,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )

    _add_mean_median_legend(ax, loc="upper right")
    _save_figure(output_path)
    return True


def plot_per_file_box_by_rat(per_file: pd.DataFrame, output_path: Path) -> bool:
    if per_file is None or not {"env", "rat"}.issubset(per_file.columns):
        return False

    rats = sorted(per_file["rat"].dropna().unique())
    if not rats:
        return False

    label_count = sum(
        1 for rat in rats for env in ["inside", "outdoor_driving"] if not per_file[(per_file["rat"] == rat) & (per_file["env"] == env)]["mean"].dropna().empty
    )
    fig_width = max(8, label_count * 0.9)
    plt.figure(figsize=(fig_width, 4.2))
    data: List[pd.Series] = []
    labels: List[str] = []
    for rat in rats:
        for env in ["inside", "outdoor_driving"]:
            subset = per_file[(per_file["rat"] == rat) & (per_file["env"] == env)]["mean"].dropna()
            if subset.empty:
                continue
            data.append(subset)
            labels.append(f"{rat} - {_pretty_env(env)}")

    if not data:
        return False

    plt.boxplot(
        data,
        labels=labels,
        showmeans=True,
        meanline=True,
        meanprops={"color": "black", "linewidth": 1.5, "linestyle": "-", "marker": "None"},
        medianprops={"color": "red", "linewidth": 2.0},
    )
    plt.xlabel("RAT and Environment")
    plt.xticks(rotation=30, ha="right")
    plt.subplots_adjust(bottom=0.28)

    ax = plt.gca()
    combined = pd.concat(data, ignore_index=True)
    normalized, limits = _apply_axis_meta(
        ax,
        combined,
        base_label="Per-file Mean NetworkQuality",
    )
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_title(f"Per-file NetworkQuality Distribution by RAT and Environment {scale_suffix}")

    annotation_lines = [
        f"{label}: mean={_format_stat(series.mean())}, median={_format_stat(series.median())}, σ={_format_stat(series.std(ddof=1))}, n={len(series)}"
        for label, series in zip(labels, data)
    ]
    ax.text(
        1.02,
        0.95,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )

    _add_mean_median_legend(ax, loc="upper left")
    _save_figure(output_path)
    return True


def plot_ecdf_env_by_rat(per_file: pd.DataFrame, output_path: Path) -> bool:
    if per_file is None or not {"env", "rat"}.issubset(per_file.columns):
        return False

    rats = sorted(per_file["rat"].dropna().unique())
    if not rats:
        return False

    plt.figure(figsize=(7.5, 4.2))
    series_bundle: List[pd.Series] = []
    for rat in rats:
        for env in ["inside", "outdoor_driving"]:
            subset = per_file[(per_file["rat"] == rat) & (per_file["env"] == env)]["mean"].dropna()
            if subset.empty:
                continue
            sorted_vals = np.sort(subset)
            ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            plt.step(sorted_vals, ecdf, where="post", label=f"{rat} - {_pretty_env(env)}")
            series_bundle.append(pd.Series(sorted_vals))

    if not series_bundle:
        return False

    ax = plt.gca()
    normalized, limits = _apply_axis_meta(
        ax,
        pd.concat(series_bundle, ignore_index=True),
        base_label="Per-file Mean NetworkQuality",
        axis="x",
    )
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_ylabel("ECDF (0-1)")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right")
    ax.set_title(f"Per-file ECDF by RAT and Environment {scale_suffix}")

    _save_figure(output_path)
    return True


def plot_scatter_file_mean_vs_std(per_file: pd.DataFrame, output_path: Path) -> bool:
    if per_file is None or not {"mean", "std"}.issubset(per_file.columns):
        return False
    if per_file.empty:
        return False

    plt.figure(figsize=(6, 4))
    jitter = np.random.default_rng(42).normal(scale=0.002, size=len(per_file))
    plt.scatter(per_file["mean"], per_file["std"] + jitter, alpha=0.7)

    ax = plt.gca()
    mean_normalized, mean_limits = _apply_axis_meta(
        ax,
        per_file["mean"],
        base_label="Per-file Mean NetworkQuality",
        axis="x",
    )
    std_normalized, std_limits = _apply_axis_meta(
        ax,
        per_file["std"],
        base_label="Per-file Std of NetworkQuality",
        axis="y",
    )
    scale_suffix = "(normalized)" if mean_normalized else "(raw)"
    ax.set_title(f"Per-file Mean vs Std {scale_suffix}")

    _save_figure(output_path)
    return True


def plot_kpi_presence(kpi_presence: pd.DataFrame, output_path: Path) -> bool:
    if kpi_presence is None or "kpi" not in kpi_presence.columns:
        return False

    sorted_df = kpi_presence.sort_values("files_with_kpi", ascending=True)
    plt.figure(figsize=(8, 4))
    y_positions = np.arange(len(sorted_df))
    plt.barh(y_positions, sorted_df["files_with_kpi"])
    plt.yticks(y_positions, sorted_df["kpi"])
    plt.xlabel("Files with KPI (count)")
    plt.title("KPI Availability (File Counts)")
    _save_figure(output_path)
    return True


def plot_per_file_violin_env(per_file: pd.DataFrame, output_path: Path) -> bool:
    if per_file is None or "env" not in per_file.columns:
        return False

    groups = ["inside", "outdoor_driving"]
    data = [per_file.loc[per_file["env"] == env, "mean"].dropna() for env in groups]
    if any(series.empty for series in data):
        return False

    plt.figure(figsize=(7.5, 4.2))
    parts = plt.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
    _apply_violin_colors(parts, [_env_color(env) for env in groups])
    if "cmeans" in parts:
        parts["cmeans"].set_color("black")
        parts["cmeans"].set_linewidth(1.5)
    if "cmedians" in parts:
        parts["cmedians"].set_color("red")
        parts["cmedians"].set_linewidth(2.0)
    plt.xticks(
        np.arange(1, len(groups) + 1),
        [_pretty_env(env) for env in groups],
    )
    plt.xlabel("Environment")
    plt.subplots_adjust(bottom=0.22)

    ax = plt.gca()
    combined = pd.concat(data, ignore_index=True)
    normalized, limits = _apply_axis_meta(
        ax,
        combined,
        base_label="Per-file Mean NetworkQuality",
    )
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_title(f"Per-file NetworkQuality Violin by Environment {scale_suffix}")

    inside_samples, outdoor_samples = data
    p_value = _welch_pvalue(inside_samples, outdoor_samples)
    delta = inside_samples.mean() - outdoor_samples.mean()
    pct = np.nan
    if np.isfinite(outdoor_samples.mean()) and outdoor_samples.mean() != 0:
        pct = delta / outdoor_samples.mean()
    ymin, ymax = ax.get_ylim()
    ax.text(
        0.5,
        ymax - (ymax - ymin) * 0.05,
        f"Δ={delta:.3f} ({_format_percent(pct)}) • p={_format_pvalue(p_value)}",
        ha="center",
        va="top",
    )

    stats_lines = [
        f"{_pretty_env(env)}: mean={_format_stat(series.mean())}, median={_format_stat(series.median())}, n={len(series)}"
        for env, series in zip(groups, data)
    ]
    ax.text(
        1.02,
        0.95,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )

    _add_mean_median_legend(ax, loc="upper right")
    _save_figure(output_path)
    return True


def plot_per_file_violin_by_rat(per_file: pd.DataFrame, output_path: Path) -> bool:
    if per_file is None or not {"env", "rat"}.issubset(per_file.columns):
        return False

    combinations: List[str] = []
    combo_envs: List[str] = []
    data: List[pd.Series] = []
    for rat in sorted(per_file["rat"].dropna().unique()):
        for env in ["inside", "outdoor_driving"]:
            subset = per_file[(per_file["rat"] == rat) & (per_file["env"] == env)]["mean"].dropna()
            if subset.empty:
                continue
            combinations.append(f"{rat} - {_pretty_env(env)}")
            combo_envs.append(env)
            data.append(subset)

    if not data:
        return False

    plt.figure(figsize=(max(8, len(data) * 1.5), 4))
    parts = plt.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
    _apply_violin_colors(parts, [_env_color(env) for env in combo_envs])
    if "cmeans" in parts:
        parts["cmeans"].set_color("black")
        parts["cmeans"].set_linewidth(1.5)
    if "cmedians" in parts:
        parts["cmedians"].set_color("red")
        parts["cmedians"].set_linewidth(2.0)
    plt.xticks(np.arange(1, len(combinations) + 1), combinations, rotation=30)
    plt.xlabel("RAT and Environment")

    ax = plt.gca()
    combined = pd.concat(data, ignore_index=True)
    normalized, limits = _apply_axis_meta(
        ax,
        combined,
        base_label="Per-file Mean NetworkQuality",
    )
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_title(f"Per-file NetworkQuality Violin by RAT and Environment {scale_suffix}")

    annotation_lines = [
        f"{combo}: μ={series.mean():.3f}, σ={series.std(ddof=1):.3f}, n={len(series)}"
        for combo, series in zip(combinations, data)
    ]
    ax.text(
        1.02,
        0.95,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )

    _add_mean_median_legend(ax, loc="upper left")
    _save_figure(output_path)
    return True


def plot_per_file_violin_operator(
    per_file: pd.DataFrame,
    operator_metadata: Optional[pd.DataFrame],
    output_path: Path,
) -> bool:
    if per_file is None or "operator" not in per_file.columns:
        return False

    operator_mapping = _build_operator_mapping(operator_metadata)

    operators = sorted(per_file["operator"].dropna().unique())
    operator_env_info: List[Tuple[str, str, str]] = []  # (operator_code, env_key, label)
    data: List[pd.Series] = []
    colors: List[str] = []

    for operator in operators:
        display_name = operator_mapping.get(operator, operator)
        for env_key, env_label in [("inside", "Indoor"), ("outdoor_driving", "Outdoor")]:
            subset = per_file[
                (per_file["operator"] == operator) & (per_file["env"] == env_key)
            ]["mean"].dropna()
            if subset.empty:
                continue
            operator_env_info.append((operator, env_key, f"{display_name}\n{env_label}"))
            data.append(subset)
            colors.append(_operator_env_color(display_name, env_key))

    if not data:
        return False

    labels = [label for _, _, label in operator_env_info]
    plt.figure(figsize=(max(8, len(labels) * 0.85), 4.2))
    parts = plt.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
    _apply_violin_colors(parts, colors)
    if "cmeans" in parts:
        parts["cmeans"].set_color("black")
        parts["cmeans"].set_linewidth(1.5)
    if "cmedians" in parts:
        parts["cmedians"].set_color("red")
        parts["cmedians"].set_linewidth(2.0)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=25, ha="right")
    plt.xlabel("Operator and Environment")
    plt.subplots_adjust(bottom=0.3)

    ax = plt.gca()
    combined = pd.concat(data, ignore_index=True)
    normalized, limits = _apply_axis_meta(
        ax,
        combined,
        base_label="Per-file Mean NetworkQuality",
    )
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_title(f"Per-file NetworkQuality Violin by Operator {scale_suffix}")

    if {"env"}.issubset(per_file.columns):
        annotation_lines: List[str] = []
        for operator in operators:
            display_name = operator_mapping.get(operator, operator)
            inside_samples = per_file.loc[
                (per_file["operator"] == operator) & (per_file["env"] == "inside"),
                "mean",
            ]
            outdoor_samples = per_file.loc[
                (per_file["operator"] == operator) & (per_file["env"] == "outdoor_driving"),
                "mean",
            ]
            if inside_samples.empty or outdoor_samples.empty:
                continue
            delta = inside_samples.mean() - outdoor_samples.mean()
            pct = np.nan
            if np.isfinite(outdoor_samples.mean()) and outdoor_samples.mean() != 0:
                pct = delta / outdoor_samples.mean()
            p_value = _welch_pvalue(inside_samples, outdoor_samples)
            annotation_lines.append(
                f"{display_name}: Δ={delta:.3f} ({_format_percent(pct)}), p={_format_pvalue(p_value)}"
            )
        if annotation_lines:
            ax.text(
                1.02,
                0.95,
                "\n".join(annotation_lines),
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )

    stats_lines = [
        f"{label.replace(chr(10), ' ')}: mean={_format_stat(series.mean())}, median={_format_stat(series.median())}, n={len(series)}"
        for label, series in zip(labels, data)
    ]
    ax.text(
        1.02,
        0.75,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )

    _add_mean_median_legend(ax, loc="upper left")
    _save_figure(output_path)
    return True


def plot_speed_bucket_violin(per_file_speed: pd.DataFrame, output_path: Path) -> bool:
    if per_file_speed is None or "speed_bucket" not in per_file_speed.columns:
        return False

    order = ["inside", "od_quasi_static", "od_slow", "od_fast"]
    data = []
    bucket_keys: List[str] = []
    display_labels: List[str] = []
    for bucket in order:
        subset = per_file_speed[per_file_speed["speed_bucket"] == bucket]["mean"].dropna()
        if subset.empty:
            continue
        bucket_keys.append(bucket)
        display_labels.append(_pretty_speed_bucket(bucket))
        data.append(subset)

    if not data:
        return False

    plt.figure(figsize=(7, 4))
    parts = plt.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
    _apply_violin_colors(parts, [_speed_color(bucket) for bucket in bucket_keys])
    if "cmeans" in parts:
        parts["cmeans"].set_color("black")
        parts["cmeans"].set_linewidth(1.5)
    if "cmedians" in parts:
        parts["cmedians"].set_color("red")
        parts["cmedians"].set_linewidth(2.0)
    plt.xticks(np.arange(1, len(display_labels) + 1), display_labels, rotation=20, ha="right")
    plt.xlabel("Speed Bucket")
    plt.subplots_adjust(bottom=0.22)

    ax = plt.gca()
    combined = pd.concat(data, ignore_index=True)
    normalized, limits = _apply_axis_meta(
        ax,
        combined,
        base_label="Per-file Mean NetworkQuality",
    )
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_title(
        f"Per-file NetworkQuality Violin by Speed Bucket {scale_suffix}\n"
        "Inside (indoor); Outdoor <=0.5 m/s; Outdoor 0.5-5 m/s; Outdoor >5 m/s"
    )

    text_y = 0.95
    if "inside" in bucket_keys:
        inside_index = bucket_keys.index("inside")
        inside_samples = data[inside_index]
        annotation_lines: List[str] = []
        for bucket, display, sample in zip(bucket_keys, display_labels, data):
            if bucket == "inside":
                continue
            delta = inside_samples.mean() - sample.mean()
            pct = np.nan
            if np.isfinite(sample.mean()) and sample.mean() != 0:
                pct = delta / sample.mean()
            p_value = _welch_pvalue(inside_samples, sample)
            annotation_lines.append(
                f"Inside vs {display}: Δ={delta:.3f} ({_format_percent(pct)}), p={_format_pvalue(p_value)}"
            )
        if annotation_lines:
            ax.text(
                1.02,
                text_y,
                "\n".join(annotation_lines),
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
            text_y -= 0.25

    stats_lines = [
        f"{label}: mean={_format_stat(series.mean())} | median={_format_stat(series.median())}"
        for label, series in zip(display_labels, data)
    ]
    if stats_lines:
        ax.text(
            1.02,
            text_y,
            "\n".join(stats_lines),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )

    _save_figure(output_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate matplotlib plots from passive quality summaries."
    )
    parser.add_argument(
        "--results_dir",
        default="./results_normalized",
        help="Directory containing CSV outputs from the upstream analysis.",
    )
    parser.add_argument(
        "--save_all",
        action="store_true",
        default=True,
        help="If set, save all plots defined; default behaviour is to save all.",
    )
    parser.add_argument(
        "--skip_ecdf",
        action="store_true",
        default=False,
        help="If set, skip ECDF plots (useful if per_file_stats.csv is large or missing).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    try:
        data = load_data(results_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    created: List[str] = []
    skipped: List[str] = []

    operator_metadata = data.get("operator_metadata")

    output_map = {
        "plot_overall_mean_bar.png": lambda: plot_overall_mean_bar(
            data["overall"], data["per_file"], results_dir / "plot_overall_mean_bar.png"
        ),
        "plot_by_rat_mean_bar.png": lambda: plot_by_rat_mean_bar(
            data["by_rat"], data["per_file"], results_dir / "plot_by_rat_mean_bar.png"
        ),
        "plot_by_carrier_mean_bar.png": lambda: plot_by_operator_mean_bar(
            data["by_operator"], data["per_file"], operator_metadata, results_dir / "plot_by_carrier_mean_bar.png"
        ),
        "plot_by_rat_carrier_mean_bar.png": lambda: plot_by_rat_operator_mean_bar(
            data["by_rat_operator"], data["per_file"], operator_metadata, results_dir / "plot_by_rat_carrier_mean_bar.png"
        ),
        "plot_speed_buckets_bar.png": lambda: plot_speed_buckets_bar(
            data["by_speed"], data["per_file_speed"], results_dir / "plot_speed_buckets_bar.png"
        ),
        "plot_speed_buckets_carrier_bar.png": lambda: plot_speed_buckets_operator_bar(
            data["by_speed_operator"], data["per_file_speed"], operator_metadata, results_dir / "plot_speed_buckets_carrier_bar.png"
        ),
        "plot_per_file_box_env.png": lambda: plot_per_file_box_env(
            data["per_file"], results_dir / "plot_per_file_box_env.png"
        ),
        "plot_per_file_box_by_rat.png": lambda: plot_per_file_box_by_rat(
            data["per_file"], results_dir / "plot_per_file_box_by_rat.png"
        ),
        "plot_ecdf_env_by_rat.png": lambda: plot_ecdf_env_by_rat(
            data["per_file"], results_dir / "plot_ecdf_env_by_rat.png"
        ),
        "plot_scatter_file_mean_vs_std.png": lambda: plot_scatter_file_mean_vs_std(
            data["per_file"], results_dir / "plot_scatter_file_mean_vs_std.png"
        ),
        "plot_kpi_presence.png": lambda: plot_kpi_presence(
            data["kpi_presence"], results_dir / "plot_kpi_presence.png"
        ),
        "plot_per_file_violin_env.png": lambda: plot_per_file_violin_env(
            data["per_file"], results_dir / "plot_per_file_violin_env.png"
        ),
        "plot_per_file_violin_by_rat.png": lambda: plot_per_file_violin_by_rat(
            data["per_file"], results_dir / "plot_per_file_violin_by_rat.png"
        ),
        "plot_per_file_violin_carrier.png": lambda: plot_per_file_violin_operator(
            data["per_file"], operator_metadata, results_dir / "plot_per_file_violin_carrier.png"
        ),
        "plot_speed_bucket_violin.png": lambda: plot_speed_bucket_violin(
            data["per_file_speed"], results_dir / "plot_speed_bucket_violin.png"
        ),
    }

    for filename, plot_func in output_map.items():
        if filename == "plot_ecdf_env_by_rat.png" and args.skip_ecdf:
            skipped.append(filename)
            continue
        success = plot_func()
        if success:
            created.append(filename)
        else:
            skipped.append(filename)

    if created:
        print("Plots created:")
        for name in created:
            print(f"  - {name}")
    else:
        print("No plots created.")

    if skipped:
        print("Plots skipped or unavailable inputs:")
        for name in skipped:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
