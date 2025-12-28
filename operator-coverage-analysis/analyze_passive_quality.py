#!/usr/bin/env python3
"""
This script analyzes passive cellular measurement logs to quantify RSRP quality
differences between Indoor Static (IS) and Outdoor Driving (OD) environments across
4G and 5G. It discovers files from the 4G/ and 5G/ folders, excludes Active datasets
and archives, selects consistent RF KPIs, performs robust per-RAT normalization
(5th–95th percentile), computes a percentile-scaled RSRPQuality score, assigns OD speed buckets,
aggregates at the file level to avoid file-size bias, and outputs summary statistics
and optional plots.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover - optional dependency
    stats = None


KPI_PRIORITY: List[str] = ["SSS-RSRP", "SS_PBCH-RSRP", "DM_RS-RSRP", "RSRP", "PSS-RSRP"]
KPI_FALLBACK: List[str] = []
OPTIONAL_COLUMNS: List[str] = [
    "Speed",
    "Band",
    "Latitude",
    "Longitude",
    "scenario",
    "operator",
    "Operator",
    "MNC",
]

OPERATOR_COLUMN_CANDIDATES = ["operator", "Operator", "OPERATOR"]
OPERATOR_TV_LABEL = "TIM/Vodafone"
OPERATOR_IW_LABEL = "Iliad/Wind"
OPERATOR_FILENAME_PATTERNS = [
    (re.compile(r"[_\-]tv[_\-]", re.IGNORECASE), OPERATOR_TV_LABEL),
    (re.compile(r"[_\-]iw[_\-]", re.IGNORECASE), OPERATOR_IW_LABEL),
]

IGNORE_DIR_NAMES = {"Active", "__MACOSX"}
IGNORE_FILES = {
    "4G_2023_passive.zip",
    "5G_2023_passive.zip",
    "Active Performance Dataset - 1.zip",
}

ENV_PATTERNS = {
    "IS": re.compile(r"(?i)(?:^|[^a-z0-9])is(?:[^a-z0-9]|$)"),
    "OD": re.compile(r"(?i)(?:^|[^a-z0-9])od(?:[^a-z0-9]|$)"),
    "OW": re.compile(r"(?i)(?:^|[^a-z0-9])ow(?:[^a-z0-9]|$)"),
}

ENV_MAPPING = {"IS": "inside", "OD": "outdoor_driving", "OW": "outside_walking"}
INDOOR_LABEL = "inside"
OD_BUCKET_LABELS = ["od_quasi_static", "od_slow", "od_fast"]
QUALITY_COLUMN = "RSRPQuality"
MEAN_COLOR = "#000000"
STD_COLOR = "#f2b200"
MEDIAN_COLOR = "#c62828"


@dataclass
class LoadResult:
    dataframe: pd.DataFrame
    kpi_columns: List[str]
    kpi_presence: pd.DataFrame


def select_kpis(columns: Sequence[str]) -> List[str]:
    """Select the preferred RSRP column if present (single metric)."""
    for candidate in KPI_PRIORITY:
        if candidate in columns:
            return [candidate]
    for fallback in KPI_FALLBACK:
        if fallback in columns:
            return [fallback]
    return []


def infer_environment(file_path: Path) -> str | None:
    """Infer environment class (IS/OD/OW) from the filename using configured patterns."""
    stem = file_path.stem
    for tag, pattern in ENV_PATTERNS.items():
        if pattern.search(stem):
            return tag
    return None


def _map_operator_group(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    text = str(label).strip().lower()
    if not text:
        return None

    if any(key in text for key in ["tim", "vodafone", "tv"]):
        return OPERATOR_TV_LABEL
    if any(key in text for key in ["iliad", "wind", "iw"]):
        return OPERATOR_IW_LABEL

    if re.search(r"op[^0-9]*1", text) or re.search(r"operator[^0-9]*1", text) or re.search(r"op[^0-9]*2", text) or re.search(
        r"operator[^0-9]*2", text
    ):
        return OPERATOR_TV_LABEL
    if re.search(r"op[^0-9]*3", text) or re.search(r"operator[^0-9]*3", text) or re.search(r"op[^0-9]*4", text) or re.search(
        r"operator[^0-9]*4", text
    ):
        return OPERATOR_IW_LABEL

    return None


def _operator_from_filename(file_path: Path) -> Optional[str]:
    stem = file_path.stem.lower()
    for pattern, label in OPERATOR_FILENAME_PATTERNS:
        if pattern.search(stem):
            return label
    parent = file_path.parent.name.lower()
    for pattern, label in OPERATOR_FILENAME_PATTERNS:
        if pattern.search(parent):
            return label
    if "_tv" in stem or "-tv" in stem:
        return OPERATOR_TV_LABEL
    if "_iw" in stem or "-iw" in stem:
        return OPERATOR_IW_LABEL
    if "tv" in parent:
        return OPERATOR_TV_LABEL
    if "iw" in parent:
        return OPERATOR_IW_LABEL
    return None


def _clean_operator_value(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().strip('"').strip("'")
    if not text or text.lower() in {"nan", "none"}:
        return None
    mapped = _map_operator_group(text)
    if mapped:
        return mapped
    bracket_match = re.search(r"\[([^\]]+)\]", text)
    prefix_match = re.findall(r"[A-Za-z]+", text)
    if bracket_match:
        numeric_component = bracket_match.group(1).strip()
        prefix = prefix_match[-1].capitalize() if prefix_match else None
        if prefix:
            mapped = _map_operator_group(prefix)
            if mapped:
                return mapped
            return f"{prefix} {numeric_component}"
        if numeric_component.isdigit():
            mapped = _map_operator_group(f"Operator {numeric_component}")
            if mapped:
                return mapped
            return f"Operator {numeric_component}"
        return numeric_component
    text = re.sub(r"[^0-9A-Za-z]+", " ", text).strip()
    if not text:
        return None
    tokens = text.split()
    if len(tokens) == 1:
        token = tokens[0]
        if token.isdigit():
            mapped = _map_operator_group(f"Operator {token}")
            if mapped:
                return mapped
            return f"Operator {token}"
        mapped = _map_operator_group(token)
        if mapped:
            return mapped
        return token.capitalize()
    # combine alpha tokens with numeric suffix if present
    alpha_tokens = [tok for tok in tokens if tok.isalpha()]
    digit_tokens = [tok for tok in tokens if tok.isdigit()]
    if alpha_tokens and digit_tokens:
        mapped = _map_operator_group(alpha_tokens[-1])
        if mapped:
            return mapped
        return f"{alpha_tokens[-1].capitalize()} {digit_tokens[0]}"
    joined = " ".join(tokens)
    mapped = _map_operator_group(joined)
    if mapped:
        return mapped
    return joined


def infer_operator(df_raw: pd.DataFrame, file_path: Path) -> str:
    candidate = _operator_from_filename(file_path)
    if candidate:
        return candidate

    if "scenario" in df_raw.columns:
        scenario_values = df_raw["scenario"].dropna().astype(str).str.lower()
        for value in scenario_values:
            mapped = _map_operator_group(value)
            if mapped:
                return mapped

    for col in df_raw.columns:
        if col.lower() == "operator":
            value = df_raw[col].dropna().head(1)
            if not value.empty:
                cleaned = _clean_operator_value(value.iloc[0])
                if cleaned:
                    return cleaned

    for candidate in OPERATOR_COLUMN_CANDIDATES:
        if candidate in df_raw.columns:
            value = df_raw[candidate].dropna().head(1)
            if not value.empty:
                cleaned = _clean_operator_value(value.iloc[0])
                if cleaned:
                    return cleaned

    if "MNC" in df_raw.columns:
        value = df_raw["MNC"].dropna().head(1)
        if not value.empty:
            cleaned = _clean_operator_value(value.iloc[0])
            if cleaned:
                return cleaned

    stem = file_path.stem
    mapped = _map_operator_group(stem)
    if mapped:
        return mapped

    parent = file_path.parent.name
    mapped = _map_operator_group(parent)
    if mapped:
        return mapped

    return "unknown"


def is_ignored(path: Path) -> bool:
    """Return True if the path should be ignored due to folder or filename rules."""
    if any(part in IGNORE_DIR_NAMES for part in path.parts):
        return True
    if path.name in IGNORE_FILES:
        return True
    if path.suffix.lower() != ".csv":
        return True
    return False


def load_passive_files(
    five_g_dir: Path,
    include_ow: bool,
) -> LoadResult:
    """
    Load eGaming CSVs from 5G/ only. Infer env from filename (is/od/optional ow),
    rat from folder. Keep KPI columns, Speed, Band, scenario if present. Return a
    single DataFrame with columns: chosen KPIs + env + rat + file + Speed (if exists).
    """
    frames: List[pd.DataFrame] = []
    all_kpis: List[str] = []
    kpi_presence_files: Dict[str, set[str]] = {}
    kpi_presence_rows: Dict[str, int] = {}

    root_dir = five_g_dir.parent

    for rat_dir, rat_label in ((five_g_dir, "5G"),):
        if not rat_dir.exists():
            continue

        for csv_path in sorted(rat_dir.rglob("*.csv")):
            if "egaming" not in csv_path.stem.lower():
                continue
            if is_ignored(csv_path):
                continue

            env_tag = infer_environment(csv_path)
            if env_tag is None:
                continue
            if env_tag == "OW" and not include_ow:
                continue

            env_label = ENV_MAPPING[env_tag]

            try:
                df_raw = pd.read_csv(csv_path)
                # According to the dataset documentation, "?" repeats the previous
                # measurement rather than representing a missing value. Replace
                # the markers with the last observed value in-column.
                if (df_raw == "?").any().any():
                    df_raw = df_raw.replace("?", pd.NA).ffill()
                    for column in df_raw.columns:
                        if df_raw[column].dtype == object:
                            try:
                                df_raw[column] = pd.to_numeric(df_raw[column])
                            except (ValueError, TypeError):
                                pass
            except Exception as exc:  # pragma: no cover - IO guard
                print(f"Warning: failed to read {csv_path}: {exc}")
                continue

            kpis = select_kpis(df_raw.columns)
            if not kpis:
                continue

            target_kpi = kpis[0]
            keep_cols = [target_kpi]
            keep_cols.extend(col for col in OPTIONAL_COLUMNS if col in df_raw.columns)
            df_subset = df_raw.loc[:, keep_cols].copy()
            df_subset[target_kpi] = pd.to_numeric(df_subset[target_kpi], errors="coerce")
            df_subset.dropna(subset=[target_kpi], how="all", inplace=True)
            if df_subset.empty:
                continue

            df_subset.rename(columns={target_kpi: "RSRP"}, inplace=True)

            operator_label = infer_operator(df_raw, csv_path)
            df_subset["operator"] = operator_label

            kpi_presence_files.setdefault("RSRP", set()).add(csv_path.name)
            kpi_presence_rows["RSRP"] = kpi_presence_rows.get("RSRP", 0) + int(
                df_subset["RSRP"].notna().sum()
            )

            df_subset["env"] = env_label
            df_subset["rat"] = rat_label
            try:
                file_label = str(csv_path.relative_to(root_dir))
            except ValueError:
                file_label = str(csv_path)
            df_subset["file"] = file_label

            frames.append(df_subset)
            if "RSRP" not in all_kpis:
                all_kpis.append("RSRP")

    if not frames:
        empty_df = pd.DataFrame(
            columns=["env", "rat", "file", "RSRP"] + OPTIONAL_COLUMNS
        )
        presence_df = pd.DataFrame(columns=["kpi", "files_with_kpi", "rows_with_kpi"])
        return LoadResult(empty_df, [], presence_df)

    combined = pd.concat(frames, ignore_index=True, sort=False)

    ordered_kpis: List[str] = ["RSRP"] if "RSRP" in combined.columns else []

    for col in ordered_kpis:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    optional_present = [col for col in OPTIONAL_COLUMNS if col in combined.columns]
    meta_columns = ["operator", "env", "rat", "file"]
    optional_non_meta = [col for col in optional_present if col not in meta_columns]
    ordered_columns = ordered_kpis + optional_non_meta + meta_columns
    ordered_columns = [col for col in ordered_columns if col in combined.columns]
    combined = combined.loc[:, ordered_columns]

    presence_records = []
    for kpi in ordered_kpis:
        presence_records.append(
            {
                "kpi": kpi,
                "files_with_kpi": len(kpi_presence_files.get(kpi, set())),
                "rows_with_kpi": kpi_presence_rows.get(kpi, 0),
            }
        )
    presence_df = pd.DataFrame(presence_records)

    return LoadResult(combined, ordered_kpis, presence_df)


def assign_speed_buckets(
    df: pd.DataFrame, thresholds: Tuple[float, float], indoor_label: str = INDOOR_LABEL
) -> pd.DataFrame:
    """Create `speed_bucket` for OD rows using thresholds; label IS rows as `inside`."""
    df = df.copy()
    low, high = thresholds

    if "Speed" in df.columns:
        speed = pd.to_numeric(df["Speed"], errors="coerce")
    else:
        speed = pd.Series(np.nan, index=df.index)

    speed_bucket = pd.Series(indoor_label, index=df.index, dtype=object)
    od_mask = df["env"] == "outdoor_driving"

    speed_bucket.loc[od_mask & (speed.isna() | (speed <= low))] = OD_BUCKET_LABELS[0]
    speed_bucket.loc[od_mask & (speed > low) & (speed <= high)] = OD_BUCKET_LABELS[1]
    speed_bucket.loc[od_mask & (speed > high)] = OD_BUCKET_LABELS[2]

    ow_mask = df["env"] == "outside_walking"
    speed_bucket.loc[ow_mask] = "outside_walking"

    df["speed_bucket"] = speed_bucket
    return df



def compute_raw_mean(df: pd.DataFrame, kpi_columns: Sequence[str]) -> pd.Series:
    """Compute mean of raw KPI values for fallback non-normalized score."""
    series = df[kpi_columns].mean(axis=1, skipna=True)
    series[df[kpi_columns].isna().all(axis=1)] = np.nan
    return series


def _series_limits(series: pd.Series) -> Optional[Tuple[float, float]]:
    finite = pd.to_numeric(series, errors="coerce").dropna()
    if finite.empty:
        return None
    return float(finite.min()), float(finite.max())


def _series_is_normalized(series: pd.Series) -> bool:
    finite = pd.to_numeric(series, errors="coerce").dropna()
    if finite.empty:
        return True
    q_low = float(finite.quantile(0.01))
    q_high = float(finite.quantile(0.99))
    return q_low >= -0.2 and q_high <= 1.2


def _expanded_limits_from_series(limits: Optional[Tuple[float, float]], normalized: bool) -> Optional[Tuple[float, float]]:
    if limits is None:
        return None
    low, high = limits
    if normalized:
        low_lim = min(0.0, low - 0.05)
        high_lim = max(1.0, high + 0.05)
        return low_lim, high_lim
    span = high - low
    if span == 0:
        padding = max(1.0, max(abs(high), abs(low), 1.0) * 0.1)
    else:
        padding = max(1.0, abs(span) * 0.1)
    return low - padding, high + padding


def _axis_label_from_series(base: str, normalized: bool, limits: Optional[Tuple[float, float]]) -> str:
    if normalized:
        return f"{base} (0-1)"
    if limits:
        return f"{base} (raw range {limits[0]:.1f}..{limits[1]:.1f})"
    return f"{base} (raw units)"


def _dunn_posthoc(groups: Sequence[np.ndarray], labels: Sequence[str]) -> List[Dict[str, Any]]:
    """Perform Dunn's post-hoc test with Bonferroni correction."""
    if stats is None or len(groups) < 2:
        return []

    total_n = sum(len(group) for group in groups)
    if total_n <= 1:
        return []

    combined = np.concatenate(groups)
    ranks = stats.rankdata(combined)

    tie_correction = 0.0
    if total_n > 1:
        _, counts = np.unique(combined, return_counts=True)
        tie_correction = float(np.sum(counts**3 - counts))

    base = total_n * (total_n + 1.0) / 12.0
    denom_base = base
    if total_n > 1:
        denom_base -= tie_correction / (12.0 * total_n * (total_n - 1.0))
    if denom_base <= 0 or not np.isfinite(denom_base):
        return []

    group_ranks: List[np.ndarray] = []
    offset = 0
    for group in groups:
        size = len(group)
        group_ranks.append(ranks[offset : offset + size])
        offset += size

    comparisons = len(groups) * (len(groups) - 1) // 2
    if comparisons == 0:
        return []

    results: List[Dict[str, Any]] = []
    for i, j in combinations(range(len(groups)), 2):
        ni = len(groups[i])
        nj = len(groups[j])
        if ni == 0 or nj == 0:
            continue

        se = np.sqrt(denom_base * (1.0 / ni + 1.0 / nj))
        if se == 0 or not np.isfinite(se):
            continue

        z_stat = (group_ranks[i].mean() - group_ranks[j].mean()) / se
        p_raw = 2.0 * stats.norm.sf(abs(z_stat))
        p_adj = min(1.0, p_raw * comparisons)
        results.append(
            {
                "groups": (labels[i], labels[j]),
                "z": float(z_stat),
                "p_value": float(p_raw),
                "p_adjusted": float(p_adj),
                "significant": bool(p_adj < 0.05),
            }
        )

    return results


def perform_nonparametric_tests(
    data_series: Sequence[pd.Series],
    labels: Sequence[str],
    context: str,
) -> Optional[Dict[str, Any]]:
    """Run Mann-Whitney or Kruskal-Wallis + Dunn tests for the provided groups."""
    if stats is None:
        return None

    cleaned: List[Tuple[str, np.ndarray]] = []
    for label, series in zip(labels, data_series):
        numeric = pd.to_numeric(series, errors="coerce")
        values = numeric.dropna().to_numpy()
        if values.size > 0:
            cleaned.append((label, values))

    if len(cleaned) < 2:
        return None

    arrays = [values for _, values in cleaned]
    final_labels = [label for label, _ in cleaned]

    result: Dict[str, Any] = {
        "context": context,
        "group_sizes": {label: int(values.size) for label, values in cleaned},
    }

    if len(arrays) == 2:
        u_res = stats.mannwhitneyu(arrays[0], arrays[1], alternative="two-sided")
        if hasattr(u_res, "statistic"):
            u_stat = float(u_res.statistic)
            p_value = float(u_res.pvalue)
        else:
            u_stat, p_value = map(float, u_res)
        result["primary_test"] = {
            "name": "Mann-Whitney U",
            "statistic": u_stat,
            "p_value": p_value,
        }
        return result

    kw_res = stats.kruskal(*arrays)
    if hasattr(kw_res, "statistic"):
        h_stat = float(kw_res.statistic)
        p_value = float(kw_res.pvalue)
    else:
        h_stat, p_value = map(float, kw_res)

    result["primary_test"] = {
        "name": "Kruskal-Wallis H",
        "statistic": h_stat,
        "p_value": p_value,
    }

    posthoc = _dunn_posthoc(arrays, final_labels)
    if posthoc:
        result["posthoc"] = posthoc
        result["posthoc_method"] = "Bonferroni"

    return result


def write_stat_results_md(
    results_dir: Path,
    stat_results: Sequence[Dict[str, Any]],
    stats_available: bool,
) -> None:
    """Write statistical test outcomes to results/stat_tests.md."""
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "stat_tests.md"

    lines: List[str] = ["# Statistical Test Results\n"]
    if not stats_available:
        lines.append("\nSciPy is not available; statistical tests were skipped.\n")
    elif not stat_results:
        lines.append("\nNo eligible groupings were found for statistical testing.\n")
    else:
        for entry in stat_results:
            lines.append(f"\n## {entry['context']}\n\n")
            group_info = ", ".join(
                f"{label} (n={size})" for label, size in entry["group_sizes"].items()
            )
            lines.append(f"- Groups: {group_info}\n")
            primary = entry["primary_test"]
            lines.append(
                f"- {primary['name']}: statistic={primary['statistic']:.3f}, "
                f"p={primary['p_value']:.4g}\n"
            )

            posthoc = entry.get("posthoc")
            if posthoc:
                method = entry.get("posthoc_method", "Bonferroni")
                lines.append(f"- Post-hoc ({method} correction):\n")
                for comp in posthoc:
                    status = "significant" if comp["significant"] else "ns"
                    lines.append(
                        f"  - {comp['groups'][0]} vs {comp['groups'][1]}: "
                        f"z={comp['z']:.3f}, p_adj={comp['p_adjusted']:.4g} ({status})\n"
                    )

    output_path.write_text("".join(lines), encoding="utf-8")


# --- RSRP quality normalization helpers ---
def _normalize_percentile(series: pd.Series, low_q: float = 0.05, high_q: float = 0.95) -> pd.Series:
    """Percentile-based logistic normalization to the open interval (0, 1).
    Values near the chosen percentile window map close to 0 and 1 without piling up
    at the bounds, preserving smooth differences in the extremes.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return pd.Series(np.nan, index=s.index)
    q_low = float(s.quantile(low_q))
    q_high = float(s.quantile(high_q))
    if not np.isfinite(q_low) or not np.isfinite(q_high) or q_high == q_low:
        return pd.Series(np.nan, index=s.index)

    center = 0.5 * (q_low + q_high)
    scale = 0.5 * (q_high - q_low)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(s.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            return pd.Series(np.nan, index=s.index)

    z = (s - center) / scale
    z = np.clip(z, -60.0, 60.0)
    scaled = 1.0 / (1.0 + np.exp(-z))
    scaled = scaled.clip(0.0, 1.0)

    finite = pd.to_numeric(scaled, errors="coerce").dropna()
    if finite.empty:
        return pd.Series(np.nan, index=s.index)

    min_val = float(finite.min())
    max_val = float(finite.max())
    if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val <= min_val:
        return scaled

    return (scaled - min_val) / (max_val - min_val)


def compute_rsrp_quality(
    df: pd.DataFrame,
    kpi_columns: Sequence[str],
    disable_normalization: bool = False,
) -> pd.Series:
    """Compute the percentile-scaled RSRP quality score.
    If normalization is enabled (default), transform the first KPI column with a robust
    percentile-based logistic mapping into [0, 1], keeping tails smooth instead of stacking
    at the bounds. If disabled, return the raw mean of KPIs as a fallback.
    """
    if not kpi_columns:
        # Fallback: try RSRP column if present
        if "RSRP" in df.columns:
            kpi_columns = ["RSRP"]
        else:
            return pd.Series(np.nan, index=df.index)

    if disable_normalization:
        return compute_raw_mean(df, kpi_columns)

    # Normalize the first KPI only (MVP). Extend later for multi-KPI fusion if needed.
    first = kpi_columns[0]
    if first not in df.columns and first.upper() == "RSRP" and "RSRP" in df.columns:
        first = "RSRP"
    if first not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _normalize_percentile(df[first], 0.05, 0.95)


def aggregate_per_file(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each (rat, operator, env/speed_bucket, file), compute mean, median, std of the RSRP quality score.
    Returns (per_file_env, per_file_speed) DataFrames.
    """
    if df.empty or QUALITY_COLUMN not in df.columns:
        empty_env = pd.DataFrame(
            columns=[
                "rat",
                "operator",
                "env",
                "file",
                "mean_quality",
                "median_quality",
                "std_quality",
                "sample_count",
            ]
        )
        empty_speed = pd.DataFrame(
            columns=[
                "rat",
                "operator",
                "speed_bucket",
                "file",
                "mean_quality",
                "median_quality",
                "std_quality",
                "sample_count",
            ]
        )
        return empty_env, empty_speed

    agg_spec = ["mean", "median", "std", "count"]

    env_group_cols = ["rat", "operator", "env", "file"]
    speed_group_cols = ["rat", "operator", "speed_bucket", "file"]

    env_stats = (
        df.groupby(env_group_cols)[QUALITY_COLUMN]
        .agg(agg_spec)
        .reset_index()
        .rename(
            columns={
                "mean": "mean_quality",
                "median": "median_quality",
                "std": "std_quality",
                "count": "sample_count",
            }
        )
    )
    env_stats = env_stats[env_stats["sample_count"] > 0]

    speed_stats = (
        df.groupby(speed_group_cols)[QUALITY_COLUMN]
        .agg(agg_spec)
        .reset_index()
        .rename(
            columns={
                "mean": "mean_quality",
                "median": "median_quality",
                "std": "std_quality",
                "count": "sample_count",
            }
        )
    )
    speed_stats = speed_stats[speed_stats["sample_count"] > 0]

    return env_stats, speed_stats


def _summarise(
    df: pd.DataFrame, group_cols: Sequence[str]
) -> pd.DataFrame:
    """Helper to summarise per-file statistics across groups."""
    if df.empty:
        return pd.DataFrame(columns=[*group_cols, "mean_quality", "median_quality", "std_quality", "file_count"])

    summary = (
        df.groupby(list(group_cols))["mean_quality"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_quality",
                "median": "median_quality",
                "std": "std_quality",
                "count": "file_count",
            }
        )
    )
    return summary


def summarize_groups(
    per_file_env: pd.DataFrame, per_file_speed: pd.DataFrame
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Aggregate per-file stats into group-level summaries for overall, by RAT, by operator,
    and by speed buckets; return DataFrames covering all combinations.
    """
    overall = _summarise(per_file_env, ["env"])
    by_rat = _summarise(per_file_env, ["rat", "env"])
    by_operator = _summarise(per_file_env, ["operator", "env"])
    by_rat_operator = _summarise(per_file_env, ["rat", "operator", "env"])
    by_speed = _summarise(per_file_speed, ["speed_bucket"])
    by_speed_operator = _summarise(per_file_speed, ["operator", "speed_bucket"])

    if not overall.empty:
        overall["env"] = pd.Categorical(
            overall["env"],
            categories=["inside", "outdoor_driving", "outside_walking"],
            ordered=True,
        )
        overall = overall.sort_values("env").reset_index(drop=True)

    if not by_rat.empty:
        by_rat["env"] = pd.Categorical(
            by_rat["env"],
            categories=["inside", "outdoor_driving", "outside_walking"],
            ordered=True,
        )
        by_rat = by_rat.sort_values(["rat", "env"]).reset_index(drop=True)

    if not by_operator.empty:
        by_operator["env"] = pd.Categorical(
            by_operator["env"],
            categories=["inside", "outdoor_driving", "outside_walking"],
            ordered=True,
        )
        by_operator = by_operator.sort_values(["operator", "env"]).reset_index(drop=True)

    if not by_rat_operator.empty:
        by_rat_operator["env"] = pd.Categorical(
            by_rat_operator["env"],
            categories=["inside", "outdoor_driving", "outside_walking"],
            ordered=True,
        )
        by_rat_operator = by_rat_operator.sort_values(["rat", "operator", "env"]).reset_index(drop=True)

    desired_speed_order = ["inside", *OD_BUCKET_LABELS, "outside_walking"]
    if not by_speed.empty:
        by_speed["speed_bucket"] = pd.Categorical(
            by_speed["speed_bucket"], categories=desired_speed_order, ordered=True
        )
        by_speed = by_speed.sort_values("speed_bucket").reset_index(drop=True)

    if not by_speed_operator.empty:
        by_speed_operator["speed_bucket"] = pd.Categorical(
            by_speed_operator["speed_bucket"], categories=desired_speed_order, ordered=True
        )
        by_speed_operator = by_speed_operator.sort_values(["operator", "speed_bucket"]).reset_index(drop=True)

    return (
        overall,
        by_rat,
        by_speed,
        by_operator,
        by_rat_operator,
        by_speed_operator,
    )


def save_results(
    results_dir: Path,
    summary_overall: pd.DataFrame,
    summary_by_rat: pd.DataFrame,
    summary_by_speed: pd.DataFrame,
    summary_by_operator: pd.DataFrame,
    summary_by_rat_operator: pd.DataFrame,
    summary_by_speed_operator: pd.DataFrame,
    per_file_env: Optional[pd.DataFrame] = None,
    per_file_speed: Optional[pd.DataFrame] = None,
    kpi_presence: Optional[pd.DataFrame] = None,
) -> None:
    """Create results/ if missing; write summary CSVs."""
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_overall.to_csv(results_dir / "summary_overall.csv", index=False)
    summary_by_rat.to_csv(results_dir / "summary_by_rat.csv", index=False)
    summary_by_speed.to_csv(results_dir / "summary_by_speed.csv", index=False)
    summary_by_operator.to_csv(results_dir / "summary_by_operator.csv", index=False)
    summary_by_rat_operator.to_csv(results_dir / "summary_by_rat_operator.csv", index=False)
    summary_by_speed_operator.to_csv(results_dir / "summary_by_speed_operator.csv", index=False)

    if per_file_env is not None and not per_file_env.empty:
        export_env = per_file_env.rename(
            columns={
                "mean_quality": "mean",
                "median_quality": "median",
                "std_quality": "std",
                "sample_count": "count",
            }
        )
        export_env.to_csv(results_dir / "per_file_stats.csv", index=False)

    if per_file_speed is not None and not per_file_speed.empty:
        export_speed = per_file_speed.rename(
            columns={
                "mean_quality": "mean",
                "median_quality": "median",
                "std_quality": "std",
                "sample_count": "count",
            }
        )
        export_speed.to_csv(results_dir / "per_file_stats_by_speed.csv", index=False)

    if kpi_presence is not None and not kpi_presence.empty:
        kpi_presence.to_csv(results_dir / "kpi_presence.csv", index=False)


def plot_optionals(
    df: pd.DataFrame,
    summary_by_rat: pd.DataFrame,
    results_dir: Path,
    save_plots: bool,
    quality_column: str = QUALITY_COLUMN,
) -> None:
    """
    If `save_plots` is set, produce matplotlib plots (no seaborn): box/violin plots of
    the RSRP quality score by environment and operator, plus bar charts comparing RATs
    and operators. Regardless of plotting, compute configured non-parametric statistical
    tests and write a Markdown summary that maps results to the generated figures.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    stat_results: List[Dict[str, Any]] = []

    if quality_column not in df.columns:
        print(f"Skipping plot generation; missing {quality_column} column.")
        write_stat_results_md(results_dir, stat_results, stats_available=stats is not None)
        return

    metric_series = df[quality_column].dropna()
    if metric_series.empty:
        print(f"Skipping plot generation; {quality_column} column has no data.")
        write_stat_results_md(results_dir, stat_results, stats_available=stats is not None)
        return

    normalized_scale = _series_is_normalized(metric_series)
    scale_suffix = "(normalized 0-1)" if normalized_scale else "(raw)"
    overall_limits = _series_limits(metric_series)
    axis_label_default = _axis_label_from_series("RSRP quality score", normalized_scale, overall_limits)

    plot_figures = save_plots
    plt = None  # type: ignore[assignment]
    if save_plots:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            from matplotlib.lines import Line2D
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"Warning: matplotlib not available ({exc}); skipping plots.")
            plot_figures = False

    env_order = ["inside", "outdoor_driving", "outside_walking", "od_unknown"]
    env_series_all = [df.loc[df["env"] == env, quality_column].dropna() for env in env_order]
    env_labels = [label for label, series in zip(env_order, env_series_all) if not series.empty]
    env_data = [series for series in env_series_all if not series.empty]

    if env_data:
        combined_env = pd.concat(env_data, ignore_index=True)
        env_limits = _series_limits(combined_env)
        env_axis_label = _axis_label_from_series("RSRP quality score", normalized_scale, env_limits)
        env_context = "violin_quality_by_env.png / boxplot_quality_by_env.png"
        env_test = perform_nonparametric_tests(env_data, env_labels, env_context)
        if env_test:
            stat_results.append(env_test)

        if plot_figures:
            env_cmap = plt.get_cmap("Set2", len(env_data))
            env_colors = [env_cmap(i) for i in range(len(env_data))]
            env_positions = np.arange(1, len(env_labels) + 1)
            env_means = [series.mean() if not series.empty else np.nan for series in env_data]
            env_stds = [series.std(ddof=1) if series.size > 1 else np.nan for series in env_data]
            expanded_env = _expanded_limits_from_series(env_limits, normalized_scale)

            plt.figure(figsize=(8, 5))
            box = plt.boxplot(env_data, labels=env_labels, showmeans=True, patch_artist=True)
            for patch, color in zip(box["boxes"], env_colors):
                patch.set_facecolor(color)
                patch.set_edgecolor("#444444")
                patch.set_alpha(0.9)
            for element in ["whiskers", "caps"]:
                for artist in box[element]:
                    artist.set_color("#444444")
            for artist in box["medians"]:
                artist.set_color(MEDIAN_COLOR)
                artist.set_linewidth(2)
            for artist in box.get("means", []):
                artist.set_color(MEAN_COLOR)
                artist.set_markeredgecolor(MEAN_COLOR)
                artist.set_markerfacecolor("white")
                artist.set_markersize(6)
                artist.set_markeredgewidth(1.5)
            plt.ylabel(env_axis_label)
            if expanded_env:
                plt.ylim(expanded_env)
            plt.errorbar(
                env_positions,
                env_means,
                yerr=env_stds,
                fmt="D",
                color=MEAN_COLOR,
                ecolor=STD_COLOR,
                markerfacecolor="#ffffff",
                markeredgewidth=1.5,
                markeredgecolor=MEAN_COLOR,
                label="Mean",
            )
            handles = [
                Line2D([], [], marker="D", linestyle="None", color=MEAN_COLOR, markerfacecolor="white", markeredgewidth=1.5, label="Mean"),
                Line2D([], [], linestyle="-", color=STD_COLOR, linewidth=1.5, label="Standard deviation"),
                Line2D([], [], color=MEDIAN_COLOR, linewidth=2, label="Median"),
            ]
            plt.legend(handles=handles, loc="best")
            plt.title(f"RSRP Quality Distribution by Environment {scale_suffix}")
            plt.tight_layout()
            plt.savefig(results_dir / "boxplot_quality_by_env.png", dpi=200)
            plt.close()

            plt.figure(figsize=(8, 5))
            ax_env = plt.gca()
            violin_parts = ax_env.violinplot(env_data, positions=env_positions, showmeans=False, showmedians=True)
            for part, color in zip(violin_parts["bodies"], env_colors):
                part.set_facecolor(color)
                part.set_edgecolor("#444444")
                part.set_alpha(0.75)
            if "cmedians" in violin_parts:
                violin_parts["cmedians"].set_color(MEDIAN_COLOR)
                violin_parts["cmedians"].set_linewidth(2)
            ax_env.errorbar(
                env_positions,
                env_means,
                yerr=env_stds,
                fmt="D",
                color=MEAN_COLOR,
                ecolor=STD_COLOR,
                markerfacecolor="#ffffff",
                markeredgewidth=1.5,
                markeredgecolor=MEAN_COLOR,
            )
            ax_env.set_xticks(env_positions)
            ax_env.set_xticklabels(env_labels)
            ax_env.set_ylabel(env_axis_label)
            if expanded_env:
                ax_env.set_ylim(expanded_env)
            ax_env.set_title(f"RSRP Quality Violin Plot by Environment {scale_suffix}")
            ax_env.legend(handles=handles, loc="best")
            plt.tight_layout()
            plt.savefig(results_dir / "violin_quality_by_env.png", dpi=200)
            plt.close()

    operators = sorted(
        {
            op
            for op in df["operator"].dropna().unique()
            if df.loc[df["operator"] == op, quality_column].dropna().size > 0
        }
    )
    operator_data = [df.loc[df["operator"] == op, quality_column].dropna() for op in operators]

    if operator_data:
        combined_operator = pd.concat(operator_data, ignore_index=True)
        operator_limits = _series_limits(combined_operator)
        operator_axis_label = _axis_label_from_series("RSRP quality score", normalized_scale, operator_limits)
        operator_context = (
            "violin_quality_by_operator.png / boxplot_quality_by_operator.png / "
            "bar_quality_mean_std_by_operator.png"
        )
        operator_test = perform_nonparametric_tests(operator_data, operators, operator_context)
        if operator_test:
            stat_results.append(operator_test)

        expanded_operator = _expanded_limits_from_series(operator_limits, normalized_scale)

        if plot_figures:
            operator_cmap = plt.get_cmap("tab20", max(len(operators), 1))
            operator_colors = [operator_cmap(i % operator_cmap.N) for i in range(len(operators))]
            operator_positions = np.arange(1, len(operators) + 1)
            operator_means = [
                series.mean() if not series.empty else np.nan for series in operator_data
            ]
            operator_stds = [
                series.std(ddof=1) if series.size > 1 else np.nan for series in operator_data
            ]

            plt.figure(figsize=(max(8, 1.5 * len(operators)), 5))
            box = plt.boxplot(operator_data, labels=operators, showmeans=True, patch_artist=True)
            for patch, color in zip(box["boxes"], operator_colors):
                patch.set_facecolor(color)
                patch.set_edgecolor("#444444")
                patch.set_alpha(0.9)
            for element in ["whiskers", "caps"]:
                for artist in box[element]:
                    artist.set_color("#444444")
            for artist in box["medians"]:
                artist.set_color(MEDIAN_COLOR)
                artist.set_linewidth(2)
            for artist in box.get("means", []):
                artist.set_color(MEAN_COLOR)
                artist.set_markeredgecolor(MEAN_COLOR)
                artist.set_markerfacecolor("white")
                artist.set_markeredgewidth(1.5)
                artist.set_markersize(6)
            plt.ylabel(operator_axis_label)
            if expanded_operator:
                plt.ylim(expanded_operator)
            plt.errorbar(
                operator_positions,
                operator_means,
                yerr=operator_stds,
                fmt="D",
                color=MEAN_COLOR,
                ecolor=STD_COLOR,
                markerfacecolor="#ffffff",
                markeredgewidth=1.5,
                markeredgecolor=MEAN_COLOR,
            )
            handles = [
                Line2D([], [], marker="D", linestyle="None", color=MEAN_COLOR, markerfacecolor="white", markeredgewidth=1.5, label="Mean"),
                Line2D([], [], linestyle="-", color=STD_COLOR, linewidth=1.5, label="Standard deviation"),
                Line2D([], [], color=MEDIAN_COLOR, linewidth=2, label="Median"),
            ]
            plt.legend(handles=handles, loc="best")
            plt.title(f"RSRP Quality Distribution by Operator {scale_suffix}")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(results_dir / "boxplot_quality_by_operator.png", dpi=200)
            plt.close()

            plt.figure(figsize=(max(8, 1.5 * len(operators)), 5))
            ax_op = plt.gca()
            violin_parts = ax_op.violinplot(
                operator_data, positions=operator_positions, showmeans=False, showmedians=True
            )
            for part, color in zip(violin_parts["bodies"], operator_colors):
                part.set_facecolor(color)
                part.set_edgecolor("#444444")
                part.set_alpha(0.75)
            if "cmedians" in violin_parts:
                violin_parts["cmedians"].set_color(MEDIAN_COLOR)
                violin_parts["cmedians"].set_linewidth(2)
            ax_op.errorbar(
                operator_positions,
                operator_means,
                yerr=operator_stds,
                fmt="D",
                color=MEAN_COLOR,
                ecolor=STD_COLOR,
                markerfacecolor="#ffffff",
                markeredgewidth=1.5,
                markeredgecolor=MEAN_COLOR,
            )
            ax_op.set_xticks(operator_positions)
            ax_op.set_xticklabels(operators, rotation=20, ha="right")
            ax_op.set_ylabel(operator_axis_label)
            if expanded_operator:
                ax_op.set_ylim(expanded_operator)
            ax_op.set_title(f"RSRP Quality Violin Plot by Operator {scale_suffix}")
            ax_op.legend(handles=handles, loc="best")
            plt.tight_layout()
            plt.savefig(results_dir / "violin_quality_by_operator.png", dpi=200)
            plt.close()

            means = [
                series.mean() if not series.empty else np.nan for series in operator_data
            ]
            stds = [
                series.std(ddof=1) if series.size > 1 else np.nan for series in operator_data
            ]
            positions = np.arange(len(operators))

            plt.figure(figsize=(max(8, 1.5 * len(operators)), 4.5))
            plt.bar(
                positions,
                means,
                yerr=stds,
                color=operator_colors,
                edgecolor="#444444",
                error_kw={"ecolor": STD_COLOR, "capsize": 6, "elinewidth": 1.2},
            )
            plt.scatter(
                positions,
                means,
                marker="D",
                s=70,
                facecolors="white",
                edgecolors=MEAN_COLOR,
                linewidths=1.5,
                zorder=4,
            )
            handles = [
                Line2D([], [], marker="D", linestyle="None", color=MEAN_COLOR, markerfacecolor="white", markeredgewidth=1.5, label="Mean"),
                Line2D([], [], linestyle="-", color=STD_COLOR, linewidth=1.5, label="Standard deviation"),
            ]
            plt.legend(handles=handles, loc="best")
            plt.xticks(positions, operators, rotation=20, ha="right")
            plt.ylabel(operator_axis_label)
            if expanded_operator:
                plt.ylim(expanded_operator)
            plt.title(f"Mean RSRP Quality by Operator {scale_suffix}")
            plt.tight_layout()
            plt.savefig(results_dir / "bar_quality_mean_std_by_operator.png", dpi=200)
            plt.close()

    env_operator_data: List[Tuple[str, List[pd.Series], List[str]]] = []
    for env in env_order:
        env_subset = df.loc[df["env"] == env]
        if env_subset.empty:
            continue
        env_ops = [
            op
            for op in operators
            if env_subset.loc[env_subset["operator"] == op, quality_column].dropna().size > 0
        ]
        if not env_ops:
            continue
        env_series = [
            env_subset.loc[env_subset["operator"] == op, quality_column].dropna()
            for op in env_ops
        ]
        env_operator_data.append((env, env_series, env_ops))

    if env_operator_data:
        for env_label, series_list, labels in env_operator_data:
            env_op_context = f"boxplot_quality_by_operator_env.png [{env_label}]"
            env_op_test = perform_nonparametric_tests(series_list, labels, env_op_context)
            if env_op_test:
                stat_results.append(env_op_test)

        if plot_figures:
            fig, axes = plt.subplots(
                1,
                len(env_operator_data),
                sharey=True,
                figsize=(max(8, 5 * len(env_operator_data)), 5),
            )
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            legend_added = False
            for ax_panel, (env_label, series_list, labels) in zip(axes, env_operator_data):
                panel_cmap = plt.get_cmap("tab10", max(len(labels), 1))
                panel_colors = [panel_cmap(i % panel_cmap.N) for i in range(len(labels))]
                box = ax_panel.boxplot(
                    series_list, labels=labels, showmeans=True, patch_artist=True
                )
                for patch, color in zip(box["boxes"], panel_colors):
                    patch.set_facecolor(color)
                    patch.set_edgecolor("#444444")
                    patch.set_alpha(0.9)
                for element in ["whiskers", "caps"]:
                    for artist in box[element]:
                        artist.set_color("#444444")
                for artist in box["medians"]:
                    artist.set_color(MEDIAN_COLOR)
                    artist.set_linewidth(2)
                for artist in box.get("means", []):
                    artist.set_color(MEAN_COLOR)
                    artist.set_markeredgecolor(MEAN_COLOR)
                    artist.set_markerfacecolor("white")
                    artist.set_markeredgewidth(1.5)
                    artist.set_markersize(6)
                ax_panel.set_title(env_label)
                ax_panel.tick_params(axis="x", rotation=20)
                combined_panel = pd.concat(series_list, ignore_index=True)
                panel_limits = _series_limits(combined_panel) or overall_limits
                panel_axis_label = _axis_label_from_series(
                    "RSRP quality score", normalized_scale, panel_limits
                )
                ax_panel.set_ylabel(panel_axis_label)
                positions = np.arange(1, len(labels) + 1)
                means = [
                    series.mean() if not series.empty else np.nan for series in series_list
                ]
                stds = [
                    series.std(ddof=1) if series.size > 1 else np.nan for series in series_list
                ]
                ax_panel.errorbar(
                    positions,
                    means,
                    yerr=stds,
                    fmt="D",
                    color=MEAN_COLOR,
                    ecolor=STD_COLOR,
                    markerfacecolor="#ffffff",
                    markeredgewidth=1.5,
                    markeredgecolor=MEAN_COLOR,
                )
                if not legend_added:
                    handles = [
                        Line2D([], [], marker="D", linestyle="None", color=MEAN_COLOR, markerfacecolor="white", markeredgewidth=1.5, label="Mean"),
                        Line2D([], [], linestyle="-", color=STD_COLOR, linewidth=1.5, label="Standard deviation"),
                        Line2D([], [], color=MEDIAN_COLOR, linewidth=2, label="Median"),
                    ]
                    ax_panel.legend(handles=handles, loc="best")
                    legend_added = True
                expanded_panel = _expanded_limits_from_series(panel_limits, normalized_scale)
                if expanded_panel:
                    ax_panel.set_ylim(expanded_panel)
            fig.suptitle(f"RSRP Quality by Operator within Environment {scale_suffix}")
            fig.tight_layout()
            fig.savefig(results_dir / "boxplot_quality_by_operator_env.png", dpi=200)
            plt.close(fig)

    if not summary_by_rat.empty:
        rats = list(summary_by_rat["rat"].unique())
        rat_pairs = [
            (rat, df[df["rat"] == rat][quality_column].dropna()) for rat in rats
        ]
        rat_filtered = [(label, series) for label, series in rat_pairs if not series.empty]
        if rat_filtered:
            rat_labels = [label for label, _ in rat_filtered]
            rat_series = [series for _, series in rat_filtered]
            rat_context = "bar_quality_mean_std_by_rat.png"
            rat_test = perform_nonparametric_tests(rat_series, rat_labels, rat_context)
            if rat_test:
                stat_results.append(rat_test)

        if plot_figures:
            means = []
            stds = []
            for rat in rats:
                rat_rows = df[df["rat"] == rat][quality_column].dropna()
                if rat_rows.empty:
                    means.append(np.nan)
                    stds.append(np.nan)
                else:
                    means.append(rat_rows.mean())
                    stds.append(rat_rows.std(ddof=1))

            positions = np.arange(len(rats))
            plt.figure(figsize=(6, 4))
            rat_cmap = plt.get_cmap("Set1", max(len(rats), 1))
            rat_colors = [rat_cmap(i % rat_cmap.N) for i in range(len(rats))]
            plt.bar(
                positions,
                means,
                yerr=stds,
                color=rat_colors,
                edgecolor="#444444",
                error_kw={"ecolor": STD_COLOR, "capsize": 6, "elinewidth": 1.2},
            )
            plt.scatter(
                positions,
                means,
                marker="D",
                s=70,
                facecolors="white",
                edgecolors=MEAN_COLOR,
                linewidths=1.5,
                zorder=4,
            )
            handles = [
                Line2D([], [], marker="D", linestyle="None", color=MEAN_COLOR, markerfacecolor="white", markeredgewidth=1.5, label="Mean"),
                Line2D([], [], linestyle="-", color=STD_COLOR, linewidth=1.5, label="Standard deviation"),
            ]
            plt.legend(handles=handles, loc="best")
            plt.xticks(positions, rats)
            plt.ylabel(axis_label_default)
            expanded = _expanded_limits_from_series(overall_limits, normalized_scale)
            if expanded:
                plt.ylim(expanded)
            plt.title(f"Mean RSRP Quality by RAT {scale_suffix}")
            plt.tight_layout()
            plt.savefig(results_dir / "bar_quality_mean_std_by_rat.png", dpi=200)
            plt.close()

def print_table(title: str, df: pd.DataFrame) -> None:
    """Pretty-print a DataFrame with a heading."""
    print(f"\n{title}")
    if df.empty:
        print("  No data available.")
        return

    display_df = df.copy()
    for col in display_df.select_dtypes(include=["float", "float64"]).columns:
        display_df[col] = display_df[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    print(display_df.to_string(index=False))


def maybe_print_ttests(per_file_env: pd.DataFrame) -> None:
    """Optionally print Welch t-tests comparing IS vs OD per RAT if scipy is available."""
    if stats is None or per_file_env.empty:
        return

    print("\nWelch t-test (per RAT, IS vs OD):")

    rats = sorted(per_file_env["rat"].unique())
    for rat in rats:
        inside = per_file_env[
            (per_file_env["rat"] == rat) & (per_file_env["env"] == "inside")
        ]["mean_quality"]
        outdoor = per_file_env[
            (per_file_env["rat"] == rat) & (per_file_env["env"] == "outdoor_driving")
        ]["mean_quality"]

        if len(inside) < 2 or len(outdoor) < 2:
            print(f"  {rat}: insufficient data for t-test.")
            continue

        t_stat, p_value = stats.ttest_ind(inside, outdoor, equal_var=False, nan_policy="omit")
        print(f"  {rat}: t={t_stat:.3f}, p={p_value:.4f}")


def parse_percentile_clip(pclip_arg: str) -> Tuple[float, float]:
    """Parse percentile clip argument."""
    parts = [p.strip() for p in pclip_arg.split(",")]
    if len(parts) != 2:
        raise ValueError("Percentile clip must be provided as 'low,high'.")
    low, high = float(parts[0]), float(parts[1])
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("Percentile clip values must satisfy 0 <= low < high <= 1.")
    return low, high


def parse_speed_thresholds(arg: str) -> Tuple[float, float]:
    """Parse OD speed thresholds argument."""
    parts = [p.strip() for p in arg.split(",")]
    if len(parts) != 2:
        raise ValueError("Speed thresholds must be provided as 'low,high'.")
    low, high = float(parts[0]), float(parts[1])
    if not 0.0 <= low <= high:
        raise ValueError("Speed thresholds must satisfy 0 <= low <= high.")
    return low, high


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze passive 4G/5G indoor vs outdoor RSRP quality."
    )
    parser.add_argument("--five_g_dir", default="./5G", help="Path to 5G passive CSV folder.")
    parser.add_argument(
        "--exclude_ow",
        action="store_true",
        help="Exclude outside walking (OW) files from the analysis.",
    )
    parser.add_argument(
        "--speed_thresholds",
        default="0.5,5",
        help="OD speed thresholds in m/s as 'low,high' (inclusive bounds at low, high).",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        default=True,
        help="Save optional matplotlib figures to results/.",
    )
    parser.add_argument(
        "--disable_normalization",
        action="store_true",
        help="Skip KPI normalization and compute the RSRP quality score from raw KPI values.",
    )

    args = parser.parse_args()

    five_g_dir = Path(args.five_g_dir)

    print("Starting passive quality analysis...")
    include_ow = not args.exclude_ow
    print(f"  5G directory: {five_g_dir}")
    print(f"  Include OW: {'yes' if include_ow else 'no'}")

    speed_thresholds = parse_speed_thresholds(args.speed_thresholds)
    print(f"  OD speed thresholds (m/s): {speed_thresholds[0]:.2f}, {speed_thresholds[1]:.2f}")

    print("Loading passive measurement files...")
    load_result = load_passive_files(five_g_dir, include_ow=include_ow)
    df = load_result.dataframe

    if df.empty:
        print("No passive data found under the provided directories.")
        return

    total_rows = len(df)
    unique_files = df["file"].nunique()
    print(f"  Loaded {unique_files} files with {total_rows} usable rows.")
    if load_result.kpi_columns:
        print(f"  KPI columns selected: {', '.join(load_result.kpi_columns)}")

    df = assign_speed_buckets(df, speed_thresholds)
    print("  Assigned speed buckets.")

    # Compute RSRP quality score (normalized 0–1 unless disabled)
    metric_col = load_result.kpi_columns[0] if load_result.kpi_columns else ("RSRP" if "RSRP" in df.columns else None)
    if metric_col is None:
        print("No KPI column available to compute the RSRP quality score.")
        return

    # Ensure RSRP numeric
    if metric_col in df.columns:
        df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    if metric_col != "RSRP" and "RSRP" in df.columns:
        # prefer RSRP name downstream if present
        df["RSRP"] = pd.to_numeric(df["RSRP"], errors="coerce")

    df[QUALITY_COLUMN] = compute_rsrp_quality(
        df, [metric_col], disable_normalization=args.disable_normalization
    )
    results_root = Path("results_rsrp")

    print("Aggregating per-file statistics...")
    per_file_env, per_file_speed = aggregate_per_file(df)
    (
        summary_overall,
        summary_by_rat,
        summary_by_speed,
        summary_by_operator,
        summary_by_rat_operator,
        summary_by_speed_operator,
    ) = summarize_groups(per_file_env, per_file_speed)

    print_table("Overall IS vs OD (all RATs combined)", summary_overall)
    print_table("By RAT (4G / 5G): IS vs OD", summary_by_rat)
    print_table("By speed bucket (4G+5G)", summary_by_speed)
    print_table("By carrier: IS vs OD", summary_by_operator)
    print_table("By RAT & operator: IS vs OD", summary_by_rat_operator)
    print_table("By speed bucket & operator", summary_by_speed_operator)

    if not load_result.kpi_presence.empty:
        print_table("KPI availability across files", load_result.kpi_presence)

    maybe_print_ttests(per_file_env)

    results_dir = results_root
    print(f"Writing summaries to {results_dir}/ ...")
    save_results(
        results_dir,
        summary_overall,
        summary_by_rat,
        summary_by_speed,
        summary_by_operator,
        summary_by_rat_operator,
        summary_by_speed_operator,
        per_file_env=per_file_env,
        per_file_speed=per_file_speed,
        kpi_presence=load_result.kpi_presence,
    )
    plot_optionals(df, summary_by_rat, results_dir, save_plots=args.save_plots)

    print("Analysis complete.")


if __name__ == "__main__":
    main()
