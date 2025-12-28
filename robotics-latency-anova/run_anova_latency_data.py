from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Visual style matches earlier analysis scripts
plt.style.use("dark_background")
plt.rcParams.update(
    {
        "figure.facecolor": "black",
        "axes.facecolor": "black",
        "axes.edgecolor": "white",
        "axes.labelcolor": "yellow",
        "text.color": "yellow",
        "xtick.color": "yellow",
        "ytick.color": "yellow",
        "grid.color": "#666666",
    }
)


def parse_latency_order(latencies):
    def parse_value(label):
        try:
            return int(str(label).lower().replace("ms", ""))
        except ValueError:
            return label

    return sorted(latencies, key=parse_value)


def compute_rm_anova(table):
    """One-way repeated-measures ANOVA for complete participant x latency matrix."""
    n_participants, n_latency = table.shape
    grand_mean = table.values.mean()
    mean_latency = table.mean(axis=0).values
    mean_participant = table.mean(axis=1).values

    ss_latency = n_participants * np.sum((mean_latency - grand_mean) ** 2)
    ss_participant = n_latency * np.sum((mean_participant - grand_mean) ** 2)
    ss_total = np.sum((table.values - grand_mean) ** 2)
    ss_error = ss_total - ss_latency - ss_participant

    df_latency = n_latency - 1
    df_participant = n_participants - 1
    df_error = df_latency * df_participant

    ms_latency = ss_latency / df_latency
    ms_error = ss_error / df_error
    f_latency = ms_latency / ms_error
    p_latency = stats.f.sf(f_latency, df_latency, df_error)

    return {
        "column_mean": grand_mean,
        "effect_partial_eta_sq": ss_latency / (ss_latency + ss_error),
        "ss_total": ss_total,
        "ss_latency": ss_latency,
        "ss_participant": ss_participant,
        "ss_error": ss_error,
        "ssa_div_sst": ss_latency / ss_total,
        "sse_div_sst": ss_error / ss_total,
        "df_latency": df_latency,
        "df_participant": df_participant,
        "df_error": df_error,
        "ms_latency": ms_latency,
        "ms_error": ms_error,
        "f_latency": f_latency,
        "p_latency": p_latency,
    }


def mauchly_sphericity(table):
    """
    Mauchly's test for sphericity on wide data (participants x levels).
    Returns W, chi2, df, p. If levels < 3, sphericity not applicable.
    """
    n, p = table.shape
    if p < 3:
        return {"W": np.nan, "chi2": np.nan, "df": 0, "p": np.nan, "applicable": False}
    centered = table - table.mean(axis=0)
    S = np.cov(centered, rowvar=False, bias=False)
    trace_S = np.trace(S)
    det_S = np.linalg.det(S)
    if det_S <= 0 or trace_S <= 0:
        return {
            "W": np.nan,
            "chi2": np.nan,
            "df": (p - 1) * (p + 2) / 2,
            "p": np.nan,
            "applicable": False,
        }
    W = det_S / ((trace_S / p) ** p)
    df = (p - 1) * (p + 2) / 2
    c = (2 * p * p + p + 2) / (6 * (p - 1) * (n - 1))
    chi2 = -(n - 1) * (1 - c) * np.log(W)
    p_val = stats.chi2.sf(chi2, df)
    return {"W": W, "chi2": chi2, "df": df, "p": p_val, "applicable": True}


def shapiro_residual_normality(table):
    """Shapiro-Wilk on residuals from participant and latency effects removed."""
    values = table.values
    grand_mean = values.mean()
    row_means = values.mean(axis=1, keepdims=True)
    col_means = values.mean(axis=0, keepdims=True)
    residuals = values - row_means - col_means + grand_mean
    residuals = residuals.flatten()
    stat, p = stats.shapiro(residuals)
    return {"W": stat, "p": p}


def critical_f_values(df1, df2):
    alphas = [0.05, 0.01, 0.001]
    return {alpha: stats.f.ppf(1 - alpha, df1, df2) for alpha in alphas}


def main():
    input_path = Path("participant_latency_data.csv")
    df_raw = pd.read_csv(input_path, na_values=["None", "nan", "NaN", ""])

    # Normalize column names
    if "Participant" in df_raw.columns and "Participant_ID" not in df_raw.columns:
        df_raw = df_raw.rename(columns={"Participant": "Participant_ID"})

    latency_cols = [c for c in df_raw.columns if c != "Participant_ID"]
    latency_order = parse_latency_order(latency_cols)
    print(f"Latencies (ordered): {latency_order}")

    # Ensure numeric values and drop participants with missing latency data
    df_numeric = df_raw.copy()
    df_numeric[latency_cols] = df_numeric[latency_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    participants_before = len(df_numeric)
    df_complete = df_numeric.dropna(subset=latency_cols)
    dropped = participants_before - len(df_complete)
    if dropped > 0:
        print(f"Dropped {dropped} participant(s) with missing latency cells.")

    table = df_complete.set_index("Participant_ID")[latency_order]

    stats_dict = compute_rm_anova(table)
    crit = critical_f_values(stats_dict["df_latency"], stats_dict["df_error"])
    alphas = sorted(crit.keys())
    sig_levels = [a for a in alphas if stats_dict["p_latency"] < a]
    sig_text = (
        f"significant (p < {sig_levels[0]})" if sig_levels else "not significant"
    )

    sphericity = mauchly_sphericity(table)
    residual_norm = shapiro_residual_normality(table)

    means = table.mean(axis=0)
    sems = table.sem(axis=0)

    # Assemble results table (same structure as earlier scripts)
    row = {
        "measure": "latency_completion_time",
        "participants_included": len(table),
        "participants_dropped": dropped,
        "column_mean": stats_dict["column_mean"],
        "effect_partial_eta_sq": stats_dict["effect_partial_eta_sq"],
        "SST": stats_dict["ss_total"],
        "SSA (ss_latency)": stats_dict["ss_latency"],
        "SS_Participant": stats_dict["ss_participant"],
        "SSE": stats_dict["ss_error"],
        "SSA/SST": stats_dict["ssa_div_sst"],
        "SSE/SST": stats_dict["sse_div_sst"],
        "df_latency": stats_dict["df_latency"],
        "df_participant": stats_dict["df_participant"],
        "df_error": stats_dict["df_error"],
        "MSA": stats_dict["ms_latency"],
        "MSE": stats_dict["ms_error"],
        "F_latency": stats_dict["f_latency"],
        "P (RT)": stats_dict["p_latency"],
        "Fcrit_0.05": crit[0.05],
        "Fcrit_0.01": crit[0.01],
        "Fcrit_0.001": crit[0.001],
        "significant": bool(sig_levels),
        "significant_at_alpha": sig_levels[0] if sig_levels else None,
        "mauchly_W": sphericity["W"],
        "mauchly_chi2": sphericity["chi2"],
        "mauchly_df": sphericity["df"],
        "mauchly_p": sphericity["p"],
        "residual_shapiro_W": residual_norm["W"],
        "residual_shapiro_p": residual_norm["p"],
    }
    for lat, mean_val in means.items():
        row[f"mean_{lat}"] = mean_val
    for lat, sem_val in sems.items():
        row[f"sem_{lat}"] = sem_val

    results_df = pd.DataFrame([row])
    results_df.to_csv("anova_latency_results.csv", index=False)
    print("Saved table to anova_latency_results.csv")

    # Compact calculation table mirroring the manual F-ratio layout
    calc_df = pd.DataFrame(
        [
            {
                "SS_effect (SSA)": stats_dict["ss_latency"],
                "df_effect": stats_dict["df_latency"],
                "SS_error": stats_dict["ss_error"],
                "df_error": stats_dict["df_error"],
                "MS_effect": stats_dict["ms_latency"],
                "MS_error": stats_dict["ms_error"],
                "F_latency": stats_dict["f_latency"],
                "SST (total)": stats_dict["ss_total"],
            }
        ]
    )
    calc_df.to_csv("anova_latency_calc_table.csv", index=False)
    print("Saved calc table to anova_latency_calc_table.csv")

    sig_df = pd.DataFrame(
        [
            {
                "F_latency": stats_dict["f_latency"],
                "p_value": stats_dict["p_latency"],
                "significant_0.05": "Yes" if stats_dict["p_latency"] < 0.05 else "No",
            }
        ]
    )
    sig_df.to_csv("anova_latency_significance_table.csv", index=False)
    print("Saved significance table to anova_latency_significance_table.csv")

    # Console summary
    print("-" * 60)
    print(f"Participants included: {len(table)}")
    print(f"Per-latency means:\n{means}\n")
    print(
        f"Column mean: {stats_dict['column_mean']:.4f}, "
        f"Effect (partial eta^2): {stats_dict['effect_partial_eta_sq']:.4f}"
    )
    print(
        f"SST: {stats_dict['ss_total']:.4f}, "
        f"SSA (ss_latency): {stats_dict['ss_latency']:.4f}, "
        f"SS_Participant: {stats_dict['ss_participant']:.4f}, "
        f"SSE: {stats_dict['ss_error']:.4f}"
    )
    print(
        f"SSA/SST: {stats_dict['ssa_div_sst']:.4f}, "
        f"SSE/SST: {stats_dict['sse_div_sst']:.4f}"
    )
    print(
        f"df_latency: {stats_dict['df_latency']}, "
        f"df_participant: {stats_dict['df_participant']}, "
        f"df_error: {stats_dict['df_error']}"
    )
    print(
        f"MSA: {stats_dict['ms_latency']:.4f}, "
        f"MSE: {stats_dict['ms_error']:.4f}"
    )
    print(
        f"F_latency: {stats_dict['f_latency']:.4f}, "
        f"P (RT): {stats_dict['p_latency']:.6f}"
    )
    print(
        "Critical F-values (alpha): "
        + ", ".join(f"{alpha} -> {value:.4f}" for alpha, value in crit.items())
    )
    print(f"Result is {sig_text}")
    if sphericity["applicable"]:
        print(
            f"Sphericity (Mauchly): W={sphericity['W']:.4f}, "
            f"chi2={sphericity['chi2']:.3f}, df={int(sphericity['df'])}, "
            f"p={sphericity['p']:.4f}"
        )
    else:
        print("Sphericity (Mauchly): not applicable (levels < 3 or invalid matrix)")
    print(
        f"Residual normality (Shapiro-Wilk): W={residual_norm['W']:.4f}, "
        f"p={residual_norm['p']:.4f}"
    )
    print("-" * 60)

    # Visuals
    visuals_dir = Path("visuals")
    visuals_dir.mkdir(exist_ok=True)

    def savefig(path, *args, **kwargs):
        path = visuals_dir / path
        if path.exists():
            path.unlink()
        plt.savefig(path, facecolor="black", **kwargs)
        print(f"Saved visual to {path}")

    x = np.arange(len(latency_order))
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(latency_order)))

    # Mean ± SEM with participant trajectories
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, lat in enumerate(latency_order):
        jitter = np.random.normal(loc=x[idx], scale=0.05, size=len(table))
        ax.scatter(
            jitter,
            table[lat],
            color=colors[idx],
            edgecolors="white",
            linewidth=0.6,
            alpha=0.7,
            s=35,
            label=f"{lat} (n={len(table)})" if idx == 0 else None,
        )
    ax.errorbar(
        x,
        means.values,
        yerr=sems.values,
        fmt="o",
        capsize=5,
        color="white",
        ecolor="deepskyblue",
        linewidth=2,
        label="Mean ± SEM",
        zorder=5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(latency_order)
    ax.set_xlabel("Latency condition")
    ax.set_ylabel("Completion time")
    ax.set_title("Latency effect on completion time (RM-ANOVA)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    savefig("anova_latency_means.png", dpi=300)
    plt.close(fig)

    # Boxplots to show distribution per condition
    fig, ax = plt.subplots(figsize=(8, 5))
    box = ax.boxplot(
        [table[lat].values for lat in latency_order],
        patch_artist=True,
        tick_labels=latency_order,
        widths=0.6,
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("white")
    for whisker in box["whiskers"]:
        whisker.set_color("white")
    for median in box["medians"]:
        median.set_color("white")
        median.set_linewidth(2)
    ax.axhline(stats_dict["column_mean"], color="yellow", linestyle="--", linewidth=1, label="Grand mean")
    ax.set_xlabel("Latency condition")
    ax.set_ylabel("Completion time")
    ax.set_title("Completion time distribution by latency")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    savefig("anova_latency_boxplot.png", dpi=300)
    plt.close(fig)

    # Assumption check visual (p-values)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Residual normality", "Sphericity"],
        [residual_norm["p"], sphericity["p"]],
        color=["white", "deepskyblue"],
        edgecolor="white",
    )
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="α = 0.05")
    for bar, val in zip(bars, [residual_norm["p"], sphericity["p"]]):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_ylim(0, 1)
    ax.set_ylabel("p-value")
    ax.set_title("Assumption checks")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    savefig("anova_latency_assumptions.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
