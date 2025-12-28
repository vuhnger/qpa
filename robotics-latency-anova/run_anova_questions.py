from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

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
            return int(label.lower().replace("ms", ""))
        except ValueError:
            return label

    return sorted(latencies, key=parse_value)


def compute_rm_anova(table):
    # table is a participant x latency matrix with no missing cells
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
    # Center columns
    centered = table - table.mean(axis=0)
    S = np.cov(centered, rowvar=False, bias=False)
    trace_S = np.trace(S)
    det_S = np.linalg.det(S)
    if det_S <= 0 or trace_S <= 0:
        return {"W": np.nan, "chi2": np.nan, "df": (p - 1) * (p + 2) / 2, "p": np.nan, "applicable": False}
    W = det_S / ((trace_S / p) ** p)
    df = (p - 1) * (p + 2) / 2
    # Correction factor for small samples
    c = (2 * p * p + p + 2) / (6 * (p - 1) * (n - 1))
    chi2 = -(n - 1) * (1 - c) * np.log(W)
    p_val = stats.chi2.sf(chi2, df)
    return {"W": W, "chi2": chi2, "df": df, "p": p_val, "applicable": True}


def shapiro_residual_normality(table):
    """
    Shapiro-Wilk on residuals from participant and latency effects removed.
    """
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
    df = pd.read_csv("manually_pruned_data.csv", sep=";")
    question_cols = [
        col
        for col in df.columns
        if col not in ["Participant_ID", "Latency", "Run_Number"]
    ]

    latency_order = parse_latency_order(df["Latency"].unique())
    print(f"Latencies (ordered): {latency_order}\n")

    visuals_dir = Path("visuals")
    visuals_dir.mkdir(exist_ok=True)

    def savefig(path, *args, **kwargs):
        path = visuals_dir / path
        if path.exists():
            path.unlink()
        plt.savefig(path, facecolor="black", **kwargs)
        print(f"Saved visual to {path}")

    results_rows = []
    plot_data = []
    effect_sizes = []
    assumption_data = []

    for question in question_cols:
        avg = (
            df.groupby(["Participant_ID", "Latency"], as_index=False)[question]
            .mean()
            .pivot(index="Participant_ID", columns="Latency", values=question)
        )
        avg = avg[latency_order]

        stats_dict = compute_rm_anova(avg)
        crit = critical_f_values(stats_dict["df_latency"], stats_dict["df_error"])
        alphas = sorted(crit.keys())
        sig_levels = [a for a in alphas if stats_dict["p_latency"] < a]
        sig_text = (
            f"significant (p < {sig_levels[0]})" if sig_levels else "not significant"
        )

        sphericity = mauchly_sphericity(avg)
        residual_norm = shapiro_residual_normality(avg)

        means = avg.mean(axis=0)
        sems = avg.sem(axis=0)

        row = {
            "question": question,
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
        # Include per-latency means for reference/post hoc setup.
        for lat, mean_val in avg.mean(axis=0).items():
            row[f"mean_{lat}"] = mean_val
        results_rows.append(row)
        plot_data.append({"question": question, "means": means, "sems": sems})
        effect_sizes.append(
            {"question": question, "partial_eta_sq": stats_dict["effect_partial_eta_sq"]}
        )
        assumption_data.append(
            {
                "question": question,
                "mauchly_p": sphericity["p"],
                "mauchly_applicable": sphericity["applicable"],
                "shapiro_p": residual_norm["p"],
            }
        )

        print(f"Question: {question}")
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
            + ", ".join(
                f"{alpha} -> {value:.4f}" for alpha, value in crit.items()
            )
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

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv("anova_results.csv", index=False)
    print("Saved table to anova_results.csv")

    # Visuals: per-question latency means with SEM, and partial eta squared bars.
    if plot_data:
        fig, axes = plt.subplots(len(plot_data), 1, figsize=(8, 4 * len(plot_data)), sharex=True)
        if len(plot_data) == 1:
            axes = [axes]
        x = np.arange(len(latency_order))
        for ax, pd_entry in zip(axes, plot_data):
            ax.errorbar(
                x,
                pd_entry["means"].values,
                yerr=pd_entry["sems"].values,
                fmt="o",
                linestyle="none",
                capsize=4,
                color="white",
                ecolor="deepskyblue",
                linewidth=2,
                label="Mean ± SEM",
            )
            ax.set_title(f"Per-latency means (SEM) — {pd_entry['question']}")
            ax.set_ylabel("Mean rating")
            ax.set_xticks(x)
            ax.set_xticklabels(latency_order)
            ax.grid(alpha=0.3)
            ax.legend()
        axes[-1].set_xlabel("Latency condition")
        plt.tight_layout()
        # Keep legacy filename and add a clearer name for the questionnaire means plot
        savefig("anova_latency_means.png", dpi=300)
        savefig("anova_question_means_sem.png", dpi=300)
        plt.close()

    if effect_sizes:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [e["question"] for e in effect_sizes]
        vals = [e["partial_eta_sq"] for e in effect_sizes]
        bars = ax.bar(labels, vals, color="mediumseagreen", edgecolor="white")
        ax.set_ylabel("Partial eta squared")
        ax.set_title("Effect size by question")
        ax.set_ylim(0, max(vals + [0.1]) * 1.2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        savefig("anova_effect_sizes.png", dpi=300)
        plt.close()

    if assumption_data:
        labels = [a["question"] for a in assumption_data]
        mauchly_ps = [a["mauchly_p"] if a["mauchly_applicable"] else np.nan for a in assumption_data]
        shapiro_ps = [a["shapiro_p"] for a in assumption_data]
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - width / 2, shapiro_ps, width, label="Residual normality p", color="white", edgecolor="white")
        ax.bar(x + width / 2, mauchly_ps, width, label="Sphericity p", color="deepskyblue", edgecolor="white")
        ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="α = 0.05")
        # Annotate values for clarity (especially tiny p-values)
        for xi, pval in zip(x - width / 2, shapiro_ps):
            if not np.isnan(pval):
                ax.text(xi, pval + 0.02, f"{pval:.3f}", ha="center", va="bottom", fontsize=8)
        for xi, pval in zip(x + width / 2, mauchly_ps):
            if not np.isnan(pval):
                ax.text(xi, pval + 0.02, f"{pval:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("p-values")
        ax.set_title("Assumption checks")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        savefig("anova_assumption_checks.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
