import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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


def holm_correction(p_values):
    """Return Holm-Bonferroni adjusted p-values in original order."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    prev_adj = 0
    for idx, rank in enumerate(order):
        raw_p = p_values[rank]
        adj_p = min(1.0, (m - idx) * raw_p)
        adj_p = max(adj_p, prev_adj)  # ensure monotonicity
        adjusted[rank] = adj_p
        prev_adj = adj_p
    return adjusted


def cohen_dz(x, y):
    diff = x - y
    sd_diff = diff.std(ddof=1)
    if sd_diff == 0:
        return np.nan
    return diff.mean() / sd_diff


def main():
    df = pd.read_csv("manually_pruned_data.csv", sep=";")
    question_cols = [
        col
        for col in df.columns
        if col not in ["Participant_ID", "Latency", "Run_Number"]
    ]
    latency_order = parse_latency_order(df["Latency"].unique())

    results_rows = []
    visual_data = []
    visual_question = question_cols[0] if question_cols else None
    heatmap_matrix = None
    print(f"Latencies (ordered): {latency_order}\n")

    visuals_dir = Path("visuals")
    visuals_dir.mkdir(exist_ok=True)

    def savefig(path, *args, **kwargs):
        path = visuals_dir / path
        if path.exists():
            path.unlink()
        plt.savefig(path, facecolor="black", **kwargs)
        print(f"Saved visual to {path}")

    for question in question_cols:
        avg = (
            df.groupby(["Participant_ID", "Latency"], as_index=False)[question]
            .mean()
            .pivot(index="Participant_ID", columns="Latency", values=question)
        )
        avg = avg[latency_order]
        means = avg.mean(axis=0)

        print(f"Question: {question}")
        print(f"Per-latency means:\n{means}\n")

        pairs = list(itertools.combinations(latency_order, 2))
        raw_ps = []
        stats_cache = []

        for lat_a, lat_b in pairs:
            a = avg[lat_a]
            b = avg[lat_b]
            t_stat, p_raw = stats.ttest_rel(a, b)
            dz = cohen_dz(a, b)
            mean_diff = a.mean() - b.mean()
            stats_cache.append(
                {
                    "lat_a": lat_a,
                    "lat_b": lat_b,
                    "mean_a": a.mean(),
                    "mean_b": b.mean(),
                    "mean_diff": mean_diff,
                    "t_stat": t_stat,
                    "df": len(a) - 1,
                    "p_raw": p_raw,
                    "effect_size_dz": dz,
                }
            )
            raw_ps.append(p_raw)

        p_holm = holm_correction(np.array(raw_ps))
        alphas = [0.05, 0.01, 0.001]

        for (stats_row, adj_p) in zip(stats_cache, p_holm):
            sig_levels = [a for a in alphas if adj_p < a]
            sig_text = (
                f"significant after Holm (p_adj < {sig_levels[0]})"
                if sig_levels
                else "not significant after Holm"
            )

            results_rows.append(
                {
                    "question": question,
                    "latency_a": stats_row["lat_a"],
                    "latency_b": stats_row["lat_b"],
                    "mean_a": stats_row["mean_a"],
                    "mean_b": stats_row["mean_b"],
                    "mean_diff_a_minus_b": stats_row["mean_diff"],
                    "t_stat": stats_row["t_stat"],
                    "df": stats_row["df"],
                    "p_raw": stats_row["p_raw"],
                    "p_holm": adj_p,
                    "significant_0.05": adj_p < 0.05,
                    "significant_0.01": adj_p < 0.01,
                    "significant_0.001": adj_p < 0.001,
                    "significant_at_alpha": sig_levels[0] if sig_levels else None,
                    "effect_size_dz": stats_row["effect_size_dz"],
                }
            )

            print(
                f"{stats_row['lat_a']} vs {stats_row['lat_b']}: "
                f"mean {stats_row['mean_a']:.3f} vs {stats_row['mean_b']:.3f}, "
                f"diff {stats_row['mean_diff']:.3f}; "
                f"t({stats_row['df']}) = {stats_row['t_stat']:.3f}, "
                f"p = {stats_row['p_raw']:.6f}, "
                f"Holm p = {adj_p:.6f}, "
                f"{sig_text}"
            )
            if question == visual_question:
                label = f"{stats_row['lat_a']}-{stats_row['lat_b']}"
                visual_data.append((label, adj_p))
        if question == visual_question:
            heatmap_matrix = pd.DataFrame(
                np.nan, index=latency_order, columns=latency_order, dtype=float
            )
            for (stats_row, adj_p) in zip(stats_cache, p_holm):
                a, b = stats_row["lat_a"], stats_row["lat_b"]
                heatmap_matrix.loc[a, b] = adj_p
                heatmap_matrix.loc[b, a] = adj_p
        print("-" * 60)

    out_df = pd.DataFrame(results_rows)
    out_df.to_csv("posthoc_latency_pairs.csv", index=False)
    print("Saved pairwise post hoc results to posthoc_latency_pairs.csv")

    if visual_data:
        labels, adj_ps = zip(*visual_data)
        plt.figure(figsize=(10, 5))
        bars = plt.bar(labels, adj_ps, color="mediumpurple", edgecolor="white")
        plt.axhline(0.05, color="red", linestyle="--", label="α = 0.05")
        plt.ylabel("Holm-adjusted p-value (higher = less evidence)")
        plt.title(
            f"Holm-adjusted p-values for latency pairs\n({visual_question})"
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, max(adj_ps) * 1.1)
        plt.legend()
        plt.tight_layout()
        outfile = "posthoc_holm_pvalues_first_question.png"
        savefig(outfile, dpi=300)
        plt.close()

    if heatmap_matrix is not None:
        plt.figure(figsize=(6, 5))
        data = heatmap_matrix.values.copy()
        # Show only the lower triangle (upper triangle is redundant)
        triu_indices = np.triu_indices_from(data, k=1)
        data[triu_indices] = np.nan
        data_masked = np.ma.masked_invalid(data)
        vmax = max(np.nanmax(data), 0.1)
        cmap = plt.get_cmap("plasma")
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        im = plt.imshow(data_masked, cmap=cmap, norm=norm)
        plt.colorbar(im, label="Holm-adjusted p-value (higher = less evidence)")
        plt.xticks(
            ticks=np.arange(len(latency_order)),
            labels=latency_order,
            rotation=45,
            ha="right",
        )
        plt.yticks(ticks=np.arange(len(latency_order)), labels=latency_order)
        plt.title(f"Pairwise Holm-adjusted p-values\n({visual_question})")
        for i in range(len(latency_order)):
            for j in range(len(latency_order)):
                val = heatmap_matrix.iloc[i, j]
                if not np.isnan(val) and j <= i:
                    rgba = cmap(norm(val))
                    # Perceived luminance for contrast decision
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    text_color = "black" if lum > 0.6 else "white"
                    plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)
        plt.tight_layout()
        outfile = "posthoc_holm_heatmap_first_question.png"
        savefig(outfile, dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
