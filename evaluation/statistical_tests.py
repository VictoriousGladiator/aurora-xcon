"""
Usage: python statistical_tests.py results.csv
"""
import sys
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import mannwhitneyu, kruskal

def run_tests(csv_path, metrics=None):
    df = pd.read_csv(csv_path)
    conditions = df["condition_label"].unique()

    if metrics is None:
        metrics = ["final_qd_score", "best_fitness_reached", "auc_qd_score", "gen_best_fitness"]

    for metric in metrics:
        print(f"\n{'='*60}")
        print(f"METRIC: {metric}")
        print(f"{'='*60}")

        # Summary stats
        summary = df.groupby("condition_label")[metric].agg(["median", "mean", "std", "count"])
        print(f"\n{summary.to_string()}\n")

        # Kruskal-Wallis (omnibus test across all conditions)
        if len(conditions) > 2:
            groups = [df[df["condition_label"] == c][metric].values for c in conditions]
            stat, p = kruskal(*groups)
            print(f"Kruskal-Wallis: H={stat:.4f}, p={p:.4f} {'*' if p < 0.05 else ''}")

        # Pairwise Mann-Whitney U
        pairs = list(combinations(conditions, 2))
        n_pairs = len(pairs)
        print(f"\nPairwise Mann-Whitney U (Holm-Bonferroni corrected, {n_pairs} comparisons):")
        print("-" * 60)

        results = []
        for c1, c2 in pairs:
            x = df[df["condition_label"] == c1][metric].values
            y = df[df["condition_label"] == c2][metric].values
            stat, p = mannwhitneyu(x, y, alternative="two-sided")
            results.append((c1, c2, stat, p))

        # Holm-Bonferroni correction
        results.sort(key=lambda r: r[3])  # sort by p-value
        for rank, (c1, c2, stat, p_raw) in enumerate(results, 1):
            p_adj = min(p_raw * (n_pairs - rank + 1), 1.0)
            sig = "*" if p_adj < 0.05 else ""
            print(f"  {c1:25s} vs {c2:25s}  U={stat:6.1f}  p_raw={p_raw:.4f}  p_adj={p_adj:.4f} {sig}")

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "evaluation/final_all_relevant/metrics_summary_final_all_relevant.csv"
    run_tests(csv_path)