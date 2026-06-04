"""
Usage: python evaluation/generate_tables.py evaluation/experiments/exp1.csv
Prints booktabs LaTeX tables to stdout and saves:
  expN_table.tex      — main results table (enhanced with QD significance + KW footnote)
  expN_sig_table.tex  — pairwise significance table for all condition pairs × metrics
"""
import sys
import re
import pathlib
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal


def fmt_mean_std(vals, decimals):
    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return r"\textemdash{}"
    m = vals.mean()
    s = vals.std(ddof=1) if len(vals) > 1 else 0.0
    fmt = f".{decimals}f"
    return f"${m:{fmt}} \\pm {s:{fmt}}$"


def bold(s):
    return r"\textbf{" + s + "}"


def clean_label(label):
    return label.replace("_", " ")


def holm_bonferroni_vs_baseline(df, metric, baseline, conditions,
                                solved_only=False, global_optimum=None):
    """Returns dict[condition -> bool] for significance vs baseline."""
    non_baseline = [c for c in conditions if c != baseline]
    if not non_baseline:
        return {}

    def get_vals(cond):
        mask = df["condition_label"] == cond
        if solved_only and global_optimum is not None:
            mask &= df["best_fitness_reached"] == global_optimum
        return df[mask][metric].dropna().values

    base_vals = get_vals(baseline)
    results = []
    for c in non_baseline:
        cond_vals = get_vals(c)
        if len(base_vals) < 2 or len(cond_vals) < 2:
            results.append((c, 1.0))
            continue
        _, p = mannwhitneyu(base_vals, cond_vals, alternative="two-sided")
        results.append((c, p))

    n_pairs = len(results)
    results.sort(key=lambda r: r[1])

    sig = {}
    for rank, (c, p_raw) in enumerate(results, 1):
        p_adj = min(p_raw * (n_pairs - rank + 1), 1.0)
        sig[c] = p_adj < 0.05
    return sig


def _kruskal_p(df, metric, conditions):
    """Kruskal-Wallis p-value across all conditions for a metric."""
    groups = [df[df["condition_label"] == c][metric].dropna().values for c in conditions]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return float("nan")
    try:
        _, p = kruskal(*groups)
    except ValueError:
        return float("nan")
    return p


def _fmt_p(p):
    """Format a p-value; bold if significant."""
    if np.isnan(p):
        return "n/a"
    s = f"p{{=}}{p:.3f}"
    return f"\\textbf{{{s}}}" if p < 0.05 else s


def generate_table(csv_path):
    csv_path = pathlib.Path(csv_path)
    m = re.search(r"exp(\d+)", csv_path.stem)
    exp_num = m.group(1) if m else csv_path.stem

    df = pd.read_csv(csv_path)
    global_optimum = df["best_fitness_reached"].max()
    conditions = list(dict.fromkeys(df["condition_label"].tolist()))
    baseline = conditions[0]

    sig_best = holm_bonferroni_vs_baseline(
        df, "best_fitness_reached", baseline, conditions)
    sig_gen = holm_bonferroni_vs_baseline(
        df, "gen_best_fitness", baseline, conditions,
        solved_only=True, global_optimum=global_optimum)
    sig_qd = holm_bonferroni_vs_baseline(
        df, "final_qd_score", baseline, conditions)

    # Kruskal-Wallis omnibus p-values
    kw_best = _kruskal_p(df, "best_fitness_reached", conditions)
    kw_qd   = _kruskal_p(df, "final_qd_score", conditions)
    solved_conds = [c for c in conditions
                    if len(df[(df["condition_label"] == c) &
                              (df["best_fitness_reached"] == global_optimum)]) >= 2]
    kw_gen = _kruskal_p(
        df[df["best_fitness_reached"] == global_optimum],
        "gen_best_fitness", solved_conds) if len(solved_conds) >= 2 else float("nan")

    rows_data = []
    for c in conditions:
        sub = df[df["condition_label"] == c]
        solved_sub = sub[sub["best_fitness_reached"] == global_optimum]
        n_solved = len(solved_sub)
        n_total = len(sub)

        best_vals = sub["best_fitness_reached"].values
        qd_vals = sub["final_qd_score"].values
        ext_vals = sub["extinction_count"].values
        interval_vals = sub["mean_extinction_interval"].values
        gen_vals = solved_sub["gen_best_fitness"].values if n_solved > 0 else np.array([])

        def is_blank(arr):
            arr = np.array(arr, dtype=float)
            return np.nansum(np.abs(arr)) == 0

        rows_data.append({
            "condition": c,
            "best_mean": float(np.nanmean(best_vals)),
            "best_str": fmt_mean_std(best_vals, 2),
            "solved_n": n_solved,
            "solved_total": n_total,
            "solved_str": f"{n_solved}/{n_total}",
            "gen_mean": float(np.nanmean(gen_vals)) if len(gen_vals) > 0 else np.inf,
            "gen_str": fmt_mean_std(gen_vals, 0) if len(gen_vals) > 0 else r"\textemdash{}",
            "qd_mean": float(np.nanmean(qd_vals)),
            "qd_str": fmt_mean_std(qd_vals, 2),
            "ext_str": r"\textemdash{}" if is_blank(ext_vals) else fmt_mean_std(ext_vals, 0),
            "interval_str": r"\textemdash{}" if is_blank(interval_vals) else fmt_mean_std(interval_vals, 0),
        })

    best_best = max(r["best_mean"] for r in rows_data)
    gen_finite = [r["gen_mean"] for r in rows_data if r["gen_mean"] != np.inf]
    best_gen = min(gen_finite) if gen_finite else None
    best_qd = max(r["qd_mean"] for r in rows_data)

    latex_rows = []
    for r in rows_data:
        c = r["condition"]
        label = clean_label(c)

        markers = ""
        if sig_best.get(c, False):
            markers += r"$^\dagger$"
        if sig_gen.get(c, False):
            markers += r"$^\ddagger$"
        if sig_qd.get(c, False):
            markers += r"$^\S$"
        label_cell = label + markers

        best_cell = bold(r["best_str"]) if r["best_mean"] == best_best else r["best_str"]
        gen_cell = (bold(r["gen_str"])
                    if best_gen is not None and r["gen_mean"] == best_gen
                    else r["gen_str"])
        qd_cell = bold(r["qd_str"]) if r["qd_mean"] == best_qd else r["qd_str"]

        latex_rows.append(
            f"    {label_cell} & {best_cell} & "
            f"{gen_cell} & {qd_cell} & {r['ext_str']} & {r['interval_str']} \\\\"
        )

    # Caption note for conditions that didn't solve all seeds
    unsolved = [(r["condition"], r["solved_n"], r["solved_total"])
                for r in rows_data if r["solved_n"] < r["solved_total"]]
    if unsolved:
        parts = ", ".join(
            f"{clean_label(c)} ({n}/{t})" for c, n, t in unsolved
        )
        unsolved_note = f"Not all seeds reached the global optimum: {parts}."
    else:
        unsolved_note = None

    any_dagger  = any(sig_best.get(c, False) for c in conditions if c != baseline)
    any_ddagger = any(sig_gen.get(c, False)  for c in conditions if c != baseline)
    any_section = any(sig_qd.get(c, False)   for c in conditions if c != baseline)

    footnote_parts = []
    if any_dagger:
        footnote_parts.append(r"$^\dagger$~best fitness")
    if any_ddagger:
        footnote_parts.append(r"$^\ddagger$~gen-to-best (solved seeds)")
    if any_section:
        footnote_parts.append(r"$^\S$~QD score")
    if footnote_parts:
        footnote = (
            ", ".join(footnote_parts)
            + r" significantly differ from baseline (Mann-Whitney~U, Holm-Bonferroni, $\alpha{=}0.05$)."
        )
    else:
        footnote = None

    kw_line = (
        r"Kruskal-Wallis (all conditions): "
        f"best fitness {_fmt_p(kw_best)}, "
        f"QD score {_fmt_p(kw_qd)}, "
        f"gen-to-best {_fmt_p(kw_gen)}."
    )

    caption_body = f"Results for Experiment {exp_num}. Mean $\\pm$ std across seeds."
    if unsolved_note:
        caption_body += f" {unsolved_note}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        f"\\caption{{{caption_body}}}",
        f"\\label{{tab:exp{exp_num}}}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        (r"Condition & Best Fitness & Gen.\ to Best"
         r" & QD-Score & Ext.\ Count & Mean Ext.\ Interval \\"),
        (r" & (mean$\pm$std) & (mean$\pm$std)"
         r" & (mean$\pm$std) & (mean$\pm$std) & (mean$\pm$std) \\"),
        r"\midrule",
        *latex_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        f"\\footnotesize\\noindent {kw_line}",
    ]
    if footnote:
        lines.append(f"\\footnotesize\\noindent {footnote}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_significance_table(df, exp_num, global_optimum=None):
    """Pairwise Mann-Whitney U significance table for all condition pairs × key metrics."""
    conditions = list(dict.fromkeys(df["condition_label"].tolist()))
    pairs = list(combinations(conditions, 2))
    metrics = [
        ("best_fitness_reached", "Best Fitness"),
        ("final_qd_score",       "QD Score"),
        ("auc_qd_score",         "AUC QD"),
        ("gen_best_fitness",     "Gen.\\ to Best"),
    ]

    # Compute Holm-Bonferroni corrected significance per metric
    sig: dict[str, dict[tuple, str]] = {}
    for metric_key, _ in metrics:
        df_test = (df[df["best_fitness_reached"] == global_optimum]
                   if metric_key == "gen_best_fitness" and global_optimum is not None
                   else df)
        results = []
        for c1, c2 in pairs:
            x = df_test[df_test["condition_label"] == c1][metric_key].dropna().values
            y = df_test[df_test["condition_label"] == c2][metric_key].dropna().values
            if len(x) < 2 or len(y) < 2:
                results.append((c1, c2, 1.0))
                continue
            _, p = mannwhitneyu(x, y, alternative="two-sided")
            results.append((c1, c2, p))

        results.sort(key=lambda r: r[2])
        n = len(results)
        cell: dict[tuple, str] = {}
        for rank, (c1, c2, p_raw) in enumerate(results, 1):
            p_adj = min(p_raw * (n - rank + 1), 1.0)
            if p_adj < 0.01:
                cell[(c1, c2)] = r"\textbf{**}"
            elif p_adj < 0.05:
                cell[(c1, c2)] = r"*"
            else:
                cell[(c1, c2)] = r"---"
        sig[metric_key] = cell

    col_header = " & ".join(label for _, label in metrics)
    latex_rows = []
    for c1, c2 in pairs:
        cells = []
        for metric_key, _ in metrics:
            cells.append(sig[metric_key].get((c1, c2), r"---"))
        latex_rows.append(
            f"    {clean_label(c1)} & {clean_label(c2)} & " + " & ".join(cells) + r" \\"
        )

    n_metrics = len(metrics)
    col_spec = "ll" + "c" * n_metrics

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        f"\\caption{{Pairwise significance for Experiment {exp_num}."
        r" \textbf{**}~$p{<}0.01$, *~$p{<}0.05$, ---~n.s."
        r" (Mann-Whitney~U, Holm-Bonferroni per metric).}}",
        f"\\label{{tab:exp{exp_num}_sig}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"Cond.\\ A & Cond.\\ B & {col_header} \\\\",
        r"\midrule",
        *latex_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "evaluation/experiments/exp1.csv"
    csv_path = pathlib.Path(path)

    df = pd.read_csv(csv_path)
    m = re.search(r"exp(\d+)", csv_path.stem)
    exp_num = m.group(1) if m else csv_path.stem

    table = generate_table(path)
    print(table)
    tex_path = csv_path.with_name(csv_path.stem + "_table.tex")
    tex_path.write_text(table, encoding="utf-8")
    print(f"\n% Saved to {tex_path}", file=sys.stderr)

    sig_table = generate_significance_table(df, exp_num,
        global_optimum=df["best_fitness_reached"].max())
    print("\n" + sig_table)
    sig_path = csv_path.with_name(csv_path.stem + "_sig_table.tex")
    sig_path.write_text(sig_table, encoding="utf-8")
    print(f"% Saved to {sig_path}", file=sys.stderr)
