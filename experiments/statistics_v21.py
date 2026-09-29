import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

HIGHER_BETTER = {
    "quality": True,
    "cost_per_task": False,
    "p95_latency": False,
    "sla_violation_rate": False,
    "fairness": True,
    "user_utility_gap": False,
    "worst_user_utility": True,
    "throughput": True,
}

DESCRIPTIVE_ONLY = ["aggregation_rate"]

MAPS_NAME = "MAPS_LLM_V2"


def holm_adjust(pvals):
    """Holm step-down adjusted p-values, preserving input order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=float)
    order = np.argsort(p, kind="mergesort")
    adj_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m-rank) * p[idx])
        running = max(running, val)
        adj_sorted[rank] = running
    out = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        out[idx] = adj_sorted[rank]
    return out


def paired_bootstrap_ci(diff, rng, B=20000, alpha=0.05):
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    idx = rng.integers(0, n, size=(B, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha/2, 1-alpha/2])
    return float(lo), float(hi)


def paired_dz(diff):
    diff = np.asarray(diff, dtype=float)
    sd = diff.std(ddof=1)
    if abs(sd) < 1e-15:
        if abs(diff.mean()) < 1e-15:
            return 0.0
        return float(np.sign(diff.mean()) * np.inf)
    return float(diff.mean() / sd)


def rank_biserial_advantage(diff):
    """
    Rank-biserial correlation for paired differences.
    diff is direction-coded so positive means MAPS is better.
    """
    d = np.asarray(diff, dtype=float)
    nz = d[np.abs(d) > 1e-15]
    if len(nz) == 0:
        return 0.0
    order = np.argsort(np.abs(nz), kind="mergesort")
    ranks = np.empty(len(nz), dtype=float)
    # average ranks for exact ties in abs differences
    vals = np.abs(nz)[order]
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals) and vals[j] == vals[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    wplus = ranks[nz > 0].sum()
    wminus = ranks[nz < 0].sum()
    denom = wplus + wminus
    return 0.0 if denom == 0 else float((wplus - wminus) / denom)


def safe_wilcoxon(diff):
    d = np.asarray(diff, dtype=float)
    if np.all(np.abs(d) <= 1e-15):
        return 0.0, 1.0
    stat, p = wilcoxon(d, zero_method="pratt", alternative="two-sided", method="auto")
    return float(stat), float(p)


def validate(df):
    need = {"method", "seed"} | set(HIGHER_BETTER) | set(DESCRIPTIVE_ONLY)
    missing = sorted(need - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if MAPS_NAME not in set(df["method"]):
        raise ValueError(f"{MAPS_NAME} not found")
    counts = df.groupby("method")["seed"].nunique()
    if counts.nunique() != 1:
        raise ValueError(f"Methods do not have equal numbers of seeds: {counts.to_dict()}")
    # paired-design check
    seedsets = [set(g["seed"]) for _, g in df.groupby("method")]
    if any(s != seedsets[0] for s in seedsets[1:]):
        raise ValueError("Methods do not share the exact same paired seed set.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/v21_full/v2_summary_by_seed.csv")
    ap.add_argument("--outdir", default="results/v21_stats")
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--rng-seed", type=int, default=20260910)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    validate(df)

    methods = sorted(df["method"].unique())
    seeds = sorted(df["seed"].unique())
    rng = np.random.default_rng(args.rng_seed)

    # Descriptive statistics.
    all_metrics = list(HIGHER_BETTER) + DESCRIPTIVE_ONLY
    desc = (
        df.groupby("method")[all_metrics]
          .agg(["mean", "std"])
          .sort_index()
    )
    desc.to_csv(out / "descriptive_stats.csv")

    # Global Friedman tests.
    fried_rows = []
    for metric in HIGHER_BETTER:
        wide = df.pivot(index="seed", columns="method", values=metric).loc[seeds, methods]
        stat, p = friedmanchisquare(*[wide[m].to_numpy() for m in methods])
        fried_rows.append({
            "metric": metric,
            "higher_is_better": HIGHER_BETTER[metric],
            "friedman_chi2": float(stat),
            "df": len(methods)-1,
            "p_value": float(p),
        })
    fried = pd.DataFrame(fried_rows)
    fried["holm_across_metrics"] = holm_adjust(fried["p_value"].to_numpy())
    fried.to_csv(out / "friedman_tests.csv", index=False)

    # MAPS-vs-each-baseline paired comparisons.
    pair_rows = []
    for metric, hib in HIGHER_BETTER.items():
        wide = df.pivot(index="seed", columns="method", values=metric).loc[seeds, methods]
        maps = wide[MAPS_NAME].to_numpy()
        metric_rows = []
        metric_p = []
        for base in methods:
            if base == MAPS_NAME:
                continue
            b = wide[base].to_numpy()
            # Positive advantage means MAPS is better.
            advantage = maps - b if hib else b - maps
            raw_diff = maps - b
            ci_lo, ci_hi = paired_bootstrap_ci(advantage, rng, B=args.bootstrap)
            wstat, p = safe_wilcoxon(advantage)
            row = {
                "metric": metric,
                "higher_is_better": hib,
                "baseline": base,
                "n_pairs": len(advantage),
                "maps_mean": float(maps.mean()),
                "baseline_mean": float(b.mean()),
                "raw_mean_difference_maps_minus_baseline": float(raw_diff.mean()),
                "mean_advantage_positive_is_maps_better": float(advantage.mean()),
                "advantage_ci95_low": ci_lo,
                "advantage_ci95_high": ci_hi,
                "paired_cohen_dz_advantage": paired_dz(advantage),
                "rank_biserial_advantage": rank_biserial_advantage(advantage),
                "wilcoxon_stat": wstat,
                "wilcoxon_p": p,
            }
            metric_rows.append(row)
            metric_p.append(p)
        adjusted = holm_adjust(metric_p)
        for row, padj in zip(metric_rows, adjusted):
            row["holm_p_within_metric"] = float(padj)
            row["significant_0_05"] = bool(padj < 0.05)
            pair_rows.append(row)

    pair = pd.DataFrame(pair_rows)
    pair.to_csv(out / "maps_vs_baselines.csv", index=False)

    # Compact publication-facing table.
    compact = pair[[
        "metric","baseline","maps_mean","baseline_mean",
        "mean_advantage_positive_is_maps_better",
        "advantage_ci95_low","advantage_ci95_high",
        "rank_biserial_advantage","holm_p_within_metric","significant_0_05"
    ]].copy()
    compact.to_csv(out / "maps_vs_baselines_compact.csv", index=False)

    # Machine-readable metadata.
    meta = {
        "input_csv": args.csv,
        "n_seeds": len(seeds),
        "methods": methods,
        "bootstrap_resamples": args.bootstrap,
        "bootstrap_rng_seed": args.rng_seed,
        "notes": [
            "Positive advantage values always mean MAPS performed better.",
            "Wilcoxon tests are paired by seed.",
            "Holm correction is applied across the six MAPS-vs-baseline comparisons within each metric.",
            "Friedman tests compare all methods jointly for each metric.",
            "Aggregation rate is descriptive only and is not treated as a performance objective."
        ]
    }
    (out/"analysis_metadata.json").write_text(json.dumps(meta, indent=2))

    print("Statistical analysis complete")
    print("seeds:", len(seeds))
    print("methods:", len(methods))
    print("\nFriedman tests:")
    print(fried.round(6).to_string(index=False))
    print("\nMAPS vs baselines (Holm-adjusted):")
    print(compact.round(6).to_string(index=False))
    print("\nwrote:", out)

if __name__ == "__main__":
    main()
