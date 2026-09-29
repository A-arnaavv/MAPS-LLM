import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METRICS = {
    "quality": True,
    "cost_per_task": False,
    "p95_latency": False,
    "sla_violation_rate": False,
    "user_utility_gap": False,
    "worst_user_utility": True,
    "throughput": True,
}


ABLATIONS = [
    "NO_AGGREGATION",
    "NO_FAIRNESS_DEBT",
    "NO_UNCERTAINTY",
    "SCALAR_NO_PARETO",
]


def holm_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)

    order = np.argsort(p, kind="mergesort")
    adjusted = np.empty(m, dtype=float)

    running = 0.0

    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running

    return adjusted


def bootstrap_ci(values, rng, B=20000):
    values = np.asarray(values, dtype=float)

    idx = rng.integers(
        0,
        len(values),
        size=(B, len(values))
    )

    means = values[idx].mean(axis=1)

    lo, hi = np.quantile(
        means,
        [0.025, 0.975]
    )

    return float(lo), float(hi)


def paired_dz(values):
    values = np.asarray(values, dtype=float)

    sd = values.std(ddof=1)

    if abs(sd) <= 1e-15:
        return 0.0

    return float(values.mean() / sd)


def rank_biserial(values):
    values = np.asarray(values, dtype=float)

    nz = values[np.abs(values) > 1e-15]

    if len(nz) == 0:
        return 0.0

    absvals = np.abs(nz)

    order = np.argsort(
        absvals,
        kind="mergesort"
    )

    ranks = np.empty(len(nz), dtype=float)

    sorted_vals = absvals[order]

    i = 0

    while i < len(sorted_vals):

        j = i + 1

        while (
            j < len(sorted_vals)
            and sorted_vals[j] == sorted_vals[i]
        ):
            j += 1

        avg_rank = (i + 1 + j) / 2.0

        ranks[order[i:j]] = avg_rank

        i = j

    wp = ranks[nz > 0].sum()
    wm = ranks[nz < 0].sum()

    if wp + wm == 0:
        return 0.0

    return float(
        (wp - wm) / (wp + wm)
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--csv",
        default=(
            "results/v25_scenarios_ablations/"
            "scenario_ablation_by_seed.csv"
        ),
    )

    ap.add_argument(
        "--outdir",
        default="results/v25_scenarios_ablations_stats",
    )

    ap.add_argument(
        "--bootstrap",
        type=int,
        default=20000,
    )

    ap.add_argument(
        "--rng-seed",
        type=int,
        default=20260911,
    )

    args = ap.parse_args()

    os.makedirs(
        args.outdir,
        exist_ok=True
    )

    df = pd.read_csv(args.csv)

    rng = np.random.default_rng(
        args.rng_seed
    )

    results = []

    for scenario in sorted(
        df["scenario"].unique()
    ):

        g = df[
            df["scenario"] == scenario
        ]

        scenario_rows = []
        scenario_pvals = []

        for ablation in ABLATIONS:

            for metric, higher_is_better in METRICS.items():

                full = (
                    g[g["ablation"] == "MAPS_FULL"]
                    .set_index("seed")[metric]
                    .sort_index()
                )

                ab = (
                    g[g["ablation"] == ablation]
                    .set_index("seed")[metric]
                    .sort_index()
                )

                common = full.index.intersection(
                    ab.index
                )

                full_values = (
                    full.loc[common]
                    .to_numpy(dtype=float)
                )

                ab_values = (
                    ab.loc[common]
                    .to_numpy(dtype=float)
                )

                # Positive advantage means MAPS_FULL is better.
                if higher_is_better:
                    advantage = (
                        full_values - ab_values
                    )
                else:
                    advantage = (
                        ab_values - full_values
                    )

                raw_difference = (
                    full_values - ab_values
                )

                lo, hi = bootstrap_ci(
                    advantage,
                    rng,
                    args.bootstrap,
                )

                if np.all(
                    np.abs(advantage) <= 1e-15
                ):
                    stat = 0.0
                    p = 1.0
                else:
                    stat, p = wilcoxon(
                        advantage,
                        zero_method="pratt",
                        alternative="two-sided",
                        method="auto",
                    )

                    stat = float(stat)
                    p = float(p)

                row = {
                    "scenario": scenario,
                    "ablation": ablation,
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "n_pairs": len(common),
                    "maps_full_mean": float(
                        full_values.mean()
                    ),
                    "ablation_mean": float(
                        ab_values.mean()
                    ),
                    "raw_difference_full_minus_ablation":
                        float(
                            raw_difference.mean()
                        ),
                    "advantage_positive_means_full_better":
                        float(
                            advantage.mean()
                        ),
                    "advantage_ci95_low": lo,
                    "advantage_ci95_high": hi,
                    "paired_cohen_dz_advantage":
                        paired_dz(
                            advantage
                        ),
                    "rank_biserial_advantage":
                        rank_biserial(
                            advantage
                        ),
                    "wilcoxon_stat": stat,
                    "wilcoxon_p": p,
                }

                scenario_rows.append(row)
                scenario_pvals.append(p)

        # Conservative correction over all
        # ablation × metric comparisons within a scenario.
        adjusted = holm_adjust(
            scenario_pvals
        )

        for row, padj in zip(
            scenario_rows,
            adjusted
        ):
            row[
                "holm_p_within_scenario"
            ] = float(padj)

            row[
                "significant_0_05"
            ] = bool(padj < 0.05)

            results.append(row)

    R = pd.DataFrame(results)

    all_path = os.path.join(
        args.outdir,
        "v25_ablation_stats.csv"
    )

    R.to_csv(
        all_path,
        index=False
    )

    focus_metrics = [
        "quality",
        "cost_per_task",
        "p95_latency",
        "sla_violation_rate",
        "user_utility_gap",
        "worst_user_utility",
    ]

    focus = R[
        R["metric"].isin(
            focus_metrics
        )
    ].copy()

    focus_path = os.path.join(
        args.outdir,
        "v25_ablation_focus.csv"
    )

    focus.to_csv(
        focus_path,
        index=False
    )

    print(
        "MAPS-LLM v2.5 paired "
        "ablation statistics complete"
    )

    print(
        "scenarios:",
        df["scenario"].nunique()
    )

    print(
        "seeds:",
        df["seed"].nunique()
    )

    print(
        "comparisons:",
        len(R)
    )

    print()

    print(
        focus[
            [
                "scenario",
                "ablation",
                "metric",
                "maps_full_mean",
                "ablation_mean",
                "advantage_positive_means_full_better",
                "advantage_ci95_low",
                "advantage_ci95_high",
                "holm_p_within_scenario",
                "significant_0_05",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print()
    print("wrote:", all_path)
    print("wrote:", focus_path)


if __name__ == "__main__":
    main()