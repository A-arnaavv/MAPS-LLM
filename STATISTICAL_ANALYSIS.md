# MAPS-LLM v2.1 Statistical Analysis

This package analyzes the already generated paired-seed file
`results/v21_full/v2_summary_by_seed.csv`. It does not rerun or tune the scheduler.

## Primary inferential workflow

For each performance metric:

1. **Friedman test** across all seven methods over the same 20 seeds.
2. **MAPS-vs-baseline paired Wilcoxon signed-rank tests**.
3. **Holm correction** across the six MAPS-vs-baseline comparisons within that metric.
4. **20,000 paired bootstrap resamples** for a 95% confidence interval on the mean
   MAPS advantage.
5. Two paired effect-size summaries:
   - paired Cohen's dz;
   - rank-biserial correlation.

All differences are direction-coded so that **positive advantage means MAPS is better**.
For cost, P95 latency, SLA violation rate, and user-utility gap, lower values are better.
For quality, fairness, worst-user utility, and throughput, higher values are better.

`aggregation_rate` is descriptive only and is not subjected to superiority testing.

## Run

```bash
python3 -m pip install -r requirements.txt

PYTHONPATH=. python3 experiments/statistics_v21.py \
  --csv results/v21_full/v2_summary_by_seed.csv \
  --outdir results/v21_stats
```

Outputs:

- `descriptive_stats.csv`
- `friedman_tests.csv`
- `maps_vs_baselines.csv`
- `maps_vs_baselines_compact.csv`
- `analysis_metadata.json`

## Interpretation discipline

A small p-value does not by itself establish practical importance. Interpret the
Holm-adjusted p-value together with the paired effect size and confidence interval.

Because v2.1 remains a per-task online Pareto plan selector rather than a horizon-level
joint scheduler, statistical significance here supports claims about this implementation
only; it does not establish the claims intended for the future v3 architecture.
