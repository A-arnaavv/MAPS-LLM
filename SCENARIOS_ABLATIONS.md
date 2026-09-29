# MAPS-LLM v2.1 — Scenario and Ablation Evaluation

This package extends the frozen v2.1 evaluation. It does **not** retune MAPS.

## Scenarios

- `balanced`: frozen canonical workload, unchanged.
- `strict_sla`: deadlines compressed to 72%.
- `cost_pressure`: per-task budgets compressed to 68%.
- `bursty_multiuser`: arrivals deterministically compressed into 25-task bursts.
- `specialist_heavy`: deterministic shift toward reasoning/code tasks with higher quality floors.
- `model_degradation`: strongest simulated resource is slower, less reliable, and lower quality.

These are synthetic controlled stress tests, not claims about measured production LLM behavior.

## Ablations

- `MAPS_FULL`
- `NO_AGGREGATION`
- `NO_FAIRNESS_DEBT`
- `NO_UNCERTAINTY`
- `SCALAR_NO_PARETO`

The purpose is causal component analysis: determine which MAPS mechanisms materially
affect quality, resource cost, SLA behavior, and user-level outcomes.

## First run a smoke test

```bash
PYTHONPATH=. python3 experiments/run_scenarios_ablations.py \
  --tasks 100 --seeds 3 \
  --outdir results/v21_sa_smoke
```

This is 6 scenarios × 5 variants × 3 seeds = 90 runs.

Do not run the full 600-run evaluation until the smoke output has been inspected.

## Full evaluation after validation

```bash
PYTHONPATH=. python3 experiments/run_scenarios_ablations.py \
  --tasks 500 --seeds 20 \
  --outdir results/v21_sa_full
```

That is 6 × 5 × 20 = 600 runs.
