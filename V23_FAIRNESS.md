# MAPS-LLM v2.3 — Debt-conditioned fairness admissibility

This is an algorithmic development release. It replaces the v2.2 multiplicative
fairness penalty with a structural candidate-admissibility rule.

For each plan:

    R(p) = 0.70 * predicted quality shortfall
         + 0.30 * predicted SLA risk

For user debt D:

    threshold(D) =
        R_min + (R_max - R_min) * exp(-gamma * D)

A plan is fairness-admissible when:

    R(p) <= threshold(D)

with gamma = 0.70.

Properties:
- D = 0 leaves the complete candidate set unchanged.
- Higher debt progressively excludes relatively high-risk plans.
- The rule is scale-relative to the candidate set, not an arbitrary absolute
  risk cutoff.
- Pareto filtering happens only after fairness admissibility.

The diagnostic compares the selected plan under the exact same scheduler state
with fairness enabled versus disabled. Before any scenario experiment, require:

1. selected_plan_changed > 0;
2. changed_to_lower_risk > 0;
3. changed_to_higher_risk = 0.

Run:

```bash
python3 -m unittest \
  tests/test_v2.py \
  tests/test_v211_diagnostics.py \
  tests/test_v22_fairness.py \
  tests/test_v23_fairness.py

PYTHONPATH=. python3 experiments/diagnose_v23_fairness.py \
  --tasks 100 --seeds 3 \
  --outdir results/v23_fairness_diag
```

Do not run the 90-run or 600-run scenario evaluation until the diagnostic has
been inspected.
