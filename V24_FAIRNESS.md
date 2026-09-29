# MAPS-LLM v2.4 — Fairness-safe structural scheduling

v2.4 adds a counterfactual service-risk safety invariant to v2.3.

For every request, MAPS first computes the ordinary no-fairness counterfactual
choice p0 under the exact same learned state.

The debt-conditioned fairness gate is then applied. A second safety filter
retains only fairness-admissible plans satisfying:

    R(p) <= R(p0)

where:

    R(p) = 0.70 * predicted quality shortfall
         + 0.30 * predicted SLA risk

The final MAPS Pareto/crowding/exploration/exploitation selection runs only
inside this fairness-safe set.

Therefore, when fairness changes a decision, the implementation guarantees:

    R(p_fair) <= R(p_no-fair)

This is a predicted-risk invariant, not a guarantee about realized stochastic
outcomes.

Validation criteria before any scenario experiment:

- fairness_changed > 0
- changed_to_lower_risk > 0
- changed_to_higher_risk = 0
- safety_violations = 0

Run:

```bash
python3 -m unittest \
  tests/test_v2.py \
  tests/test_v211_diagnostics.py \
  tests/test_v22_fairness.py \
  tests/test_v23_fairness.py \
  tests/test_v24_fairness.py

PYTHONPATH=. python3 experiments/diagnose_v24_fairness.py \
  --tasks 100 \
  --seeds 3 \
  --outdir results/v24_fairness_diag
```

Do not run the 90-run or 600-run scenario evaluation until this diagnostic
passes on the target machine.
