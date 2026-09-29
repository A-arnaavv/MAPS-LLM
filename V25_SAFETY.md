# MAPS-LLM v2.5 — Endogenous regime-aware safety

The scheduler is not told scenario labels. It maintains decayed system stress:

    stress_t = 0.90*stress_(t-1)
             + 0.10*(0.55*SLA_failure + 0.25*quality_floor_failure + 0.20*budget_failure)

Candidate operational risk:

    O(p) = 0.55*normalized_latency + 0.25*service_risk + 0.20*escalation_probability

As stress grows, the admissible operational-risk region contracts.

When safety changes a v2.4 fairness-safe decision, v2.5 enforces:

    O(p_v2.5) <= O(p_v2.4-counterfactual)

This is a predicted-risk invariant, not a guarantee about realized stochastic outcomes.

Run:

```bash
python3 -m unittest \
  tests/test_v2.py \
  tests/test_v211_diagnostics.py \
  tests/test_v22_fairness.py \
  tests/test_v23_fairness.py \
  tests/test_v24_fairness.py \
  tests/test_v25_safety.py

PYTHONPATH=. python3 experiments/diagnose_v25_safety.py \
  --tasks 100 --seeds 3 \
  --outdir results/v25_safety_diag
```

Do not run a large experiment until:
- safety_changed > 0
- changed_to_lower_operational_risk > 0
- changed_to_higher_operational_risk = 0
- safety_violations = 0
