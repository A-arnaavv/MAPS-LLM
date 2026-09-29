# MAPS-LLM v2.2 — Plan-sensitive fairness

The fairness objective is now plan-sensitive:

`(1 + 0.35 * debt) * (0.70 * quality_shortfall + 0.30 * SLA_risk)`

This is an algorithmic development change, not retrospective tuning of v2.1 results. First verify that debt changes relative plan penalties and actual selections before any larger scenario run.

```bash
python3 -m unittest tests/test_v2.py tests/test_v211_diagnostics.py tests/test_v22_fairness.py
PYTHONPATH=. python3 experiments/diagnose_v22_fairness.py --tasks 100 --seeds 3 --outdir results/v22_fairness_diag
```
