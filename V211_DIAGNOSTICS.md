# MAPS-LLM v2.1.1 diagnostic patch

This is an instrumentation-only checkpoint. It does not tune MAPS.

It answers two questions raised by the v2.1 scenario smoke test:

1. Is fairness debt numerically nonzero, and if so does it change objective 5,
   Pareto membership, or the selected plan?
2. Is a planned verifier retained in the execution record even when conditional
   escalation does not occur, so both Beta success and failure updates are learned?

Run:

```bash
python3 -m unittest tests/test_v2.py tests/test_v211_diagnostics.py

PYTHONPATH=. python3 experiments/diagnose_v211.py \
  --tasks 100 --seeds 3 \
  --outdir results/v211_diag_smoke
```

Do not change fairness coefficients based on these diagnostics. First establish whether
the mechanism is inactive because of implementation, objective geometry, or insufficient
decision leverage.
