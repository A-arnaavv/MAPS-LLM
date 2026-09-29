# Reproducibility protocol

v2.0.2 freezes experimental exogenous randomness into byte-identical canonical
JSON inputs.

Input hashes are computed from canonical JSON subsets for the selected task
count. Schedule hashes are computed from the complete ordered per-request
record list.

Interpretation:
- workload hash differs -> task input mismatch
- trace hash differs -> stochastic input mismatch
- both inputs match but schedule differs -> policy/numeric/runtime discrepancy
- schedule matches but reported aggregate differs -> evaluator discrepancy

No optimization parameter was intentionally tuned in this patch.
