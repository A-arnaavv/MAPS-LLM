# MAPS-LLM

Research implementation and reproducibility package for the manuscript:

**MAPS-LLM: Multiobjective Adaptive Pareto Scheduling for Fair and Risk-Aware Multi-User LLM Serving**

MAPS-LLM is a simulation-based framework for online scheduling of heterogeneous LLM services. The implementation combines multiobjective Pareto plan selection, selective two-model aggregation, uncertainty-aware prediction, history-dependent user fairness, and endogenous regime-aware operational safety.

> **Evaluation scope.** The experiments in this repository use controlled simulation and synthetic stress regimes. They are not measurements of production LLM endpoints and should not be interpreted as production latency, pricing, or quality predictions.

## Repository contents

- `maps_llm/` — scheduler, Pareto logic, predictors, fairness, baselines, simulator, and workload utilities.
- `frozen_v24/` — frozen v2.4 scheduler used for paired comparison with v2.5.
- `experiments/` — baseline evaluation, six-regime ablations, fairness/safety diagnostics, paired v2.5-vs-v2.4 evaluation, and statistical analysis.
- `canonical_data/` — canonical workloads and stochastic traces for reproducibility.
- `configs/` — experiment configuration.
- `tests/` — unit/regression tests for the scheduler and fairness/safety mechanisms.
- `results/` — retained seed-level and aggregate outputs from the reported evaluation workflow.
- `REPRODUCIBILITY.md`, `SCENARIOS_ABLATIONS.md`, `STATISTICAL_ANALYSIS.md`, `V24_FAIRNESS.md`, `V25_SAFETY.md`, `V25_VS_V24.md` — detailed protocol notes.

## Requirements

- Python 3.11+ recommended
- NumPy 2.4.6
- pandas 3.0.5
- SciPy 1.17.0

Create an isolated environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Verify the implementation

Run the complete test suite before reproducing experiments:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Main experiments

### 1. Baseline comparison

The baseline workflow evaluates MAPS-LLM and the implemented comparison schedulers over paired seeds.

```bash
PYTHONPATH=. python3 experiments/run_v2.py \
  --tasks 500 --seeds 20 \
  --outdir results/v21_full

PYTHONPATH=. python3 experiments/statistics_v21.py \
  --csv results/v21_full/v2_summary_by_seed.csv \
  --outdir results/v21_stats
```

The statistical workflow includes paired Wilcoxon tests, Holm correction, and 20,000 paired bootstrap resamples; see `STATISTICAL_ANALYSIS.md`.

### 2. Six-regime ablation evaluation

The controlled regimes are `balanced`, `strict_sla`, `cost_pressure`, `bursty_multiuser`, `specialist_heavy`, and `model_degradation`. The evaluated configurations are `MAPS_FULL`, `NO_AGGREGATION`, `NO_FAIRNESS_DEBT`, `NO_UNCERTAINTY`, and `SCALAR_NO_PARETO`.

```bash
PYTHONPATH=. python3 experiments/run_scenarios_ablations.py \
  --tasks 500 --seeds 20 \
  --outdir results/v25_scenarios_ablations

PYTHONPATH=. python3 experiments/statistics_v25_ablations.py \
  --csv results/v25_scenarios_ablations/scenario_ablation_by_seed.csv \
  --outdir results/v25_scenarios_ablations_stats
```

See `SCENARIOS_ABLATIONS.md` for the scenario definitions.

### 3. Fairness diagnostic

The repository retains the development trail for the fairness mechanism. The final structural fairness implementation is documented in `V24_FAIRNESS.md` and exercised by `tests/test_v24_fairness.py` and `experiments/diagnose_v24_fairness.py`.

```bash
PYTHONPATH=. python3 experiments/diagnose_v24_fairness.py \
  --tasks 100 --seeds 3 \
  --outdir results/v24_fairness_diag
```

### 4. Regime-aware safety diagnostic

```bash
PYTHONPATH=. python3 experiments/diagnose_v25_safety.py \
  --tasks 100 --seeds 3 \
  --outdir results/v25_safety_diag
```

The safety mechanism and its predicted-risk preservation invariant are documented in `V25_SAFETY.md`.

### 5. Paired quality-oriented vs regime-aware comparison

The final paired comparison uses four scenarios, 20 paired seeds, and 500 tasks per run. In the implementation history, the frozen v2.4 scheduler is the quality-oriented reference and v2.5 adds endogenous regime-aware safety.

```bash
PYTHONPATH=. python3 experiments/run_v25_vs_v24.py \
  --tasks 500 --seeds 20 \
  --outdir results/v25_vs_v24

PYTHONPATH=. python3 experiments/statistics_v25_vs_v24.py \
  --csv results/v25_vs_v24/v25_vs_v24_by_seed.csv \
  --outdir results/v25_vs_v24_stats
```

See `V25_VS_V24.md` for the paired design and statistical interpretation.

## Reproducibility

Canonical workloads and stochastic traces are stored under `canonical_data/`. The reproducibility utilities compare input hashes and ordered schedule records to distinguish workload/trace differences from policy, runtime, or evaluator differences. See `REPRODUCIBILITY.md` for details.

For a quick smoke check before a full experiment, reduce `--tasks` and `--seeds`, for example `--tasks 100 --seeds 3`.

## Results and interpretation

The repository contains retained experiment outputs under `results/`. The manuscript should be treated as the authoritative source for the reported tables, figures, experimental interpretation, and limitations. Because the evaluation is simulation-based, absolute values are specific to the simulator and configured scenarios.

## Code availability

This repository contains the simulation and experimental evaluation code for MAPS-LLM. A fixed release/tag should be created for the version associated with a journal submission so that the reviewed code remains reproducible even if later development continues.

## Citation

If you use this repository, please cite the associated MAPS-LLM manuscript. Bibliographic metadata can be updated with the DOI and final publication details after acceptance. A `CITATION.cff` file is included for GitHub citation support.

## License

Released under the MIT License. See `LICENSE`.
