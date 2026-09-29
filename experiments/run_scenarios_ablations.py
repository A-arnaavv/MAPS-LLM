import os
import sys
import json
import argparse
import copy

import numpy as np
import pandas as pd

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from maps_llm.defaults import models
from maps_llm.canonical import load_workload, load_trace
from maps_llm.simulator import execute, summarize
from maps_llm.optimizer import MAPSv2
from maps_llm.entities import Plan
from maps_llm.predict import feasible


# ============================================================
# Controlled scenario transformations
# ============================================================

SCENARIOS = [
    "balanced",
    "strict_sla",
    "cost_pressure",
    "bursty_multiuser",
    "specialist_heavy",
    "model_degradation",
]

ABLATIONS = [
    "MAPS_FULL",
    "NO_AGGREGATION",
    "NO_FAIRNESS_DEBT",
    "NO_UNCERTAINTY",
    "SCALAR_NO_PARETO",
]


def scenario_tasks(tasks, name):
    """
    Deterministic stress transformations of the frozen canonical workloads.
    The underlying workload seed and task identities are preserved.
    """
    x = copy.deepcopy(tasks)

    if name == "balanced":
        return x

    for t in x:

        if name == "strict_sla":
            t.deadline = max(0.45, t.deadline * 0.72)

        elif name == "cost_pressure":
            t.budget = max(0.002, t.budget * 0.68)

        elif name == "bursty_multiuser":
            # Preserve task order and identity, but compress arrivals
            # into deterministic bursts.
            block = t.task_id // 25
            within = t.task_id % 25
            t.arrival = block * 7.0 + within * 0.035

        elif name == "specialist_heavy":
            # Deterministically shift a majority of selected requests
            # toward specialist-oriented task categories.
            if t.task_id % 5 in (0, 1, 2):
                t.kind = "reasoning" if t.task_id % 2 == 0 else "coding"
                t.quality_floor = min(0.94, t.quality_floor + 0.06)

        elif name == "model_degradation":
            # Workload unchanged; the model transformation is applied
            # in scenario_models().
            pass

        else:
            raise ValueError(f"Unknown scenario: {name}")

    return x


def scenario_models(name):
    """
    Return the model pool associated with the requested scenario.
    """
    ms = copy.deepcopy(models())

    if name == "model_degradation":
        # Degrade the model with the highest average base quality.
        k = max(
            range(len(ms)),
            key=lambda i: sum(ms[i].quality.values()) / len(ms[i].quality)
        )

        from dataclasses import replace

        m = ms[k]

        ms[k] = replace(
            m,
            latency=m.latency * 1.65,
            reliability=max(0.70, m.reliability - 0.12),
            quality={
                task_kind: max(0.50, q - 0.07)
                for task_kind, q in m.quality.items()
            },
        )

    return ms


# ============================================================
# v2.5-consistent ablations
# ============================================================

class NoAggregation(MAPSv2):
    """
    MAPS-LLM v2.5 with multi-model aggregation removed.

    Only candidate generation is changed. Fairness, regime-aware safety,
    uncertainty-aware contextual selection, Pareto filtering, and all
    learned quantities remain identical to MAPS_FULL.
    """

    def _candidate_plans(self, t, ms):
        feasible_idx = sorted(
            i for i, m in enumerate(ms)
            if feasible(t, m)
        )

        plans = [
            Plan(i, -1)
            for i in feasible_idx
        ]

        return feasible_idx, plans


class NoFairnessDebt(MAPSv2):
    """
    MAPS-LLM v2.5 with the fairness-debt intervention disabled.

    The debt state may still be updated internally, but it cannot modify
    candidate admissibility. All other v2.5 mechanisms remain enabled.
    """

    def __init__(self, seed=1, **kwargs):
        super().__init__(
            seed=seed,
            fairness_enabled=False,
            **kwargs
        )


class NoUncertainty(MAPSv2):
    """
    MAPS-LLM v2.5 with the explicit uncertainty bonus removed.

    This sets beta=0, eliminating beta * uncertainty from contextual
    ranking while retaining the low-observation diversification term
    and Pareto crowding term.
    """

    def __init__(self, seed=1, **kwargs):
        super().__init__(
            seed=seed,
            beta=0.0,
            **kwargs
        )


class ScalarNoPareto(MAPSv2):
    """
    MAPS-LLM v2.5 with Pareto candidate reduction removed.

    The same fairness and safety admissibility stages are retained.
    Within each stage, all admissible candidates are ranked directly by
    the contextual scalar score instead of first restricting selection
    to the nondominated set.

    Because Pareto reduction is disabled, the Pareto crowding bonus is
    also omitted. Uncertainty and low-observation diversification remain.
    """

    @staticmethod
    def _stable_plan_key(p):
        return (int(p.primary), int(p.verifier))

    @staticmethod
    def _q(x):
        return float(np.round(float(x), 12))

    def _select_from_indices(self, t, rows, indices):
        ranked = []
        details = []

        for oi in indices:
            r = rows[oi]

            cost_ratio = r["c"] / max(t.budget, 0.001)
            latency_ratio = r["l"] / max(t.deadline, 0.1)

            exploit = (
                r["learned"]
                - 0.06 * cost_ratio
                - 0.035 * latency_ratio
                - 0.15 * r["sla_risk"]
                - 0.12 * r["service_risk"]
            )

            diversification_bonus = (
                0.035 if r["n"] < 3 else 0.0
            )

            score = self._q(
                exploit
                + self.beta * r["unc"]
                + diversification_bonus
            )

            p = r["plan"]

            ranked.append(
                (
                    -score,
                    self._stable_plan_key(p),
                    int(oi),
                    p,
                )
            )

            details.append(
                dict(
                    original_index=int(oi),
                    plan=[p.primary, p.verifier],
                    service_risk=float(r["service_risk"]),
                    operational_risk=float(r["operational_risk"]),
                    score=float(score),
                )
            )

        ranked.sort()

        _, _, idx, plan = ranked[0]

        # In the scalar ablation, every supplied candidate participates
        # directly in ranking; there is no nondominated subset.
        retained = list(range(len(indices)))

        return plan, int(idx), retained, details


# ============================================================
# Scheduler factory
# ============================================================

def policy(name, seed):
    scheduler_seed = 51000 + seed

    if name == "MAPS_FULL":
        return MAPSv2(scheduler_seed)

    if name == "NO_AGGREGATION":
        return NoAggregation(scheduler_seed)

    if name == "NO_FAIRNESS_DEBT":
        return NoFairnessDebt(scheduler_seed)

    if name == "NO_UNCERTAINTY":
        return NoUncertainty(scheduler_seed)

    if name == "SCALAR_NO_PARETO":
        return ScalarNoPareto(scheduler_seed)

    raise ValueError(f"Unknown ablation: {name}")


# ============================================================
# Experiment
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--tasks",
        type=int,
        default=500
    )

    ap.add_argument(
        "--seeds",
        type=int,
        default=20
    )

    # Deliberately use a NEW directory so the corrected v2.5 results
    # cannot be confused with the stale v2.1-era ablation output.
    ap.add_argument(
        "--outdir",
        default="results/v25_scenarios_ablations"
    )

    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)

    rows = []

    for scenario in SCENARIOS:

        for seed in range(a.seeds):

            base_tasks = scenario_tasks(
                load_workload(seed, limit=a.tasks),
                scenario
            )

            trace = load_trace(seed)
            base_models = scenario_models(scenario)

            for ablation in ABLATIONS:

                records = execute(
                    copy.deepcopy(base_tasks),
                    copy.deepcopy(base_models),
                    policy(ablation, seed),
                    trace=trace,
                )

                summary = summarize(records, 5)

                rows.append(
                    dict(
                        scenario=scenario,
                        ablation=ablation,
                        seed=seed,
                        **summary
                    )
                )

    df = pd.DataFrame(rows)

    by_seed_path = os.path.join(
        a.outdir,
        "scenario_ablation_by_seed.csv"
    )

    df.to_csv(
        by_seed_path,
        index=False
    )

    metrics = [
        "quality",
        "cost_per_task",
        "p95_latency",
        "sla_violation_rate",
        "fairness",
        "user_utility_gap",
        "worst_user_utility",
        "aggregation_rate",
        "throughput",
    ]

    aggregate = (
        df.groupby(["scenario", "ablation"])[metrics]
        .agg(["mean", "std"])
    )

    aggregate_path = os.path.join(
        a.outdir,
        "scenario_ablation_aggregate.csv"
    )

    aggregate.to_csv(aggregate_path)

    metadata = {
        "maps_version": "v2.5",
        "tasks_per_run": int(a.tasks),
        "seeds": int(a.seeds),
        "scenarios": SCENARIOS,
        "ablations": ABLATIONS,
        "total_runs": int(
            len(SCENARIOS)
            * len(ABLATIONS)
            * a.seeds
        ),
        "ablation_definitions": {
            "MAPS_FULL":
                "Complete MAPS-LLM v2.5.",
            "NO_AGGREGATION":
                "Removes multi-model aggregation candidates only.",
            "NO_FAIRNESS_DEBT":
                "Disables fairness-based candidate admissibility.",
            "NO_UNCERTAINTY":
                "Removes the explicit beta*uncertainty ranking bonus.",
            "SCALAR_NO_PARETO":
                "Removes nondominated filtering and Pareto crowding; "
                "ranks every admissible candidate using the scalar "
                "contextual score.",
        },
    }

    metadata_path = os.path.join(
        a.outdir,
        "ablation_metadata.json"
    )

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("MAPS-LLM v2.5 scenario/ablation experiment complete")
    print("completed runs:", len(df))
    print("by-seed results:", by_seed_path)
    print("aggregate results:", aggregate_path)
    print("metadata:", metadata_path)

    print()
    print(aggregate.round(5).to_string())


if __name__ == "__main__":
    main()