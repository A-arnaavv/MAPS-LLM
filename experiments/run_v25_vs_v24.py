import os, sys, argparse, copy
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maps_llm.defaults import models
from maps_llm.canonical import load_workload, load_trace
from maps_llm.simulator import execute, summarize
from maps_llm.optimizer import MAPSv2 as MAPSv25
from frozen_v24.optimizer_v24 import MAPSv2 as MAPSv24
from experiments.run_scenarios_ablations import scenario_tasks, scenario_models

SCENARIOS = ["balanced", "strict_sla", "bursty_multiuser", "model_degradation"]

def make_policy(version, seed):
    if version == "V2_5":
        return MAPSv25(51000 + seed)
    if version == "V2_4":
        return MAPSv24(51000 + seed)
    raise ValueError(version)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--outdir", default="results/v25_vs_v24")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    versions = ["V2_4", "V2_5"]

    for scenario in SCENARIOS:
        for seed in range(args.seeds):
            base_tasks = scenario_tasks(load_workload(seed, limit=args.tasks), scenario)
            base_models = scenario_models(scenario)
            trace = load_trace(seed)

            for version in versions:
                rec = execute(
                    copy.deepcopy(base_tasks),
                    copy.deepcopy(base_models),
                    make_policy(version, seed),
                    trace=trace,
                )
                sm = summarize(rec, 5)
                rows.append(dict(
                    scenario=scenario,
                    version=version,
                    seed=seed,
                    **sm
                ))

    df = pd.DataFrame(rows)

    by_seed = os.path.join(args.outdir, "v25_vs_v24_by_seed.csv")
    df.to_csv(by_seed, index=False)

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
    agg = df.groupby(["scenario","version"])[metrics].agg(["mean","std"])
    agg.to_csv(os.path.join(args.outdir, "v25_vs_v24_aggregate.csv"))

    print("completed runs:", len(df))
    print("results:", by_seed)
    print(agg.round(5).to_string())

if __name__ == "__main__":
    main()
