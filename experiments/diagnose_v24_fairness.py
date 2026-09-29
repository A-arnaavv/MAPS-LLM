import os, sys, argparse, copy
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maps_llm.defaults import models
from maps_llm.canonical import load_workload, load_trace
from maps_llm.simulator import execute
from maps_llm.optimizer import MAPSv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--outdir", default="results/v24_fairness_diag")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)

    allrows = []
    summary = []

    for seed in range(a.seeds):
        ledger = []
        p = MAPSv2(
            seed=51000 + seed,
            fairness_enabled=True,
            fairness_safe=True,
            ledger=ledger,
        )

        execute(
            copy.deepcopy(load_workload(seed, limit=a.tasks)),
            copy.deepcopy(models()),
            p,
            trace=load_trace(seed),
        )

        d = pd.DataFrame(ledger)
        d["seed"] = seed

        changed = d[d["fairness_changed"]].copy()

        changed_to_lower = int(
            (changed["selected_risk"]
             < changed["counterfactual_no_fair_risk"] - 1e-12).sum()
        )

        changed_equal = int(
            (abs(changed["selected_risk"]
                 - changed["counterfactual_no_fair_risk"]) <= 1e-12).sum()
        )

        changed_to_higher = int(
            (changed["selected_risk"]
             > changed["counterfactual_no_fair_risk"] + 1e-12).sum()
        )

        summary.append(
            dict(
                seed=seed,
                tasks=len(d),
                debt_nonzero=int((d["user_bad"] > 1e-15).sum()),
                fairness_changed=int(d["fairness_changed"].sum()),
                changed_to_lower_risk=changed_to_lower,
                changed_equal_risk=changed_equal,
                changed_to_higher_risk=changed_to_higher,
                safety_violations=int((~d["fairness_safety_ok"]).sum()),
                mean_risk_reduction_when_changed=float(
                    (
                        changed["counterfactual_no_fair_risk"]
                        - changed["selected_risk"]
                    ).mean()
                ) if len(changed) else 0.0,
            )
        )

        allrows.append(d)

    A = pd.concat(allrows, ignore_index=True)
    S = pd.DataFrame(summary)

    A.to_csv(
        os.path.join(a.outdir, "v24_fairness_decisions_by_task.csv"),
        index=False,
    )

    S.to_csv(
        os.path.join(a.outdir, "v24_fairness_summary_by_seed.csv"),
        index=False,
    )

    print("MAPS-LLM v2.4 FAIRNESS-SAFETY DIAGNOSTIC")
    print(S.to_string(index=False))

    print("\nTOTALS")
    cols = [
        "tasks",
        "debt_nonzero",
        "fairness_changed",
        "changed_to_lower_risk",
        "changed_equal_risk",
        "changed_to_higher_risk",
        "safety_violations",
    ]
    print(S[cols].sum().to_string())

    changed = A[A["fairness_changed"]]
    if len(changed):
        print(
            "mean risk reduction on changed decisions:",
            float(
                (
                    changed["counterfactual_no_fair_risk"]
                    - changed["selected_risk"]
                ).mean()
            ),
        )

    print("\nwrote:", a.outdir)

if __name__ == "__main__":
    main()
