import os,sys,argparse,copy
import pandas as pd
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

from maps_llm.defaults import models
from maps_llm.canonical import load_workload,load_trace
from maps_llm.simulator import execute
from maps_llm.optimizer import MAPSv2

def clone_base_policy(src, fairness_enabled):
    p=MAPSv2(
        seed=1,
        beta=src.beta,
        fairness_gamma=src.fairness_gamma,
        fairness_enabled=fairness_enabled
    )
    # Copy learned online state exactly.
    p.n=copy.deepcopy(src.n)
    p.mu=copy.deepcopy(src.mu)
    p.user_bad=copy.deepcopy(src.user_bad)
    p.esc_a=copy.deepcopy(src.esc_a)
    p.esc_b=copy.deepcopy(src.esc_b)
    p.plan_n=copy.deepcopy(src.plan_n)
    p.plan_mu=copy.deepcopy(src.plan_mu)
    return p

class DiagnosticMAPS(MAPSv2):
    def __init__(self,seed):
        super().__init__(seed=seed,fairness_enabled=True)
        self.rows=[]

    def choose(self,t,ms,records):
        # Learn once, exactly matching normal online timing.
        self._learn(records)

        # Counterfactual policies share the exact same learned state and differ
        # only in whether fairness admissibility is enabled.
        with_fair=clone_base_policy(self,True)
        no_fair=clone_base_policy(self,False)

        pa=with_fair.choose(t,ms,[])
        pb=no_fair.choose(t,ms,[])

        f,plans=self._candidate_plans(t,ms)
        cand=self._build_candidates(t,ms,plans)
        keep=self._admissible_indices(cand,t)

        risk_a=float(next(r["service_risk"] for r in cand if r["plan"]==pa))
        risk_b=float(next(r["service_risk"] for r in cand if r["plan"]==pb))

        self.rows.append(dict(
            task_id=int(t.task_id),
            user_id=int(t.user_id),
            debt=float(self.user_bad[t.user_id]),
            all_plans=len(plans),
            admissible_plans=len(keep),
            plans_pruned=len(plans)-len(keep),
            selected_with_fairness=str((pa.primary,pa.verifier)),
            selected_without_fairness=str((pb.primary,pb.verifier)),
            selected_plan_changed=(pa!=pb),
            selected_fairness_risk=risk_a,
            selected_nofair_risk=risk_b
        ))

        # Actual execution follows the fairness-enabled decision.
        return pa

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tasks",type=int,default=100)
    ap.add_argument("--seeds",type=int,default=3)
    ap.add_argument("--outdir",default="results/v23_fairness_diag")
    a=ap.parse_args()
    os.makedirs(a.outdir,exist_ok=True)

    allrows=[]
    summary=[]
    for seed in range(a.seeds):
        p=DiagnosticMAPS(51000+seed)
        execute(
            copy.deepcopy(load_workload(seed,limit=a.tasks)),
            copy.deepcopy(models()),
            p,
            trace=load_trace(seed)
        )
        d=pd.DataFrame(p.rows)
        d["seed"]=seed
        allrows.append(d)

        changed=d[d.selected_plan_changed]
        summary.append(dict(
            seed=seed,
            tasks=len(d),
            debt_nonzero=int((d.debt>1e-15).sum()),
            tasks_with_pruning=int((d.plans_pruned>0).sum()),
            total_plans_pruned=int(d.plans_pruned.sum()),
            selected_plan_changed=int(d.selected_plan_changed.sum()),
            changed_to_lower_risk=int((changed.selected_fairness_risk <
                                       changed.selected_nofair_risk-1e-12).sum()),
            changed_equal_risk=int((abs(changed.selected_fairness_risk-
                                        changed.selected_nofair_risk)<=1e-12).sum()),
            changed_to_higher_risk=int((changed.selected_fairness_risk >
                                        changed.selected_nofair_risk+1e-12).sum()),
            mean_risk_reduction_when_changed=float(
                (changed.selected_nofair_risk-changed.selected_fairness_risk).mean()
            ) if len(changed) else 0.0
        ))

    A=pd.concat(allrows,ignore_index=True)
    S=pd.DataFrame(summary)
    A.to_csv(os.path.join(a.outdir,"fairness_admissibility_by_task.csv"),index=False)
    S.to_csv(os.path.join(a.outdir,"fairness_admissibility_by_seed.csv"),index=False)

    print("MAPS-LLM v2.3 STRUCTURAL FAIRNESS DIAGNOSTIC")
    print(S.to_string(index=False))
    print("\nTOTALS")
    print(S[[
        "tasks","debt_nonzero","tasks_with_pruning","total_plans_pruned",
        "selected_plan_changed","changed_to_lower_risk",
        "changed_equal_risk","changed_to_higher_risk"
    ]].sum().to_string())

    changed=A[A.selected_plan_changed]
    if len(changed):
        print("mean risk reduction on changed decisions:",
              float((changed.selected_nofair_risk-changed.selected_fairness_risk).mean()))
    print("\nwrote:",a.outdir)

if __name__=="__main__":
    main()
