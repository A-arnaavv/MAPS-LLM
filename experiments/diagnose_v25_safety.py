import os,sys,argparse,copy
import pandas as pd
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from maps_llm.defaults import models
from maps_llm.canonical import load_workload,load_trace
from maps_llm.simulator import execute
from maps_llm.optimizer import MAPSv2
from experiments.run_scenarios_ablations import scenario_tasks,scenario_models

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tasks",type=int,default=100)
    ap.add_argument("--seeds",type=int,default=3)
    ap.add_argument("--outdir",default="results/v25_safety_diag")
    a=ap.parse_args();os.makedirs(a.outdir,exist_ok=True)
    scenarios=["balanced","strict_sla","bursty_multiuser","model_degradation"]
    rows=[]
    for sc in scenarios:
        for seed in range(a.seeds):
            ledger=[]
            p=MAPSv2(51000+seed,ledger=ledger,safety_enabled=True)
            work=scenario_tasks(load_workload(seed,limit=a.tasks),sc)
            ms=scenario_models(sc)
            execute(copy.deepcopy(work),copy.deepcopy(ms),p,trace=load_trace(seed))
            d=pd.DataFrame(ledger);changed=d[d.safety_changed]
            rows.append(dict(
                scenario=sc,seed=seed,tasks=len(d),
                mean_stress=float(d.system_stress.mean()),max_stress=float(d.system_stress.max()),
                safety_interventions=int(d.safety_intervened.sum()),
                safety_changed=int(d.safety_changed.sum()),
                changed_to_lower_operational_risk=int((changed.selected_operational_risk<changed.fairness_only_operational_risk-1e-12).sum()),
                changed_equal_operational_risk=int((abs(changed.selected_operational_risk-changed.fairness_only_operational_risk)<=1e-12).sum()),
                changed_to_higher_operational_risk=int((changed.selected_operational_risk>changed.fairness_only_operational_risk+1e-12).sum()),
                safety_violations=int((~d.safety_invariant_ok).sum())))
    D=pd.DataFrame(rows)
    D.to_csv(os.path.join(a.outdir,"v25_safety_by_scenario_seed.csv"),index=False)
    print("MAPS-LLM v2.5 REGIME-AWARE SAFETY DIAGNOSTIC")
    print(D.to_string(index=False))
    print("\nTOTALS")
    cols=["tasks","safety_interventions","safety_changed","changed_to_lower_operational_risk",
          "changed_equal_operational_risk","changed_to_higher_operational_risk","safety_violations"]
    print(D[cols].sum().to_string())
    print("\nBY SCENARIO")
    print(D.groupby("scenario")[["mean_stress","max_stress","safety_interventions","safety_changed",
                                "changed_to_lower_operational_risk","changed_to_higher_operational_risk",
                                "safety_violations"]].mean().round(6).to_string())
    print("\nwrote:",a.outdir)
if __name__=="__main__":main()
