import os,sys,argparse,copy,json
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

from maps_llm.defaults import models
from maps_llm.canonical import load_workload,load_trace
from maps_llm.simulator import execute
from maps_llm.optimizer import MAPSv2
from maps_llm.entities import Plan
from maps_llm.predict import feasible,quality
from maps_llm.pareto import nondominated,crowding

def candidate_snapshot(pol,t,ms,override_debt=None):
    f=sorted(i for i,m in enumerate(ms) if feasible(t,m))
    plans=[Plan(i) for i in f]
    top=sorted(f,key=lambda i:(-round(float(quality(t,ms[i])),12),i))[:2]
    for i in top:
        others=[j for j in f if j!=i]
        if others:
            j=sorted(others,key=lambda z:(-round(float(quality(t,ms[z])),12),z))[0]
            plans.append(Plan(i,j))
    plans=sorted(set(plans),key=lambda p:(p.primary,p.verifier))
    debt=pol.user_bad[t.user_id] if override_debt is None else override_debt
    vectors=[]; meta=[]
    for p in plans:
        q,c,l,pe=pol._expected_plan(t,p,ms)
        k=(t.kind,ms[p.primary].name); n=pol.n[k]
        learned=q if n==0 else .45*q+.55*pol.mu[k]
        unc=1/np.sqrt(n+1)
        v=np.array([c/max(t.budget,.001),l/max(t.deadline,.1),1-learned,
                    max(0,l/t.deadline-1),
                    max(0,t.quality_floor-learned)+.20*debt],float)
        vectors.append(v); meta.append((p,unc,n,learned))
    nd=sorted(nondominated(vectors))
    V=np.array([vectors[i] for i in nd]); cd=crowding(V)
    ranked=[]
    for z,i in enumerate(nd):
        p,unc,n,learned=meta[i]
        div=.035 if n<3 else 0.0
        cr=0.0 if np.isinf(cd[z]) else min(float(cd[z]),3.0)*.01
        exploit=learned-.06*V[z,0]-.035*V[z,1]-.15*V[z,3]-.12*V[z,4]
        score=round(float(exploit+pol.beta*unc+div+cr),12)
        ranked.append((-score,(p.primary,p.verifier),p))
    ranked.sort()
    return {
        "vectors":vectors,
        "nd":nd,
        "selected":(ranked[0][2].primary,ranked[0][2].verifier),
        "plans":[(p.primary,p.verifier) for p in plans]
    }

class DiagnosticMAPS(MAPSv2):
    def __init__(self,seed=1):
        super().__init__(seed=seed)
        self.diag=[]
        self._last_record_id=None
    def choose(self,t,ms,records):
        # Mirror the normal learning timing exactly once.
        self._learn(records)
        with_debt=candidate_snapshot(self,t,ms)
        no_debt=candidate_snapshot(self,t,ms,override_debt=0.0)
        changed_obj5=any(abs(a[4]-b[4])>1e-15 for a,b in zip(with_debt["vectors"],no_debt["vectors"]))
        nd_changed=with_debt["nd"]!=no_debt["nd"]
        selected_changed=with_debt["selected"]!=no_debt["selected"]
        self.diag.append(dict(
            task_id=t.task_id,user_id=t.user_id,user_bad=float(self.user_bad[t.user_id]),
            debt_nonzero=abs(self.user_bad[t.user_id])>1e-15,
            objective5_changed=changed_obj5,
            pareto_set_changed=nd_changed,
            selected_plan_changed=selected_changed,
            selected_with_debt=str(with_debt["selected"]),
            selected_without_debt=str(no_debt["selected"]),
        ))
        # Avoid learning the same previous record twice: invoke the original choose
        # with an empty record list because we already called _learn above.
        return super().choose(t,ms,[])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tasks",type=int,default=500)
    ap.add_argument("--seeds",type=int,default=20)
    ap.add_argument("--outdir",default="results/v211_diagnostics")
    a=ap.parse_args(); os.makedirs(a.outdir,exist_ok=True)
    summary=[]; allrows=[]; escalation=[]
    for seed in range(a.seeds):
        tasks=load_workload(seed,limit=a.tasks); trace=load_trace(seed); ms=models()
        pol=DiagnosticMAPS(51000+seed)
        rec=execute(copy.deepcopy(tasks),copy.deepcopy(ms),pol,trace=trace)
        d=pd.DataFrame(pol.diag); d["seed"]=seed; allrows.append(d)
        # Verify planned-verifier bookkeeping and posterior accounting.
        planned=[r for r in rec if r.get("verifier","")]
        executed=[r for r in planned if r["aggregated"]]
        notexec=[r for r in planned if not r["aggregated"]]
        esc_a=sum(v-1.0 for v in pol.esc_a.values())
        esc_b=sum(v-1.0 for v in pol.esc_b.values())
        escalation.append(dict(seed=seed,planned_verifier_records=len(planned),
            executed_verifier_records=len(executed),nonexecuted_planned_verifier_records=len(notexec),
            posterior_success_updates=esc_a,posterior_failure_updates=esc_b))
        summary.append(dict(seed=seed,tasks=len(d),
            debt_nonzero=int(d.debt_nonzero.sum()),
            objective5_changed=int(d.objective5_changed.sum()),
            pareto_set_changed=int(d.pareto_set_changed.sum()),
            selected_plan_changed=int(d.selected_plan_changed.sum()),
            max_user_bad=float(d.user_bad.max()),mean_user_bad=float(d.user_bad.mean())))
    pd.concat(allrows,ignore_index=True).to_csv(os.path.join(a.outdir,"fairness_task_diagnostics.csv"),index=False)
    pd.DataFrame(summary).to_csv(os.path.join(a.outdir,"fairness_summary_by_seed.csv"),index=False)
    pd.DataFrame(escalation).to_csv(os.path.join(a.outdir,"escalation_bookkeeping.csv"),index=False)
    S=pd.DataFrame(summary); E=pd.DataFrame(escalation)
    print("FAIRNESS DIAGNOSTIC")
    print(S.to_string(index=False))
    print("\nTOTALS")
    print(S[["tasks","debt_nonzero","objective5_changed","pareto_set_changed","selected_plan_changed"]].sum().to_string())
    print("max_user_bad:",S.max_user_bad.max())
    print("\nESCALATION BOOKKEEPING")
    print(E.sum(numeric_only=True).to_string())
    print("\nwrote:",a.outdir)

if __name__=="__main__":
    main()
