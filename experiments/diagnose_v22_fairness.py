import os,sys,argparse,copy
import numpy as np
import pandas as pd
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from maps_llm.defaults import models
from maps_llm.canonical import load_workload,load_trace
from maps_llm.simulator import execute
from maps_llm.optimizer import MAPSv2
from maps_llm.entities import Plan
from maps_llm.predict import feasible,quality
from maps_llm.pareto import nondominated,crowding

def snapshot(pol,t,ms,zero_debt=False):
    f=sorted(i for i,m in enumerate(ms) if feasible(t,m)); plans=[Plan(i) for i in f]
    top=sorted(f,key=lambda i:(-round(float(quality(t,ms[i])),12),i))[:2]
    for i in top:
        others=[j for j in f if j!=i]
        if others:
            j=sorted(others,key=lambda z:(-round(float(quality(t,ms[z])),12),z))[0]; plans.append(Plan(i,j))
    plans=sorted(set(plans),key=lambda p:(p.primary,p.verifier)); debt=0.0 if zero_debt else float(pol.user_bad[t.user_id])
    vectors=[]; meta=[]
    for p in plans:
        q,c,l,pe=pol._expected_plan(t,p,ms); k=(t.kind,ms[p.primary].name); n=pol.n[k]
        learned=q if n==0 else .45*q+.55*pol.mu[k]; unc=1/np.sqrt(n+1)
        qs=max(0,t.quality_floor-learned); sr=max(0,l/max(t.deadline,.1)-1.0)
        ff=(1+.35*debt)*(.70*qs+.30*sr)
        v=np.array([c/max(t.budget,.001),l/max(t.deadline,.1),1-learned,sr,ff],float)
        vectors.append(v); meta.append((p,unc,n,learned))
    nd=sorted(nondominated(vectors)); V=np.array([vectors[i] for i in nd]); cd=crowding(V); ranked=[]
    for z,i in enumerate(nd):
        p,unc,n,learned=meta[i]; div=.035 if n<3 else 0.0; cr=0.0 if np.isinf(cd[z]) else min(float(cd[z]),3.0)*.01
        exploit=learned-.06*V[z,0]-.035*V[z,1]-.15*V[z,3]-.12*V[z,4]
        score=round(float(exploit+pol.beta*unc+div+cr),12); ranked.append((-score,(p.primary,p.verifier),p))
    ranked.sort(); return vectors,nd,(ranked[0][2].primary,ranked[0][2].verifier)

class D(MAPSv2):
    def __init__(self,seed=1): super().__init__(seed=seed); self.rows=[]
    def choose(self,t,ms,records):
        self._learn(records); vw,ndw,sw=snapshot(self,t,ms,False); vz,ndz,sz=snapshot(self,t,ms,True)
        fw=[float(v[4]) for v in vw]; fz=[float(v[4]) for v in vz]; rel=False
        for i in range(len(fw)):
            for j in range(i+1,len(fw)):
                if abs((fw[i]-fw[j])-(fz[i]-fz[j]))>1e-12: rel=True; break
            if rel: break
        self.rows.append(dict(task_id=t.task_id,user_id=t.user_id,debt=float(self.user_bad[t.user_id]),debt_nonzero=abs(self.user_bad[t.user_id])>1e-15,relative_fairness_changed=rel,pareto_set_changed=ndw!=ndz,selected_plan_changed=sw!=sz,selected_with_debt=str(sw),selected_without_debt=str(sz)))
        return super().choose(t,ms,[])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tasks',type=int,default=100); ap.add_argument('--seeds',type=int,default=3); ap.add_argument('--outdir',default='results/v22_fairness_diag'); a=ap.parse_args(); os.makedirs(a.outdir,exist_ok=True)
    rows=[]; sums=[]
    for seed in range(a.seeds):
        p=D(51000+seed); execute(copy.deepcopy(load_workload(seed,limit=a.tasks)),copy.deepcopy(models()),p,trace=load_trace(seed)); d=pd.DataFrame(p.rows); d['seed']=seed; rows.append(d)
        sums.append(dict(seed=seed,tasks=len(d),debt_nonzero=int(d.debt_nonzero.sum()),relative_fairness_changed=int(d.relative_fairness_changed.sum()),pareto_set_changed=int(d.pareto_set_changed.sum()),selected_plan_changed=int(d.selected_plan_changed.sum()),max_debt=float(d.debt.max()),mean_debt=float(d.debt.mean())))
    A=pd.concat(rows,ignore_index=True); S=pd.DataFrame(sums); A.to_csv(os.path.join(a.outdir,'fairness_counterfactual_by_task.csv'),index=False); S.to_csv(os.path.join(a.outdir,'fairness_counterfactual_by_seed.csv'),index=False)
    print('MAPS-LLM v2.2 FAIRNESS COUNTERFACTUAL'); print(S.to_string(index=False)); print('\nTOTALS'); print(S[['tasks','debt_nonzero','relative_fairness_changed','pareto_set_changed','selected_plan_changed']].sum().to_string()); print('\nwrote:',a.outdir)
if __name__=='__main__': main()
