import os,sys,csv,argparse
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from maps_llm.defaults import users,models
from maps_llm.canonical import load_workload,load_trace
from maps_llm.simulator import execute,summarize
from maps_llm.baselines import Strongest,Cheapest,SLAGreedy,Weighted,ContextualUCB
from maps_llm.nsga2_control import NSGA2Control
from maps_llm.optimizer import MAPSv2

def run(tasks=500,seeds=20,outdir="results"):
    if seeds>20: raise ValueError("Canonical package contains seeds 0..19 only.")
    os.makedirs(outdir,exist_ok=True); rows=[]
    us=users(); ms=models()
    for s in range(seeds):
        work=load_workload(s,limit=tasks); trace=load_trace(s)
        methods={
            "STRONGEST":Strongest(),"CHEAPEST":Cheapest(),"SLA_GREEDY":SLAGreedy(),
            "WEIGHTED_SUM":Weighted(),"CONTEXTUAL_UCB":ContextualUCB(),
            "NSGA2_CONTROL":NSGA2Control(),"MAPS_LLM_V2":MAPSv2(seed=51000+s)
        }
        for name,p in methods.items():
            rec=execute(work,ms,p,trace=trace)
            x=summarize(rec,len(us)); x.update(method=name,seed=s); rows.append(x)
    path=os.path.join(outdir,"v2_summary_by_seed.csv")
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print("completed runs:",len(rows));print("results:",path)

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--tasks",type=int,default=500)
    a.add_argument("--seeds",type=int,default=20)
    a.add_argument("--outdir",default="results")
    z=a.parse_args();run(z.tasks,z.seeds,z.outdir)
