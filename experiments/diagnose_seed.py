import os,sys,json,hashlib,argparse
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from maps_llm.defaults import models
from maps_llm.canonical import load_workload,load_trace,subset_sha256
from maps_llm.simulator import execute,summarize
from maps_llm.optimizer import MAPSv2

def canon(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False)

def sha(obj):
    return hashlib.sha256(canon(obj).encode()).hexdigest()

def main(seed=0,tasks=100,outdir="results/diagnostics"):
    os.makedirs(outdir,exist_ok=True)
    work=load_workload(seed,limit=tasks)
    trace=load_trace(seed)
    ledger=[]
    policy=MAPSv2(51000+seed,ledger=ledger)
    rec=execute(work,models(),policy,trace=trace)

    ledger_path=os.path.join(outdir,f"ledger_seed{seed:02d}_{tasks}.json")
    records_path=os.path.join(outdir,f"records_seed{seed:02d}_{tasks}.json")
    with open(ledger_path,"w") as f: json.dump(ledger,f,sort_keys=True,indent=2)
    with open(records_path,"w") as f: json.dump(rec,f,sort_keys=True,indent=2)

    print("seed =",seed)
    print("tasks =",tasks)
    print("WORKLOAD_SHA256 =",subset_sha256(seed,tasks,False))
    print("TRACE_SHA256    =",subset_sha256(seed,tasks,True))
    print("LEDGER_SHA256   =",sha(ledger))
    print("SCHEDULE_SHA256 =",sha(rec))
    print("summary =",json.dumps(summarize(rec,5),sort_keys=True))
    print("ledger =",ledger_path)
    print("records =",records_path)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--tasks",type=int,default=100)
    p.add_argument("--outdir",default="results/diagnostics")
    a=p.parse_args()
    main(a.seed,a.tasks,a.outdir)
