import os,sys,json,hashlib
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from maps_llm.defaults import users,models
from maps_llm.canonical import load_workload,load_trace,subset_sha256
from maps_llm.simulator import execute,summarize
from maps_llm.optimizer import MAPSv2

def canon(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False)
def h(obj):
    return hashlib.sha256(canon(obj).encode()).hexdigest()

for s in range(3):
    work=load_workload(s,limit=100)
    trace=load_trace(s)
    rec=execute(work,models(),MAPSv2(51000+s),trace=trace)
    print(f"seed={s}")
    print("WORKLOAD_SHA256 =",subset_sha256(s,100,trace=False))
    print("TRACE_SHA256    =",subset_sha256(s,100,trace=True))
    print("SCHEDULE_SHA256 =",h(rec))
    print(json.dumps(summarize(rec,5),sort_keys=True))
