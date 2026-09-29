import json, hashlib
from pathlib import Path
from .entities import Task

ROOT=Path(__file__).resolve().parent.parent/"canonical_data"

def _raw(path):
    return path.read_bytes()

def sha256_file(path):
    return hashlib.sha256(_raw(path)).hexdigest()

def load_workload(seed_index, limit=None):
    p=ROOT/"workloads"/f"workload_{int(seed_index):02d}.json"
    raw=json.loads(p.read_text(encoding="utf-8"))
    if limit is not None:
        raw=raw[:int(limit)]
    return [Task(**x) for x in raw]

def load_trace(seed_index):
    p=ROOT/"traces"/f"trace_{int(seed_index):02d}.json"
    return json.loads(p.read_text(encoding="utf-8"))

def subset_sha256(seed_index, limit=None, trace=False):
    if not trace:
        p=ROOT/"workloads"/f"workload_{int(seed_index):02d}.json"
        obj=json.loads(p.read_text(encoding="utf-8"))
        if limit is not None: obj=obj[:int(limit)]
    else:
        p=ROOT/"traces"/f"trace_{int(seed_index):02d}.json"
        obj=json.loads(p.read_text(encoding="utf-8"))
        if limit is not None:
            prefixes={str(i) for i in range(int(limit))}
            obj={k:v for k,v in obj.items() if k.split("|",1)[0] in prefixes}
    s=json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False)
    return hashlib.sha256(s.encode()).hexdigest()
