import numpy as np
from .entities import Task
from .defaults import TASKS

def generate(n, users, seed, arrival_rate=2.4):
    rng=np.random.default_rng(seed)
    kinds=rng.choice(TASKS,n,p=[.28,.22,.20,.15,.15])
    arrivals=np.cumsum(rng.exponential(1/arrival_rate,n))
    out=[]
    for i,(kind,a) in enumerate(zip(kinds,arrivals)):
        # Unequal user mix is intentional.
        uid=int(rng.choice(len(users),p=[.25,.20,.18,.20,.17]))
        u=users[uid]
        inp=int(rng.integers(6000,26000)) if kind=="long_context" else int(rng.integers(250,5200))
        ot=int(rng.integers(80,1300))
        diff=float(rng.beta(2.3,2.0))
        base={"general":2.5,"reasoning":5.2,"coding":4.8,"long_context":7.0,"translation":3.2}[kind]
        out.append(Task(i,uid,kind,float(a),inp,ot,float(base*u.deadline_scale),
                        min(.97,u.quality_floor+.07*diff),u.budget,diff,u.priority))
    return out
