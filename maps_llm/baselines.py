import numpy as np
from .entities import Plan
from .predict import quality,cost,feasible

class Strongest:
    def choose(self,t,ms,r):
        f=[i for i,m in enumerate(ms) if feasible(t,m)]
        return Plan(max(f,key=lambda i:quality(t,ms[i])))

class Cheapest:
    def choose(self,t,ms,r):
        f=[i for i,m in enumerate(ms) if feasible(t,m)]
        return Plan(min(f,key=lambda i:cost(t,ms[i])))

class SLAGreedy:
    def choose(self,t,ms,r):
        f=[i for i,m in enumerate(ms) if feasible(t,m)]
        good=[i for i in f if ms[i].latency<=t.deadline and quality(t,ms[i])>=t.quality_floor]
        if good:return Plan(min(good,key=lambda i:cost(t,ms[i])))
        return Plan(max(f,key=lambda i:quality(t,ms[i])-.03*ms[i].latency))

class Weighted:
    """Fixed scalarized quality-cost-latency baseline."""
    def choose(self,t,ms,records):
        f=sorted(i for i,m in enumerate(ms) if feasible(t,m))
        scored=[]
        for i in f:
            q=quality(t,ms[i]); c=cost(t,ms[i]); l=ms[i].latency
            # normalized fixed scalarization; intentionally static, no learning.
            score=.58*q-.24*(c/max(t.budget,.001))-.18*(l/max(t.deadline,.1))
            scored.append((-round(float(score),12),i))
        scored.sort()
        return Plan(scored[0][1],-1)

class ContextualUCB:
    def __init__(self,beta=.12):
        self.beta=beta; self.n={}; self.mean={}
    def choose(self,t,ms,r):
        if r:
            z=r[-1]; key=(z["kind"],z["model"]); n=self.n.get(key,0)+1
            old=self.mean.get(key,.70); self.mean[key]=old+(z["quality"]-old)/n; self.n[key]=n
        f=[i for i,m in enumerate(ms) if feasible(t,m)]
        scores=[]
        for i in f:
            k=(t.kind,ms[i].name); n=self.n.get(k,0)
            mu=self.mean.get(k,quality(t,ms[i]))
            scores.append(mu+self.beta/np.sqrt(n+1)-7*cost(t,ms[i])-.02*ms[i].latency)
        return Plan(f[int(np.argmax(scores))])
