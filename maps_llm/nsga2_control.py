"""
Lightweight per-task NSGA-II-style control.

For the discrete plan set, it performs non-dominated selection and crowding
without MAPS learning, fairness debt, or adaptive aggregation preference.
This is a mechanism control, not a claim of a full general-purpose NSGA-II implementation.
"""
import numpy as np
from .entities import Plan
from .predict import feasible,plan_prediction,quality
from .pareto import nondominated,crowding

class NSGA2Control:
    def choose(self,t,ms,r):
        f=[i for i,m in enumerate(ms) if feasible(t,m)]
        plans=[Plan(i) for i in f]
        for i in f:
            j=max([x for x in f if x!=i],key=lambda x:quality(t,ms[x]),default=-1)
            if j>=0: plans.append(Plan(i,j))
        V=[]
        for p in plans:
            q,c,l=plan_prediction(t,p,ms)
            V.append([c/max(t.budget,.001),l/max(t.deadline,.1),1-q,
                      max(0,l/t.deadline-1),max(0,t.quality_floor-q)])
        nd=nondominated(V); A=np.array([V[i] for i in nd]); cd=crowding(A)
        # Knee-like choice among non-dominated solutions.
        norm=(A-A.min(0))/(np.ptp(A,axis=0)+1e-9)
        score=norm.mean(1)-.015*np.where(np.isinf(cd),3,cd)
        return plans[nd[int(np.argmin(score))]]
