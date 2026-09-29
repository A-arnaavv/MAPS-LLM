import numpy as np

def _jain(x):
    x=np.asarray(x,dtype=float)
    if len(x)==0 or np.allclose(x,0): return 1.0
    return float((x.sum()**2)/(len(x)*np.square(x).sum()+1e-12))

def outcome_fairness(records,n_users):
    """
    User-level normalized outcome utility:
      35% mean delivered quality
      30% SLA satisfaction
      25% quality-floor satisfaction
      10% budget satisfaction

    Report Jain utility fairness. Also expose disparity metrics separately via
    fairness_components() for scientific analysis; do not interpret Jain alone.
    """
    vals=[]
    for u in range(n_users):
        rr=[r for r in records if r["user_id"]==u]
        if not rr: continue
        q=np.mean([r["quality"] for r in rr])
        sla=np.mean([not r["sla_violation"] for r in rr])
        qf=np.mean([r["quality_floor_met"] for r in rr])
        bud=np.mean([r["budget_met"] for r in rr])
        vals.append(.35*q+.30*sla+.25*qf+.10*bud)
    return _jain(vals)

def fairness_components(records,n_users):
    vals=[]; sla=[]; qf=[]
    for u in range(n_users):
        rr=[r for r in records if r["user_id"]==u]
        if not rr: continue
        uq=.35*np.mean([r["quality"] for r in rr])+.30*np.mean([not r["sla_violation"] for r in rr])+\
           .25*np.mean([r["quality_floor_met"] for r in rr])+.10*np.mean([r["budget_met"] for r in rr])
        vals.append(uq)
        sla.append(np.mean([not r["sla_violation"] for r in rr]))
        qf.append(np.mean([r["quality_floor_met"] for r in rr]))
    return dict(
        user_utility_gap=float(max(vals)-min(vals)) if vals else 0.0,
        worst_user_utility=float(min(vals)) if vals else 1.0,
        sla_satisfaction_gap=float(max(sla)-min(sla)) if sla else 0.0,
        quality_floor_gap=float(max(qf)-min(qf)) if qf else 0.0
    )
