import numpy as np
from collections import defaultdict
from .entities import Plan
from .predict import quality, feasible
from .pareto import nondominated, crowding

def _stable_plan_key(p):
    return (int(p.primary), int(p.verifier))

def _q(x):
    return float(np.round(float(x), 12))

class MAPSv2:
    """
    MAPS-LLM v2.5: v2.4 fairness-safe Pareto routing plus endogenous
    regime-aware operational safety. The scheduler is not given scenario labels.
    """
    def __init__(self,seed=1,beta=.10,ledger=None,
                 fairness_gamma=.70,fairness_enabled=True,fairness_safe=True,
                 safety_enabled=True,stress_decay=.90,safety_kappa=1.30):
        self.rng=np.random.default_rng(seed)
        self.beta=beta
        self.n=defaultdict(int)
        self.mu=defaultdict(lambda:.72)
        self.user_bad=defaultdict(float)
        self.esc_a=defaultdict(lambda:1.0)
        self.esc_b=defaultdict(lambda:1.0)
        self.plan_n=defaultdict(int)
        self.plan_mu=defaultdict(lambda:.78)
        self.ledger=ledger
        self.fairness_gamma=float(fairness_gamma)
        self.fairness_enabled=bool(fairness_enabled)
        self.fairness_safe=bool(fairness_safe)
        self.safety_enabled=bool(safety_enabled)
        self.stress_decay=float(stress_decay)
        self.safety_kappa=float(safety_kappa)
        self.system_stress=0.0

    def _learn(self,records):
        if not records:return
        r=records[-1]
        k=(r["kind"],r["model"])
        n=self.n[k]+1
        obs=float(r.get("primary_quality",r["quality"]))
        self.mu[k]+=(obs-self.mu[k])/n
        self.n[k]=n

        if r.get("verifier",""):
            pk=(r["kind"],r["model"],r["verifier"])
            if r["aggregated"]: self.esc_a[pk]+=1.0
            else: self.esc_b[pk]+=1.0
            pn=self.plan_n[pk]+1
            self.plan_mu[pk]+=(float(r["quality"])-self.plan_mu[pk])/pn
            self.plan_n[pk]=pn

        bad=.5*float(r["sla_violation"])+.3*float(not r["quality_floor_met"])+.2*float(not r["budget_met"])
        self.user_bad[r["user_id"]]=.92*self.user_bad[r["user_id"]]+bad

        stress_obs=.55*float(r["sla_violation"])+.25*float(not r["quality_floor_met"])+.20*float(not r["budget_met"])
        self.system_stress=self.stress_decay*self.system_stress+(1-self.stress_decay)*stress_obs

    def _expected_plan(self,t,p,ms):
        from .predict import plan_prediction
        if p.verifier<0:
            q,c,l=plan_prediction(t,p,ms)
            return float(q),float(c),float(l),0.0
        q1,c1,l1=plan_prediction(t,Plan(p.primary,-1),ms)
        q2,c2,l2=plan_prediction(t,p,ms)
        pk=(t.kind,ms[p.primary].name,ms[p.verifier].name)
        pe=self.esc_a[pk]/(self.esc_a[pk]+self.esc_b[pk])
        ec=c1+pe*(c2-c1)
        el=l1+pe*(l2-l1)
        pq=q1+pe*(q2-q1)
        pn=self.plan_n[pk]
        eq=pq if pn==0 else .45*pq+.55*self.plan_mu[pk]
        return float(eq),float(ec),float(el),float(pe)

    def _candidate_plans(self,t,ms):
        f=sorted(i for i,m in enumerate(ms) if feasible(t,m))
        plans=[Plan(i) for i in f]
        top=sorted(f,key=lambda i:(-_q(quality(t,ms[i])),i))[:2]
        for i in top:
            others=[j for j in f if j!=i]
            if others:
                j=sorted(others,key=lambda z:(-_q(quality(t,ms[z])),z))[0]
                plans.append(Plan(i,j))
        return f,sorted(set(plans),key=_stable_plan_key)

    def _build_candidates(self,t,ms,plans):
        rows=[]
        for p in plans:
            q,c,l,pe=self._expected_plan(t,p,ms)
            k=(t.kind,ms[p.primary].name)
            n=self.n[k]
            learned=q if n==0 else .45*q+.55*self.mu[k]
            unc=1/np.sqrt(n+1)
            q_short=max(0.0,t.quality_floor-learned)
            latency_ratio=l/max(t.deadline,.1)
            sla_risk=max(0.0,latency_ratio-1.0)
            service_risk=.70*q_short+.30*sla_risk
            operational_risk=.55*latency_ratio+.25*service_risk+.20*pe
            rows.append(dict(plan=p,q=float(q),c=float(c),l=float(l),pe=float(pe),
                             learned=float(learned),unc=float(unc),n=int(n),
                             q_short=float(q_short),latency_ratio=float(latency_ratio),
                             sla_risk=float(sla_risk),service_risk=float(service_risk),
                             operational_risk=float(operational_risk)))
        return rows

    def _fairness_admissible_indices(self,rows,t):
        if not rows:return []
        if not self.fairness_enabled:return list(range(len(rows)))
        debt=max(0.0,float(self.user_bad[t.user_id]))
        risks=np.array([r["service_risk"] for r in rows],float)
        rmin=float(risks.min()); rmax=float(risks.max())
        if debt<=1e-15 or abs(rmax-rmin)<=1e-15:
            return list(range(len(rows)))
        threshold=rmin+(rmax-rmin)*float(np.exp(-self.fairness_gamma*debt))
        keep=[i for i,r in enumerate(risks) if r<=threshold+1e-12]
        return keep if keep else [int(np.argmin(risks))]

    def _admissible_indices(self,rows,t,enabled=None):
        """Backward-compatible v2.3/v2.4 fairness-admissibility API."""
        original=self.fairness_enabled
        if enabled is not None:
            self.fairness_enabled=bool(enabled)
        try:
            return self._fairness_admissible_indices(rows,t)
        finally:
            self.fairness_enabled=original

    def _stress_admissible_indices(self,rows,current):
        if not current:return []
        if not self.safety_enabled or self.system_stress<=1e-15:
            return list(current)
        ops=np.array([rows[i]["operational_risk"] for i in current],float)
        omin=float(ops.min()); omax=float(ops.max())
        if abs(omax-omin)<=1e-15:return list(current)
        threshold=omin+(omax-omin)*float(np.exp(-self.safety_kappa*self.system_stress))
        keep=[i for i in current if rows[i]["operational_risk"]<=threshold+1e-12]
        return keep if keep else [current[int(np.argmin(ops))]]

    def _select_from_indices(self,t,rows,indices):
        vectors=[];meta=[]
        for oi in indices:
            r=rows[oi]
            v=np.array([r["c"]/max(t.budget,.001),r["l"]/max(t.deadline,.1),
                        1-r["learned"],r["sla_risk"],r["service_risk"]],dtype=np.float64)
            vectors.append(v);meta.append((oi,r))
        nd=sorted(nondominated(vectors))
        V=np.array([vectors[i] for i in nd],dtype=np.float64)
        cd=crowding(V)
        ranked=[];details=[]
        for z,li in enumerate(nd):
            oi,r=meta[li]
            div=.035 if r["n"]<3 else 0.0
            cr=0.0 if np.isinf(cd[z]) else min(float(cd[z]),3.0)*.01
            exploit=r["learned"]-.06*V[z,0]-.035*V[z,1]-.15*V[z,3]-.12*V[z,4]
            score=_q(exploit+self.beta*r["unc"]+div+cr)
            p=r["plan"]
            ranked.append((-score,_stable_plan_key(p),oi,p))
            details.append(dict(original_index=int(oi),plan=[p.primary,p.verifier],
                                service_risk=float(r["service_risk"]),
                                operational_risk=float(r["operational_risk"]),
                                score=float(score)))
        ranked.sort()
        _,_,idx,plan=ranked[0]
        return plan,int(idx),nd,details

    def choose(self,t,ms,records):
        self._learn(records)
        f,plans=self._candidate_plans(t,ms)
        rows=self._build_candidates(t,ms,plans)
        all_idx=list(range(len(rows)))

        p0,p0_idx,_,_=self._select_from_indices(t,rows,all_idx)
        p0_sr=float(rows[p0_idx]["service_risk"])

        fair=self._fairness_admissible_indices(rows,t)
        debt=max(0.0,float(self.user_bad[t.user_id]))
        fairness_intervened=self.fairness_enabled and debt>1e-15 and len(fair)<len(all_idx)

        if fairness_intervened and self.fairness_safe:
            fair_safe=[i for i in fair if rows[i]["service_risk"]<=p0_sr+1e-12]
            if not fair_safe:fair_safe=[p0_idx]
        else:
            fair_safe=fair

        p_fair,p_fair_idx,_,_=self._select_from_indices(t,rows,fair_safe)
        p_fair_op=float(rows[p_fair_idx]["operational_risk"])
        p_fair_sr=float(rows[p_fair_idx]["service_risk"])

        stress_set=self._stress_admissible_indices(rows,fair_safe)
        safety_intervened=self.safety_enabled and self.system_stress>1e-15 and len(stress_set)<len(fair_safe)

        if safety_intervened:
            stress_safe=[i for i in stress_set if rows[i]["operational_risk"]<=p_fair_op+1e-12]
            if not stress_safe:stress_safe=[p_fair_idx]
        else:
            stress_safe=stress_set

        selected,selected_idx,nd,details=self._select_from_indices(t,rows,stress_safe)
        selected_op=float(rows[selected_idx]["operational_risk"])

        if self.ledger is not None:
            self.ledger.append(dict(
                task_id=int(t.task_id),user_id=int(t.user_id),kind=t.kind,
                user_bad=float(self.user_bad[t.user_id]),
                system_stress=float(self.system_stress),
                fairness_only_plan=[p_fair.primary,p_fair.verifier],
                fairness_only_operational_risk=float(p_fair_op),
                fairness_changed=(p_fair!=p0),
                counterfactual_no_fair_risk=float(p0_sr),
                selected_risk=float(p_fair_sr),
                fairness_safety_ok=bool(
                    (not fairness_intervened) or p_fair_sr<=p0_sr+1e-12
                ),
                selected_plan=[selected.primary,selected.verifier],
                selected_operational_risk=float(selected_op),
                fairness_intervened=bool(fairness_intervened),
                safety_intervened=bool(safety_intervened),
                safety_changed=(selected!=p_fair),
                safety_invariant_ok=bool((not safety_intervened) or selected_op<=p_fair_op+1e-12),
                fair_candidate_count=len(fair_safe),
                stress_candidate_count=len(stress_safe),
                candidates=details))
        return selected
