import math
import numpy as np
from .predict import quality,cost,feasible
from .fairness import outcome_fairness,fairness_components

def execute(tasks,models,policy,seed=None,trace=None):
    if trace is None:
        raise ValueError("MAPS-LLM v2.0.2 canonical execution requires a fixed trace.")
    avail={i:[0.0]*m.capacity for i,m in enumerate(models)}
    records=[]
    for task in sorted(tasks,key=lambda x:(x.arrival,x.task_id)):
        plan=policy.choose(task,models,records)
        p=models[plan.primary]
        if not feasible(task,p):
            fi=sorted(i for i,m in enumerate(models) if feasible(task,m))
            best=sorted(fi,key=lambda i:(-round(quality(task,models[i]),12),i))[0]
            plan=type(plan)(best,-1); p=models[plan.primary]

        def invoke(mi, earliest, role):
            m=models[mi]
            slots=avail[mi]
            k=min(range(len(slots)),key=lambda idx:(slots[idx],idx))
            st=max(task.arrival if earliest is None else earliest,slots[k])
            draw=trace[f"{task.task_id}|{mi}|{role}"]
            # lognormal with log(mean) and sigma=jitter/2, using shipped z draw
            svc=max(.03, math.exp(math.log(m.latency)+(m.jitter/2.0)*draw["z_service"]))
            svc*=1+.000015*(task.input_tokens+task.output_tokens)
            end=st+svc; slots[k]=end
            q=quality(task,m)
            if draw["u_reliability"]>m.reliability:
                q=max(.05,q-.35)
            else:
                q=float(np.clip(q+.025*draw["z_quality"],0,1))
            return st,end,q,cost(task,m)

        st,e1,q1,c1=invoke(plan.primary,task.arrival,"primary")
        q=q1; c=c1; end=e1; aggregated=False
        if plan.verifier>=0 and feasible(task,models[plan.verifier]):
            if q1 < task.quality_floor+.055:
                _,e2,q2,c2=invoke(plan.verifier,task.arrival,"verifier")
                end=max(e1,e2)+.12
                q=min(.995,q1+(1-q1)*(.28+.32*q2))
                c+=c2; aggregated=True

        lat=end-task.arrival
        records.append(dict(
            task_id=task.task_id,user_id=task.user_id,kind=task.kind,
            model=p.name,verifier=models[plan.verifier].name if plan.verifier>=0 else "",
            aggregated=aggregated,arrival=task.arrival,completion=end,latency=lat,
            quality=q,primary_quality=q1,cost=c,sla_violation=lat>task.deadline,
            quality_floor_met=q>=task.quality_floor,budget_met=c<=task.budget
        ))
    return records

def summarize(records,n_users):
    lat=np.array([r["latency"] for r in records])
    q=np.array([r["quality"] for r in records])
    c=np.array([r["cost"] for r in records])
    sla=np.array([r["sla_violation"] for r in records],float)
    return dict(
        quality=float(q.mean()),cost_per_task=float(c.mean()),total_cost=float(c.sum()),
        p95_latency=float(np.percentile(lat,95)),p99_latency=float(np.percentile(lat,99)),
        sla_violation_rate=float(sla.mean()),fairness=outcome_fairness(records,n_users),
        aggregation_rate=float(np.mean([r["aggregated"] for r in records])),
        throughput=float(len(records)/(max(r["completion"] for r in records)-min(r["arrival"] for r in records)+1e-9)),
        **fairness_components(records,n_users)
    )
