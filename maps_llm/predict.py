import numpy as np

def quality(task,m):
    return float(np.clip(m.quality[task.kind] - .22*task.difficulty + .08*(1-task.difficulty),.05,.995))

def cost(task,m):
    return task.input_tokens/1000*m.in_cost + task.output_tokens/1000*m.out_cost

def feasible(task,m):
    return task.input_tokens+task.output_tokens <= m.context

def plan_prediction(task,plan,models):
    a=models[plan.primary]
    qa=quality(task,a); ca=cost(task,a); la=a.latency
    if plan.verifier < 0:
        return qa,ca,la
    b=models[plan.verifier]
    qb=quality(task,b); cb=cost(task,b)
    # Conservative synthetic aggregation: complementary evidence has diminishing gain.
    gain=(1-qa)*(.28+.32*qb)
    q=min(.995,qa+gain)
    return q, ca+cb, max(la,b.latency)+.12
