import pandas as pd,sys,os
p=sys.argv[1] if len(sys.argv)>1 else "results/v2_summary_by_seed.csv"
d=pd.read_csv(p)
metrics=["quality","cost_per_task","p95_latency","sla_violation_rate","fairness","aggregation_rate","throughput"]
a=d.groupby("method")[metrics].agg(["mean","std"])
print(a.round(5).to_string())
out=os.path.join(os.path.dirname(p),"v2_aggregate.csv");a.to_csv(out);print("\nwrote:",out)
