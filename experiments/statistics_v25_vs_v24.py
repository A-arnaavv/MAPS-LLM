import argparse, os, json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = {
    "quality": True,
    "cost_per_task": False,
    "p95_latency": False,
    "sla_violation_rate": False,
    "fairness": True,
    "user_utility_gap": False,
    "worst_user_utility": True,
    "throughput": True,
}

def holm_adjust(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    out = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m-rank)*p[idx])
        running = max(running, val)
        out[idx] = running
    return out

def bootstrap_ci(d, rng, B=20000):
    d = np.asarray(d, float)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [.025,.975])
    return float(lo), float(hi)

def rank_biserial(d):
    d = np.asarray(d,float)
    nz = d[np.abs(d)>1e-15]
    if len(nz)==0:
        return 0.0
    order = np.argsort(np.abs(nz), kind="mergesort")
    ranks = np.empty(len(nz))
    vals = np.abs(nz)[order]
    i=0
    while i<len(vals):
        j=i+1
        while j<len(vals) and vals[j]==vals[i]:
            j+=1
        avg=(i+1+j)/2
        ranks[order[i:j]]=avg
        i=j
    wp=ranks[nz>0].sum()
    wm=ranks[nz<0].sum()
    return float((wp-wm)/(wp+wm)) if wp+wm else 0.0

def paired_dz(d):
    d=np.asarray(d,float)
    sd=d.std(ddof=1)
    if abs(sd)<1e-15:
        return 0.0
    return float(d.mean()/sd)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/v25_vs_v24/v25_vs_v24_by_seed.csv")
    ap.add_argument("--outdir", default="results/v25_vs_v24_stats")
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--rng-seed", type=int, default=20260911)
    a=ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    df=pd.read_csv(a.csv)
    rng=np.random.default_rng(a.rng_seed)
    rows=[]

    for scenario in sorted(df.scenario.unique()):
        g=df[df.scenario==scenario]
        wide_seed = sorted(g.seed.unique())
        temp=[]
        pvals=[]
        for metric, hib in METRICS.items():
            wide=g.pivot(index="seed", columns="version", values=metric).loc[wide_seed]
            v24=wide["V2_4"].to_numpy()
            v25=wide["V2_5"].to_numpy()

            advantage = v25-v24 if hib else v24-v25
            rawdiff = v25-v24
            lo,hi=bootstrap_ci(advantage,rng,a.bootstrap)

            if np.all(np.abs(advantage)<=1e-15):
                stat,p=0.0,1.0
            else:
                stat,p=wilcoxon(advantage,zero_method="pratt",alternative="two-sided",method="auto")
                stat=float(stat);p=float(p)

            row=dict(
                scenario=scenario,
                metric=metric,
                higher_is_better=hib,
                n_pairs=len(advantage),
                v24_mean=float(v24.mean()),
                v25_mean=float(v25.mean()),
                raw_mean_difference_v25_minus_v24=float(rawdiff.mean()),
                mean_advantage_positive_is_v25_better=float(advantage.mean()),
                advantage_ci95_low=lo,
                advantage_ci95_high=hi,
                paired_cohen_dz_advantage=paired_dz(advantage),
                rank_biserial_advantage=rank_biserial(advantage),
                wilcoxon_stat=stat,
                wilcoxon_p=p,
            )
            temp.append(row)
            pvals.append(p)

        adj=holm_adjust(pvals)
        for row,padj in zip(temp,adj):
            row["holm_p_within_scenario"]=float(padj)
            row["significant_0_05"]=bool(padj<.05)
            rows.append(row)

    R=pd.DataFrame(rows)
    R.to_csv(os.path.join(a.outdir,"v25_vs_v24_stats.csv"),index=False)

    focus = R[R.metric.isin([
        "quality","p95_latency","sla_violation_rate",
        "worst_user_utility","cost_per_task"
    ])].copy()

    focus.to_csv(os.path.join(a.outdir,"v25_vs_v24_focus.csv"),index=False)

    print("Paired v2.5 vs v2.4 statistical analysis complete")
    print("scenarios:", df.scenario.nunique())
    print("seeds:", df.seed.nunique())
    print("\nFocused results:")
    print(focus[[
        "scenario","metric","v24_mean","v25_mean",
        "mean_advantage_positive_is_v25_better",
        "advantage_ci95_low","advantage_ci95_high",
        "holm_p_within_scenario","significant_0_05"
    ]].round(6).to_string(index=False))
    print("\nwrote:", a.outdir)

if __name__=="__main__":
    main()
