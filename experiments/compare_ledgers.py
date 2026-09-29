import json,sys,math

def close(a,b,abs_tol=1e-12,rel_tol=1e-12):
    if a is None or b is None:
        return a is b
    if isinstance(a,(int,str,bool)) or isinstance(b,(int,str,bool)):
        return a==b
    if isinstance(a,float) and isinstance(b,float):
        return math.isclose(a,b,abs_tol=abs_tol,rel_tol=rel_tol)
    return a==b

def main(a_path,b_path):
    A=json.load(open(a_path))
    B=json.load(open(b_path))
    n=min(len(A),len(B))
    for i in range(n):
        a,b=A[i],B[i]
        if a["selected_plan"]!=b["selected_plan"]:
            print("FIRST_SELECTED_PLAN_DIVERGENCE")
            print("ledger_index =",i)
            print("task_id =",a["task_id"],b["task_id"])
            print("A selected =",a["selected_plan"])
            print("B selected =",b["selected_plan"])
            print("\nA candidates:")
            for c in a["candidates"]: print(c)
            print("\nB candidates:")
            for c in b["candidates"]: print(c)
            return
        # even if selection same, identify first numeric/state divergence
        if a!=b:
            print("FIRST_LEDGER_STATE_DIVERGENCE_WITH_SAME_SELECTION")
            print("ledger_index =",i,"task_id =",a["task_id"])
            keys=sorted(set(a)|set(b))
            for k in keys:
                if a.get(k)!=b.get(k):
                    print("field =",k)
                    print("A =",a.get(k))
                    print("B =",b.get(k))
            return
    if len(A)!=len(B):
        print("LENGTH_DIVERGENCE",len(A),len(B))
    else:
        print("NO_DIVERGENCE")

if __name__=="__main__":
    if len(sys.argv)!=3:
        raise SystemExit("usage: python3 experiments/compare_ledgers.py A.json B.json")
    main(sys.argv[1],sys.argv[2])
