import numpy as np

_EPS = 1e-12

def dominates(a,b):
    a=np.asarray(a,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    return bool(np.all(a<=b) and np.any(a<b))

def nondominated(vectors):
    out=[]
    for i,a in enumerate(vectors):
        if not any(j!=i and dominates(b,a) for j,b in enumerate(vectors)):
            out.append(i)
    return out

def crowding(V):
    """
    Deterministic NSGA-style crowding distance with explicit tie handling.

    Important reproducibility rule:
    boundary status belongs to an objective VALUE, not to whichever member of
    a tied group happens to appear first/last in a platform-dependent sort.

    For each objective:
      * all candidates tied at the minimum receive +inf
      * all candidates tied at the maximum receive +inf
      * interior candidates use nearest strictly lower/higher DISTINCT values

    This removes dependence on unstable ordering among equal values.
    """
    V=np.asarray(V,dtype=np.float64)
    n,m=V.shape
    if n==0:
        return np.array([],dtype=np.float64)
    if n<=2:
        return np.full(n,np.inf,dtype=np.float64)

    d=np.zeros(n,dtype=np.float64)

    for j in range(m):
        col=V[:,j]
        lo=float(np.min(col))
        hi=float(np.max(col))
        span=hi-lo

        # Constant objective contributes no crowding information.
        if abs(span)<=_EPS:
            continue

        # Canonical distinct values, quantized only for tie grouping.
        qcol=np.round(col,12)
        vals=sorted(set(float(x) for x in qcol))
        if len(vals)<=1:
            continue

        qlo,qhi=vals[0],vals[-1]

        # Every member of an extreme-value tie is a boundary point.
        for i,x in enumerate(qcol):
            xv=float(x)
            if xv==qlo or xv==qhi:
                d[i]=np.inf

        # Interior points: use adjacent distinct objective values.
        pos={v:k for k,v in enumerate(vals)}
        for i,x in enumerate(qcol):
            if np.isinf(d[i]):
                continue
            xv=float(x)
            k=pos[xv]
            if k==0 or k==len(vals)-1:
                d[i]=np.inf
                continue
            lower=vals[k-1]
            upper=vals[k+1]
            d[i]+=(upper-lower)/span

    return d
