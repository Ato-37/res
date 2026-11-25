import json,sys,collections,numpy as np
from scipy import stats
def load(f):
    D=collections.defaultdict(lambda:collections.defaultdict(list))
    for l in open(f):
        d=json.loads(l)
        if "meta" in d or "error" in d: continue
        if 6<=d["layer"]<=15: D[(d["task"],d["video"])][d["cond"]].append(d["d_margin"])
    return D
for f in sys.argv[1:]:
    D=load(f); print(f)
    for t in ["Action Antonym","Scene Transition"]:
        vs=[v for (tt,v) in D if tt==t]; row=[]
        for c in ["real","adj","far","first"]:
            if c=="real": x=np.array([np.mean(D[(t,v)]["real"]) for v in vs if D[(t,v)]["real"]]); print(t,len(vs),"real %.3f"%x.mean()); continue
            x=np.array([np.mean(D[(t,v)][c])-np.mean(D[(t,v)][c+"_m"]) for v in vs]); n=len(x); m=x.mean(); h=stats.t.ppf(.975,n-1)*x.std(ddof=1)/n**.5
            print(f"  {c}-{c}_m n={n} {m:+.3f} [{m-h:+.3f},{m+h:+.3f}] raw {np.mean([np.mean(D[(t,v)][c]) for v in vs]):+.3f} matched {np.mean([np.mean(D[(t,v)][c+'_m']) for v in vs]):+.3f}")
