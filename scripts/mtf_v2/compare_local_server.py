import sys,json,gzip,numpy as np
from collections import defaultdict
def load(paths):
    R={};B={}
    for p in paths:
        f=gzip.open(p,'rt') if p.endswith('.gz') else open(p)
        for l in f:
            r=json.loads(l)
            if 'meta' in r: continue
            B[(r['task'],r['video'])]=(r['base_margin'],r['pred']==r['gt'])
            R[(r['task'],r['video'],r['k'],r['layer'],r['cond'])]=r['d_margin']
    return R,B
L,LB=load(sys.argv[1:3]); S,SB=load(sys.argv[3:5])
vids=sorted(set(LB)&set(SB)); print("common videos",len(vids), "per task",{t:sum(1 for v in vids if v[0]==t) for t in sorted({v[0] for v in vids})})
bl=np.array([LB[v][0] for v in vids]); bs=np.array([SB[v][0] for v in vids])
print("base_margin |diff| median %.4f max %.4f, pred agree %d/%d"%(np.median(abs(bl-bs)),abs(bl-bs).max(),sum(LB[v][1]==SB[v][1] for v in vids),len(vids)))
for cond in ('real','null'):
    keys=[k for k in L if k[4]==cond and k in S]
    a=np.array([L[k] for k in keys]); b=np.array([S[k] for k in keys])
    print(cond,"cells",len(keys),"r=%.3f |diff| median %.4f p95 %.4f"%(np.corrcoef(a,b)[0,1],np.median(abs(a-b)),np.percentile(abs(a-b),95)))
    for t in sorted({k[0] for k in keys}):
        kk=[k for k in keys if k[0]==t and 6<=k[3]<=15 and k[2]==13]
        if kk: print("  ",t,"k13 L6-15 mean local %+.3f server %+.3f (n_vid %d)"%(np.mean([L[k] for k in kk]),np.mean([S[k] for k in kk]),len({k[1] for k in kk})))
