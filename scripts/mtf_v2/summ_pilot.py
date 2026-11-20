import sys,json,numpy as np
from collections import defaultdict
R=[json.loads(l) for l in open(sys.argv[1])][1:]
base={}
for r in R: base[(r['task'],r['video'])]=(r['base_margin'],r['gt']==r['pred'])
print("baseline accuracy per task (dev pilot, n=5 each):")
for t in sorted({k[0] for k in base}): v=[c for (tt,_),(m,c) in base.items() if tt==t]; print(" ",t,sum(v),"/",len(v))
D=defaultdict(list)
for r in R: D[(r['task'],r['k'],r['cond'],r['layer'])].append(r['d_margin'])
for t in sorted({k[0] for k in base}):
  print(t)
  for k in (1,5,9,13):
    for c in sorted({r['cond'] for r in R}):
      m=[np.mean(D[(t,k,c,L)]) for L in range(32)]
      print(f"  k={k:2d} {c:5s} L0-5 {np.mean(m[0:6]):+.2f} L6-15 {np.mean(m[6:16]):+.2f} L16-25 {np.mean(m[16:26]):+.2f} L26-31 {np.mean(m[26:]):+.2f}")
