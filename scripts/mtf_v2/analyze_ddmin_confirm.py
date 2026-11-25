import json,sys
from scipy import stats
import numpy as np
rows=[json.loads(l) for l in open(sys.argv[1])][2:]
rows=[r for r in rows if "d" in r]
n=len(rows); mn=np.array([-r["d"]["min"] for r in rows]); al=np.array([-r["d"]["all"] for r in rows])
rd=np.array([np.mean([-v for k,v in r["d"].items() if k.startswith("rand")]) for r in rows])
x=mn-rd; m=x.mean(); h=stats.t.ppf(.975,n-1)*x.std(ddof=1)/n**.5 if n>1 else float("nan")
print(f"n={n} min {mn.mean():+.3f} rand {rd.mean():+.3f} all {al.mean():+.3f}; min-rand {m:+.3f} [{m-h:+.3f},{m+h:+.3f}]; min/all {mn.mean()/al.mean():.2f}")
