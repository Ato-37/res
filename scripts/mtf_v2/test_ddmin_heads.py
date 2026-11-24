import sys, os, random
sys.path.insert(0, os.path.dirname(__file__)); from ddmin_heads import ddmin
N = list(range(320))
def run(name, eff, truth=None):
    s, full, calls = ddmin(N, eff, 0.8); print(f"{name:28s} found {len(s)} heads, calls {calls}, full effect {full:.2f}", "| contains truth:" , set(truth) <= set(s) if truth else "-"); return s
# additive diffuse: every head 0.01 -> need >=80% of heads, nothing to localize
s = run("additive diffuse", lambda S: 0.01 * len(S)); assert len(s) >= 256
# AND-type: effect only if all 6 required heads are blocked
req = random.Random(1).sample(N, 6); s = run("AND of 6 heads", lambda S: 1.0 if set(req) <= set(S) else 0.0, req); assert set(s) == set(req)
# OR-type (redundant): effect if any of 20 heads blocked -> single head suffices
grp = random.Random(2).sample(N, 20); s = run("OR of 20 heads", lambda S: 1.0 if set(grp) & set(S) else 0.0); assert len(s) == 1 and s[0] in grp
# XOR-like pair synergy (+ small main effects): pair needed
a, b = 7, 200; s = run("synergy pair + weak mains", lambda S: (1.0 if a in S and b in S else 0.0) + 0.001 * len(S), [a, b]); assert {a, b} <= set(s) and len(s) <= 60  # weak mains (0.001 each) supply part of the 80% threshold: 1.0 + 0.001*|S| >= 0.8*1.32 -> |S|>=56
# noisy-free 3-way
t = random.Random(3).sample(N, 3); s = run("3-way AND", lambda S: 1.0 if set(t) <= set(S) else 0.0, t); assert set(s) == set(t)
print("ddmin tests OK")
