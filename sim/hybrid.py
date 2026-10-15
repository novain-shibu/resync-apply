import os as _os; _HERE = _os.path.dirname(_os.path.abspath(__file__)); _TRACE = _os.path.join(_HERE, "..", "traces", "oracle.json")
# Hybrid: linear (fast causal path) pass when the last pass was fully accepted, tree pass only after a miss.
import json, sys
from drafter import Drafter
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else _TRACE)); R = d["records"]; curve = d["curve"]; aux = json.load(open(_os.path.join(_HERE, "vocab_aux.json"))); NL, UPD = set(aux["nl"]), aux["upd"]
LS = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047]; OVER = 0.022; TREE_EXTRA = 0.016
def pc(L, C=2048):
    xs = [x + 1 for x in LS]; ys = [curve[f"{C}:{x}"] for x in LS]; n = L + 1
    if n <= xs[0]: return ys[0] + OVER
    for i in range(1, len(xs)):
        if n <= xs[i]: return ys[i - 1] + (ys[i] - ys[i - 1]) * (n - xs[i - 1]) / (xs[i] - xs[i - 1]) + OVER
    return ys[-1] * n / xs[-1] + OVER
def run(rec, mode, **kw):
    src, gt = rec["prompt"], rec["out"]
    dr = Drafter(src, NL, UPD, **kw); gen = [gt[0]]; pos = 1; passes = 1; t = pc(len(src) - 1); dr.start(gen); confident = False; trees = 0
    while pos < len(gt):
        br = dr.propose(gen, len(gt) - pos)
        if mode == "linear" or (mode == "hybrid" and confident): br = br[:1]
        best_a, best_i = 0, 0
        for i, b in enumerate(br):
            a = 0
            while a < len(b) and b[a] == gt[pos + a]: a += 1
            if a > best_a: best_a, best_i = a, i
        q = len(set(tuple(b[:j]) for b in br for j in range(1, len(b) + 1)))
        t += pc(q) + (TREE_EXTRA if len(br) > 1 else 0.0); trees += len(br) > 1; passes += 1
        confident = bool(br) and best_a == len(br[best_i]) and best_a > 0
        new = gt[pos:pos + best_a + 1]; gen += new; pos += len(new)
        dr.update(gen, best_i, best_a, br)
    return passes, t, trees
S30 = [r for r in R if "suffix_fwd" in r]
cfg = dict(K=8, strong=32, weak=8, kmin=64, kmax=384, k0=64)
for mode in ("tree", "hybrid", "linear"):
    P = T = TR = 0
    for r in S30: p, t, tr = run(r, mode, **cfg); P += p; T += t; TR += tr
    print(f"{mode:7s} first30 passes {P:4d} (tree passes {TR:4d}) predicted {T:5.1f}s  vs Snowflake real 62.1s -> {62.1/T:.2f}x")
for kmax in (256, 384, 512):
    for K in (4, 8):
        P = T = 0
        for r in S30: p, t, _ = run(r, "hybrid", K=K, strong=32, weak=8, kmin=64, kmax=kmax, k0=64); P += p; T += t
        print(f"hybrid kmax={kmax} K={K}: passes {P} predicted {T:5.1f}s -> {62.1/T:.2f}x")
