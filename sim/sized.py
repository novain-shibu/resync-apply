import os as _os; _HERE = _os.path.dirname(_os.path.abspath(__file__)); _TRACE = _os.path.join(_HERE, "..", "traces", "oracle.json")
import json, sys
from collections import defaultdict
sys.argv = [sys.argv[0], _TRACE]
import ranker_lib as RL
from ranker_lib import Learned, R, NL, UPD
curve = RL.d["curve"]
LS = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047]; OVER, TREE = 0.022, 0.016
def pc(L, C=2048):
    xs = [x + 1 for x in LS]; ys = [curve[f"{C}:{x}"] for x in LS]; n = L + 1
    if n <= xs[0]: return ys[0] + OVER
    for i in range(1, len(xs)):
        if n <= xs[i]: return ys[i - 1] + (ys[i] - ys[i - 1]) * (n - xs[i - 1]) / (xs[i] - xs[i - 1]) + OVER
    return ys[-1] * n / xs[-1] + OVER
# train table
log = []
for r in R[100:]: RL.run(r, Learned, log=log, K=8, strong=64, weak=16)
tab = defaultdict(lambda: [0.0, 0])
for f, a in log: tab[f][0] += a; tab[f][1] += 1
Learned.TABLE = {k: tuple(v) for k, v in tab.items()}
class Sized(Learned):
    MULT = 2.0; FLOOR = 8; MAINCAP = None
    def propose(self, gen, remaining):
        br = super().propose(gen, remaining)
        if self._self_draft or not br: return br
        out = []
        for j, (p, b) in enumerate(zip(self.ptrs, br)):
            e = self.score(p)
            if j == 0:
                L = len(b) if self.MAINCAP is None else min(len(b), max(self.FLOOR, int(self.MAINCAP * e)))
            else:
                L = min(len(b), max(self.FLOOR, int(self.MULT * e)))
            out.append(b[:L])
        return out
def run(rec, cls, hybrid, **kw):
    src, gt = rec["prompt"], rec["out"]
    dr = cls(src, NL, UPD, **kw); gen = [gt[0]]; pos = 1; passes = 1; t = pc(len(src) - 1); dr.start(gen); conf = False
    while pos < len(gt):
        br = dr.propose(gen, len(gt) - pos)
        if hybrid and conf: br = br[:1]
        best_a, best_i = 0, 0
        for i, b in enumerate(br):
            a = 0
            while a < len(b) and b[a] == gt[pos + a]: a += 1
            if a > best_a: best_a, best_i = a, i
        q = len(set(tuple(b[:j]) for b in br for j in range(1, len(b) + 1)))
        t += pc(q) + (TREE if len(br) > 1 else 0); passes += 1
        conf = bool(br) and best_a == len(br[best_i]) and best_a > 0
        new = gt[pos:pos + best_a + 1]; gen += new; pos += len(new); dr.update(gen, best_i, best_a, br)
    return passes, t
S30 = [r for r in R[:100] if "suffix_fwd" in r]
for tree_extra in (0.016, 0.003):
    TREE = tree_extra; best = None
    for mult in (1.0, 1.5, 2.0, 3.0):
        for floor in (4, 8, 16):
            for maincap in (None, 3.0, 6.0):
                for kmax in (384, 512, 1024):
                    Sized.MULT, Sized.FLOOR, Sized.MAINCAP = mult, floor, maincap
                    P = T = 0
                    for r in S30: p, t = run(r, Sized, True, K=8, strong=64, weak=16, kmin=64, kmax=kmax, k0=64); P += p; T += t
                    if best is None or T < best[0]: best = (T, P, mult, floor, maincap, kmax)
    T, P, mult, floor, maincap, kmax = best
    print(f"tree extra {tree_extra*1000:.0f} ms: best sized mult={mult} floor={floor} maincap={maincap} kmax={kmax}: passes {P} predicted {T:5.1f}s -> {62.1/T:.2f}x vs Snowflake", flush=True)
