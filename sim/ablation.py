# Ablation (Table 6.2): each row adds one component to a single-pointer copy drafter; prints passes on the first 30 edits and on all 199.
import os as _os; _HERE = _os.path.dirname(_os.path.abspath(__file__)); _TRACE = _os.path.join(_HERE, "..", "traces", "oracle.json")
# Local (laptop, pure Python) simulation on REAL FastApply-1.5B outputs: passes + predicted T4 time from the measured cost curve.
import json, bisect, sys, time
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else _TRACE))
R = d["records"]; curve = d["curve"]
LS = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047]
def pass_cost(L, C=2048):
    """seconds for one verification pass with a draft of L tokens (linear interpolation on the measured curve)"""
    xs = [x + 1 for x in LS]; ys = [curve[f"{C}:{x}"] for x in LS]; n = L + 1
    if n <= xs[0]: return ys[0]
    for i in range(1, len(xs)):
        if n <= xs[i]: return ys[i - 1] + (ys[i] - ys[i - 1]) * (n - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[-1] * n / xs[-1]

ONEGRAM = True
class Index:
    NS = (32, 16, 12, 8, 6, 4, 3, 2, 1)
    def __init__(self, src):
        self.m = {n: {} for n in self.NS}
        for n in self.NS:
            dd = self.m[n]
            for i in range(len(src) - n + 1): dd.setdefault(tuple(src[i:i + n]), []).append(i + n)
    def cands(self, gen, prev, K):
        out = []
        for n in self.NS:
            if len(gen) < n or (n == 1 and not ONEGRAM): continue
            pos = self.m[n].get(tuple(gen[-n:]))
            if not pos: continue
            for p in sorted(pos, key=lambda q: (q < prev, abs(q - prev))):
                if p not in out: out.append(p)
                if len(out) >= K: return out
        return out

def tree_size(branches):
    trie = set()
    for b in branches:
        for j in range(1, len(b) + 1): trie.add(tuple(b[:j]))
    return len(trie)

AUX = json.load(open(_os.path.join(_HERE, "vocab_aux.json"))); NL = set(AUX["nl"]); UPD = AUX["upd"]
FRONT = False
def region_split(src):
    for i in range(len(src) - len(UPD) + 1):
        if src[i:i + len(UPD)] == UPD: return i + len(UPD)
    return len(src)
def next_line(src, p):
    while p < len(src) and src[p] not in NL: p += 1
    return p + 1
def sim(src, gt, K=1, resync=False, kmax=2048, k0=64, selfback=True, side=32):
    idx = Index(src); gen = [gt[0]]; pos = 1; passes = 1; t = pass_cost(len(src) - 1); k = k0   # first pass = prompt prefill
    ptrs = idx.cands(gen, 0, K); prev = ptrs[0] if ptrs else 0
    split = region_split(src); fo, fs = 0, split   # read-frontiers in original / snippet
    while pos < len(gt):
        branches = [src[p:p + (k if j == 0 else min(k, side))][:len(gt) - pos] for j, p in enumerate(ptrs)]
        if not branches and selfback:   # like HF prompt lookup inside own output
            for n in (4, 3, 2):
                if len(gen) <= n: continue
                pat = gen[-n:]; hit = None
                for i in range(len(gen) - n - 1, -1, -1):
                    if gen[i:i + n] == pat: hit = i; break
                if hit is not None: branches = [gen[hit + n:hit + n + 10][:len(gt) - pos]]; break
        best_a, best_i = 0, 0
        for i, b in enumerate(branches):
            a = 0
            while a < len(b) and b[a] == gt[pos + a]: a += 1
            if a > best_a: best_a, best_i = a, i
        t += pass_cost(tree_size(branches)); passes += 1
        a = best_a; new = gt[pos:pos + a + 1]; gen += new; pos += len(new)
        chosen = ptrs[best_i] if ptrs and best_i < len(ptrs) else None
        if chosen is not None:
            if chosen < split: fo = max(fo, chosen + a)
            else: fs = max(fs, chosen + a)
        if chosen is not None and branches and a == len(branches[best_i]) and a > 0 and chosen + a < len(src) and src[chosen + a] == gt[pos - 1]:
            prev = chosen + a + 1; ptrs = [prev] + [p for p in idx.cands(gen, prev, K) if p != prev][:K - 1]; k = min(kmax, k * 2)
        else:
            prev = (chosen if chosen is not None else prev) + a; k = max(16, min(kmax, 2 * a + 16))
            rs = []
            if resync and chosen is not None:
                tkn = gen[-1]
                if chosen + a + 1 < len(src) and src[chosen + a + 1] == tkn: rs.append(chosen + a + 2)
                for q in (chosen + a + 1, chosen + a, chosen + a + 2):
                    if 0 <= q < len(src) and q not in rs: rs.append(q)
            ptrs = rs + [p for p in idx.cands(gen, prev, K) if p not in rs]
            if FRONT:
                fr = [fo, next_line(src, fo), fs, next_line(src, fs)]
                ptrs = [p for p in ptrs if p not in fr][:max(K, len(rs) + K)]
                ptrs = ptrs[:len(rs)] + [p for p in fr if 0 <= p < len(src)] + ptrs[len(rs):]
            else:
                ptrs = ptrs[:max(K, len(rs) + K)]
    return passes, t

def run(name, subset=None, **kw):
    P = T = 0; t0 = time.time()
    for r in (subset or R):
        p, t = sim(r["prompt"], r["out"], **kw); P += p; T += t
    return P, T, time.time() - t0

S30 = R[:30]
ROWS = [(False, False, "single pointer", dict(K=1)),
        (False, False, "+ resync", dict(K=1, resync=True)),
        (False, False, "+ K=4 side branches (32 tok)", dict(K=4, resync=True, side=32)),
        (True, False, "+ 1-token anchors, K=8", dict(K=8, resync=True, side=16)),
        (True, True, "+ read frontiers", dict(K=8, resync=True, side=16))]
for one, fr, name, kw in ROWS:
    ONEGRAM = one; FRONT = fr
    P, _, _ = run(name, **kw); P3, _, _ = run(name, subset=S30, **kw)
    print(f"{name:30s} first 30 {P3:4d}   all 199 {P:5d}", flush=True)
print("(next rows: sim_drafter.py = strength-weighted lengths 664 / 3,978; branch_stats.py = learned sizing + hybrid 644)")
