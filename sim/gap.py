import os as _os; _HERE = _os.path.dirname(_os.path.abspath(__file__)); _TRACE = _os.path.join(_HERE, "..", "traces", "oracle.json")
# Gap analysis: where do our hybrid passes lose to the copy oracle, and how much is new text? (laptop, seconds)
import json, sys
from drafter import Drafter
import hybrid as H
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else _TRACE)); R = d["records"]; aux = json.load(open(_os.path.join(_HERE, "vocab_aux.json"))); NL, UPD = set(aux["nl"]), aux["upd"]
cfg = dict(K=8, strong=32, weak=8, kmin=64, kmax=384, k0=64)
def idx4(src):
    m = {}
    for j in range(len(src) - 3): m.setdefault(tuple(src[j:j + 4]), []).append(j)
    return m
def oracle_at(m, src, gt, pos):
    best, bstart = 0, None
    for s in m.get(tuple(gt[pos:pos + 4]), []):
        a = 0
        while pos + a < len(gt) and s + a < len(src) and src[s + a] == gt[pos + a]: a += 1
        if a > best: best, bstart = a, s
    if best == 0:
        for n in (3, 2, 1):
            for s in range(len(src) - n + 1):
                if src[s:s + n] == gt[pos:pos + n]:
                    a = 0
                    while pos + a < len(gt) and s + a < len(src) and src[s + a] == gt[pos + a]: a += 1
                    if a > best: best, bstart = a, s
            if best: break
    return best, bstart
B = dict(ok=0, short_branch=0, missing_cand=0, novel=0); loss = dict(short_branch=0, missing_cand=0)
novel_tok = 0; tot_tok = 0; P = 0; Pnov = 0
for r in R:
    if "suffix_fwd" not in r: continue
    src, gt = r["prompt"], r["out"]; m = idx4(src)
    dr = Drafter(src, NL, UPD, **cfg); gen = [gt[0]]; pos = 1; dr.start(gen); confident = False
    while pos < len(gt):
        br = dr.propose(gen, len(gt) - pos)
        if confident: br = br[:1]
        ptrs = list(dr.ptrs)
        best_a, best_i = 0, 0
        for i, b in enumerate(br):
            a = 0
            while a < len(b) and b[a] == gt[pos + a]: a += 1
            if a > best_a: best_a, best_i = a, i
        o, os_ = oracle_at(m, src, gt, pos)
        P += 1
        if o <= 3 and best_a <= 3: B["novel"] += 1; novel_tok += best_a + 1; Pnov += 1
        elif o <= best_a + 8: B["ok"] += 1
        else:
            # was the oracle start among candidate positions offered (any branch starting at a position that copies >= o-? )
            hit = any(p is not None and src[p:p + best_a + 9] == gt[pos:pos + best_a + 9] for p in ptrs[:len(br)])
            k = "short_branch" if hit else "missing_cand"; B[k] += 1; loss[k] += o - best_a
        confident = bool(br) and best_a == len(br[best_i]) and best_a > 0
        new = gt[pos:pos + best_a + 1]; gen += new; pos += len(new); tot_tok += len(new)
        dr.update(gen, best_i, best_a, br)
print("first30 hybrid passes", P, "buckets", B, "copy tokens lost", loss)
print(f"novel passes {Pnov} emit {novel_tok} tokens of {tot_tok} ({novel_tok/tot_tok:.1%})")
# helper draft model ceiling on novel passes: tokens/pass for acceptance alpha, gamma drafted
for alpha in (0.5, 0.6, 0.7, 0.8):
    for g in (3, 5):
        tpp = (1 - alpha ** (g + 1)) / (1 - alpha)
        newP = P - Pnov + novel_tok / tpp
        print(f"helper alpha={alpha} gamma={g}: {tpp:.2f} tok/pass -> passes {newP:.0f} ({P/newP:.2f}x fewer than ours; vs Snowflake 868: {868/newP:.2f}x)")
