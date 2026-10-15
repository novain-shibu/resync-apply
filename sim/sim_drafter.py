import os as _os; _HERE = _os.path.dirname(_os.path.abspath(__file__)); _TRACE = _os.path.join(_HERE, "..", "traces", "oracle.json")
import json, sys, time
from drafter import Drafter
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else _TRACE)); R = d["records"]; aux = json.load(open(_os.path.join(_HERE, "vocab_aux.json")))
NL, UPD = set(aux["nl"]), aux["upd"]
def run(rec, **kw):
    src, gt = rec["prompt"], rec["out"]
    dr = Drafter(src, NL, UPD, **kw); gen = [gt[0]]; pos = 1; passes = 1; verified = 0
    dr.start(gen)
    while pos < len(gt):
        br = dr.propose(gen, len(gt) - pos)
        best_a, best_i = 0, 0
        for i, b in enumerate(br):
            a = 0
            while a < len(b) and b[a] == gt[pos + a]: a += 1
            if a > best_a: best_a, best_i = a, i
        trie = set(tuple(b[:j]) for b in br for j in range(1, len(b) + 1)); verified += len(trie)
        new = gt[pos:pos + best_a + 1]; gen += new; pos += len(new); passes += 1
        dr.update(gen, best_i, best_a, br)
    return passes, verified
t0 = time.time()
P = sum(run(r, K=8, strong=64, weak=16)[0] for r in R)
P3 = sum(run(r, K=8, strong=64, weak=16)[0] for r in R if "suffix_fwd" in r)
print(f"drafter (K=8, strong=64, weak=16): all 199 edits {P} passes, first 30 edits {P3} passes (SuffixDecoding real GPU: 868)  [{time.time()-t0:.1f}s]")
