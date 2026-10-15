# Per-edit pass counts for our configurations, compared with SuffixDecoding per-edit counts (from kaggle/heldout_suffix2.py).
# The learned-sizing table is trained on records 100..198, so its clean held-out split is 30..99.
import sys, io, json, random, contextlib, os
SFX_ARG = sys.argv[1] if len(sys.argv) > 1 else None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import sized as Z
from drafter import Drafter
Z.Sized.MULT, Z.Sized.FLOOR, Z.Sized.MAINCAP = 1.0, 8, 6.0
R = Z.R
def rule(r, **kw):
    src, gt = r["prompt"], r["out"]; dr = Drafter(src, Z.NL, Z.UPD, **kw); gen = [gt[0]]; pos = 1; p = 1; dr.start(gen)
    while pos < len(gt):
        br = dr.propose(gen, len(gt) - pos); ba, bi = 0, 0
        for i, b in enumerate(br):
            a = 0
            while a < len(b) and b[a] == gt[pos + a]: a += 1
            if a > ba: ba, bi = a, i
        new = gt[pos:pos + ba + 1]; gen += new; pos += len(new); p += 1; dr.update(gen, bi, ba, br)
    return p
ours = {"untuned": [rule(r, K=8, strong=64, weak=16) for r in R],
        "time-tuned": [rule(r, K=8, strong=32, weak=8, kmin=64, kmax=384) for r in R],
        "learned+hybrid (GPU config)": [Z.run(r, Z.Sized, True, K=8, strong=64, weak=16, kmin=64, kmax=1024, k0=64)[0] for r in R]}
sfx_path = SFX_ARG or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "traces", "suffix_per_record.json")
S = json.load(open(sfx_path))
def ci(a, b, idx):
    random.seed(0); bs = []
    for _ in range(2000):
        s = [random.choice(idx) for _ in idx]; bs.append(sum(b[i] for i in s) / sum(a[i] for i in s))
    bs.sort(); return bs[50], bs[1950]
for name, O in ours.items():
    for lab, idx in (("first 30", range(0, 30)), ("30..99", range(30, 100)), ("30..198", range(30, len(R))), ("all 199", range(len(R)))):
        idx = list(idx); o = sum(O[i] for i in idx); sl = sum(S["linear_passes"][i] for i in idx); st = sum(S["tree_passes"][i] for i in idx)
        lo, hi = ci(O, S["linear_passes"], idx)
        print(f"{name:28s} {lab:9s} ours {o:5d}  suffix linear {sl:5d} ({sl/o:.2f}x, CI {lo:.2f}-{hi:.2f})  suffix tree {st:5d} ({st/o:.2f}x)")
