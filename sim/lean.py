import os as _os; _HERE = _os.path.dirname(_os.path.abspath(__file__)); _TRACE = _os.path.join(_HERE, "..", "traces", "oracle.json")
# Lean-loop projection: what if per-pass Python overhead (22 ms) and tree-mask extra (16 ms) go away (CUDA graphs / fused tree kernel)?
import sys, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import sized as Z
S30 = Z.S30
Z.Sized.MULT, Z.Sized.FLOOR, Z.Sized.MAINCAP = 1.0, 8, 6.0
for over, tree in ((0.022, 0.016), (0.022, 0.0), (0.0, 0.0)):
    Z.OVER, Z.TREE = over, tree
    for kmax in (384, 512):
        P = T = 0
        for r in S30: p, t = Z.run(r, Z.Sized, True, K=8, strong=64, weak=16, kmin=64, kmax=kmax, k0=64); P += p; T += t
        sfx = 59.9 - 869 * (0.022 - over)
        print(f"loop {over*1000:.0f} ms, tree extra {tree*1000:.0f} ms, kmax {kmax}: ours {P} passes {T:5.1f}s ; SuffixDecoding est {sfx:5.1f}s -> {sfx/T:.2f}x")
