import os as _os; _HERE = _os.path.dirname(_os.path.abspath(__file__)); _TRACE = _os.path.join(_HERE, "..", "traces", "oracle.json")
# Hardware projection (FERMI, not measured except T4-1.5B): does 1.3x fewer passes turn into wall-clock?
# SuffixDecoding proxy = its real 868 passes x cost of a ~32-token linear pass (real T4 GPU: 59.9 s incl. 22 ms loop/pass fits this).
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import sized as Z
S30 = Z.S30; Z.Sized.MULT, Z.Sized.FLOOR, Z.Sized.MAINCAP = 1.0, 8, 6.0
real = Z.pc
HW = {  # name: (fixed s, flat tokens, slope s/token, loop s, tree extra s)
 "T4 1.5B measured, HF loop": None,
 "T4 1.5B lean loop + fused tree": "lean",
 "7B on 2xT4 (Fermi)": (0.075, 64, 0.0006, 0.022, 0.016),
 "H100 1.5B, vLLM-like (Fermi)": (0.006, 1024, 0.00001, 0.0, 0.001),
 "H100 7B, vLLM-like (Fermi)": (0.012, 512, 0.00003, 0.0, 0.001),
}
for name, hw in HW.items():
    if hw is None: Z.OVER, Z.TREE = 0.022, 0.016; Z.pc = real
    elif hw == "lean": Z.OVER, Z.TREE = 0.0, 0.0; Z.pc = lambda L, C=2048: real(L, C) - 0.022 * 0 if True else 0
    else:
        F, flat, sl, lp, tr = hw; Z.TREE = tr
        Z.pc = lambda L, C=2048, F=F, flat=flat, sl=sl, lp=lp: F + max(0, L + 1 - flat) * sl + lp
    if hw == "lean": Z.OVER = 0.0
    def pc_(L): return Z.pc(L)
    P = T = 0
    for r in S30: p, t = Z.run(r, Z.Sized, True, K=8, strong=64, weak=16, kmin=64, kmax=512, k0=64); P += p; T += t
    pre = sum(pc_(len(r["prompt"]) - 1) for r in S30)
    sfx = pre + 868 * pc_(32)
    print(f"{name:34s} ours {P} passes {T:6.1f}s | SuffixDecoding proxy {sfx:6.1f}s -> {sfx/T:.2f}x")
