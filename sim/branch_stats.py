# Branch and token statistics for the two main configurations; the second is the exact config of the 644-pass / 56.8 s GPU run.
import sys, io, contextlib, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import sized as Z
Z.Sized.MULT, Z.Sized.FLOOR, Z.Sized.MAINCAP = 1.0, 8, 6.0
from drafter import Drafter
def stats(cls, hybrid, **kw):
    P = V = O = 0; nb = []; T = 0
    for r in Z.S30:
        src, gt = r["prompt"], r["out"]; dr = cls(src, Z.NL, Z.UPD, **kw); gen = [gt[0]]; pos = 1; P += 1; dr.start(gen); conf = False
        while pos < len(gt):
            br = dr.propose(gen, len(gt) - pos)
            if hybrid and conf: br = br[:1]
            nb.append(len(br)); ba, bi = 0, 0
            for i, b in enumerate(br):
                a = 0
                while a < len(b) and b[a] == gt[pos + a]: a += 1
                if a > ba: ba, bi = a, i
            V += len(set(tuple(b[:j]) for b in br for j in range(1, len(b) + 1)))
            conf = bool(br) and ba == len(br[bi]) and ba > 0
            new = gt[pos:pos + ba + 1]; gen += new; pos += len(new); P += 1; dr.update(gen, bi, ba, br)
        O += len(gt)
    print(f"passes {P}  branches/pass mean {sum(nb)/len(nb):.2f} max {max(nb)}  verified tokens {V} = {V/O:.2f} per output token ({V/P:.0f} per pass)")
stats(Drafter, False, K=8, strong=64, weak=16)
stats(Z.Sized, True, K=8, strong=64, weak=16, kmin=64, kmax=1024, k0=64)
