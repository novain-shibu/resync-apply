# Held-out check (skeptic step 1): SuffixDecoding vs our drafter on ALL 199 real model outputs; report edits 30..198 (never tuned on).
import subprocess, sys, json, glob, os
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "arctic-inference"], check=False)
import numpy as np
from arctic_inference.suffix_decoding import SuffixDecodingCache
D = os.path.dirname(glob.glob("/kaggle/input/**/oracle.json", recursive=True)[0])
sys.path.insert(0, D)
from drafter import Drafter
d = json.load(open(f"{D}/oracle.json")); R = d["records"]; aux = json.load(open(f"{D}/vocab_aux.json")); NL, UPD = set(aux["nl"]), aux["upd"]
def suffix(sc, rid, prompt, gt, tree):
    sc.start_request(rid, np.array(prompt, dtype=np.int32)); resp = [gt[0]]; sc.add_active_response(rid, [gt[0]]); steps = 1
    while len(resp) < len(gt):
        dd = sc.speculate(rid, np.array(prompt + resp, dtype=np.int32), max_spec_tokens=256, max_spec_factor=4.0, min_token_prob=0.1, use_tree_spec=tree)
        toks, par = list(dd.token_ids), list(dd.parents); node, a = -1, 0
        while len(resp) + a < len(gt):
            ch = [i for i, p in enumerate(par) if p == node and toks[i] == gt[len(resp) + a]]
            if not ch: break
            node = ch[0]; a += 1
        new = gt[len(resp):len(resp) + a + 1]; resp += new; sc.add_active_response(rid, new); steps += 1
    sc.stop_request(rid); return steps
def ours(rec, **kw):
    src, gt = rec["prompt"], rec["out"]; dr = Drafter(src, NL, UPD, **kw); gen = [gt[0]]; pos = 1; passes = 1; dr.start(gen)
    while pos < len(gt):
        br = dr.propose(gen, len(gt) - pos); best_a, best_i = 0, 0
        for i, b in enumerate(br):
            a = 0
            while a < len(b) and b[a] == gt[pos + a]: a += 1
            if a > best_a: best_a, best_i = a, i
        new = gt[pos:pos + best_a + 1]; gen += new; pos += len(new); passes += 1; dr.update(gen, best_i, best_a, br)
    return passes
S = {}
for tree in (False, True):
    sc = SuffixDecodingCache(max_tree_depth=256)
    S[tree] = [suffix(sc, i, r["prompt"], r["out"], tree) for i, r in enumerate(R)]
real30 = sum(r["suffix_fwd"] for r in R if "suffix_fwd" in r)
print(f"sanity: simulated suffix linear on first 30 = {sum(S[False][:30])} (real GPU 868)")
cfgs = {"untuned K8 s64 w16 kmin16 kmax2048": dict(K=8, strong=64, weak=16), "time-tuned K8 s32 w8 kmin64 kmax384": dict(K=8, strong=32, weak=8, kmin=64, kmax=384)}
for name, kw in cfgs.items():
    O = [ours(r, **kw) for r in R]
    for lab, sl in (("first30 (tuned on)", slice(0, 30)), ("HELD-OUT 30..198", slice(30, None)), ("all 199", slice(None))):
        o, sl_, st = sum(O[sl]), sum(S[False][sl]), sum(S[True][sl])
        print(f"{name:38s} {lab:20s} ours {o:6d}  suffix linear {sl_:6d} ({sl_/o:.2f}x)  suffix TREE {st:6d} ({st/o:.2f}x)")
    import random
    random.seed(0); ho = list(range(30, len(R))); boots = []
    for _ in range(2000):
        idx = [random.choice(ho) for _ in ho]; boots.append(sum(S[False][i] for i in idx) / sum(O[i] for i in idx))
    boots.sort(); print(f"   held-out bootstrap 95% CI vs suffix linear: {boots[50]:.2f}-{boots[1950]:.2f}")
print("HELDOUT DONE")
