# Held-out check 2: per-edit SuffixDecoding passes (linear + tree) and drafted tokens per pass, all 199 real outputs.
# Output: /kaggle/working/suffix_per_record.json (ratios on any split are computed locally).
import subprocess, sys, json, glob, os
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "arctic-inference==0.3.0"], check=False)
import numpy as np
from arctic_inference.suffix_decoding import SuffixDecodingCache
D = os.path.dirname(glob.glob("/kaggle/input/**/oracle.json", recursive=True)[0])
d = json.load(open(f"{D}/oracle.json")); R = d["records"]
def suffix(sc, rid, prompt, gt, tree):
    sc.start_request(rid, np.array(prompt, dtype=np.int32)); resp = [gt[0]]; sc.add_active_response(rid, [gt[0]]); steps = 1; drafted = 0
    while len(resp) < len(gt):
        dd = sc.speculate(rid, np.array(prompt + resp, dtype=np.int32), max_spec_tokens=256, max_spec_factor=4.0, min_token_prob=0.1, use_tree_spec=tree)
        toks, par = list(dd.token_ids), list(dd.parents); node, a = -1, 0; drafted += len(toks)
        while len(resp) + a < len(gt):
            ch = [i for i, p in enumerate(par) if p == node and toks[i] == gt[len(resp) + a]]
            if not ch: break
            node = ch[0]; a += 1
        new = gt[len(resp):len(resp) + a + 1]; resp += new; sc.add_active_response(rid, new); steps += 1
    sc.stop_request(rid); return steps, drafted
out = {"out_len": [len(r["out"]) for r in R]}
for tree in (False, True):
    sc = SuffixDecodingCache(max_tree_depth=256)
    res = [suffix(sc, i, r["prompt"], r["out"], tree) for i, r in enumerate(R)]
    key = "tree" if tree else "linear"
    out[key + "_passes"] = [p for p, _ in res]; out[key + "_drafted"] = [t for _, t in res]
    P, T, O = sum(out[key + "_passes"]), sum(out[key + "_drafted"]), sum(out["out_len"])
    print(f"suffix {key}: passes all {P} first30 {sum(out[key + '_passes'][:30])}; drafted {T} = {T/O:.2f} per output token, {T/P:.0f} per pass", flush=True)
json.dump(out, open("/kaggle/working/suffix_per_record.json", "w"))
print("HELDOUT2 DONE")
