# Day-1 spark: aligned long speculation for edit-apply (lossless greedy) vs vanilla greedy vs HF prompt lookup.
import subprocess, sys, time, random, json
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "datasets", "accelerate", "arctic-inference", "pandas", "pyarrow"], check=False)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
MODEL = "Kortix/FastApply-1.5B-v1.0"
N_ITEMS = 48
dev = "cuda:0"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
EOS = set([tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|im_end|>")] if "<|im_end|>" in tok.get_vocab() else []))
print("GPU", torch.cuda.get_device_name(0), "model", MODEL, flush=True)
print("default generation_config:", model.generation_config.to_diff_dict(), flush=True)
PURE = dict(do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)

def build_prompt(before, instr):
    msgs = [{"role": "system", "content": "You are a code editor. Apply the instruction and output the complete updated file only, in one ```python block."},
            {"role": "user", "content": f"Instruction: {instr}\n\n```python\n{before}\n```"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def sync(): torch.cuda.synchronize()

@torch.no_grad()
def vanilla(ids, max_new):
    out = model.generate(ids, max_new_tokens=max_new, **PURE)
    return out[0, ids.shape[1]:].tolist()

@torch.no_grad()
def prompt_lookup(ids, max_new):
    out = model.generate(ids, max_new_tokens=max_new, prompt_lookup_num_tokens=10, **PURE)
    return out[0, ids.shape[1]:].tolist()

class Index:
    """n-gram -> sorted source positions (end of n-gram), built once per prompt."""
    NS = (32, 16, 12, 8, 6, 4, 3, 2)
    def __init__(self, src):
        self.src = src; self.m = {n: {} for n in self.NS}
        for n in self.NS:
            d = self.m[n]
            for i in range(len(src) - n + 1): d.setdefault(tuple(src[i:i + n]), []).append(i + n)
    def find(self, gen, prev_ptr):
        import bisect
        for n in self.NS:
            if len(gen) < n: continue
            pos = self.m[n].get(tuple(gen[-n:]))
            if not pos: continue
            j = bisect.bisect_left(pos, prev_ptr)
            return pos[j] if j < len(pos) else pos[-1]   # nearest at/after previous pointer, else latest before
        return None

@torch.no_grad()
def aligned(ids, max_new, kmax=2048, stats=None):
    src = ids[0].tolist(); idx = Index(src)
    out = model(ids, use_cache=True); cache = out.past_key_values
    last = int(out.logits[0, -1].argmax()); gen = [last]
    ptr = idx.find(gen, 0); k = 64; fwd = 1
    while len(gen) < max_new and last not in EOS:
        if ptr is not None:
            draft = src[ptr:ptr + k]
        else:   # unanchored: like HF prompt lookup, copy what followed the latest earlier match inside our own output
            draft = []
            for n in (4, 3, 2):
                if len(gen) <= n: continue
                pat = gen[-n:]
                for i in range(len(gen) - n - 1, -1, -1):
                    if gen[i:i + n] == pat:
                        draft = gen[i + n:i + n + 10]; break
                if draft: break
        draft = draft[:max_new - len(gen)]
        inp = torch.tensor([[last] + draft], device=dev)
        base = cache.get_seq_length()
        o = model(inp, past_key_values=cache, use_cache=True); fwd += 1
        preds = o.logits[0].argmax(-1).tolist()
        a = 0
        while a < len(draft) and draft[a] == preds[a]: a += 1
        new = draft[:a] + [preds[a]]
        cache.crop(-(len(draft) - a)) if len(draft) > a else None
        for t in new:
            gen.append(t)
            if t in EOS: break
        last = gen[-1]
        if ptr is not None and draft and a == len(draft) and ptr + a < len(src) and src[ptr + a] == preds[a]:
            ptr += a + 1; k = min(kmax, k * 2)   # all accepted (+ bonus token matches source): keep copying, grow window
        elif ptr is not None and draft and a == len(draft):
            k = min(kmax, k * 2); ptr = idx.find(gen, ptr + a)
        else:
            k = max(16, min(kmax, 2 * a + 16))   # shrink to just past the miss, not blind halving
            ptr = idx.find(gen, (ptr or 0) + a)
    if stats is not None: stats["fwd"] = fwd
    return gen[:max_new]


import numpy as np
from arctic_inference.suffix_decoding import SuffixDecodingCache
@torch.no_grad()
def suffix_run(ids, max_new, sc, rid, depth, factor, stats=None):
    prompt = ids[0].tolist()
    sc.start_request(rid, np.array(prompt, dtype=np.int32))
    out = model(ids, use_cache=True); cache = out.past_key_values
    last = int(out.logits[0, -1].argmax()); gen = [last]; sc.add_active_response(rid, [last]); fwd = 1
    while len(gen) < max_new and last not in EOS:
        d = sc.speculate(rid, np.array(prompt + gen, dtype=np.int32), max_spec_tokens=depth, max_spec_factor=factor, min_token_prob=0.1, use_tree_spec=False)
        draft = list(d.token_ids)[:max_new - len(gen)]
        inp = torch.tensor([[last] + draft], device=dev)
        o = model(inp, past_key_values=cache, use_cache=True); fwd += 1
        preds = o.logits[0].argmax(-1).tolist()
        a = 0
        while a < len(draft) and draft[a] == preds[a]: a += 1
        new = draft[:a] + [preds[a]]
        cache.crop(-(len(draft) - a)) if len(draft) > a else None
        added = []
        for t in new:
            gen.append(t); added.append(t)
            if t in EOS: break
        sc.add_active_response(rid, added)
        last = gen[-1]
    sc.stop_request(rid)
    if stats is not None: stats["fwd"] = fwd
    return gen[:max_new]

def trim(t):
    out = []
    for x in t:
        out.append(x)
        if x in EOS: break
    return out



import pandas as pd, statistics
from huggingface_hub import hf_hub_download
df = pd.read_parquet(hf_hub_download("Kortix/FastApply-dataset-v1.0", "test.parquet", repo_type="dataset")).sample(frac=1, random_state=0)
TEMPLATE = ("<|im_start|>system\nYou are a coding assistant that helps merge code updates, ensuring every modification is fully integrated.<|im_end|>\n"
            "<|im_start|>user\nMerge all changes from the <update> snippet into the <code> below.\n- Preserve the code's structure, order, comments, and indentation exactly.\n"
            "- Output only the updated code, enclosed within <updated-code> and </updated-code> tags.\n- Do not include any additional text, explanations, placeholders, ellipses, or code fences.\n\n"
            "<code>{o}</code>\n\n<update>{u}</update>\n\nProvide the complete updated code.<|im_end|>\n<|im_start|>assistant\n")
# ---- Part B: cost curve (seconds per verification pass vs draft length, at several context lengths)
curve = {}
with torch.no_grad():
    for C in (1024, 2048, 4096):
        x = torch.randint(100, 30000, (1, C), device=dev); o = model(x, use_cache=True); cache = o.past_key_values
        for L in (0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047):
            ts = []
            for _ in range(5):
                inp = torch.randint(100, 30000, (1, L + 1), device=dev); sync(); t0 = time.perf_counter()
                model(inp, past_key_values=cache, use_cache=True); sync(); ts.append(time.perf_counter() - t0)
                cache.crop(C)
            curve[f"{C}:{L}"] = statistics.median(ts)
        print("curve C", C, {k: round(v * 1000, 1) for k, v in curve.items() if k.startswith(f"{C}:")}, flush=True)
        del cache; torch.cuda.empty_cache()
# ---- Part A: oracle outputs (lossless fast generation) + real timings on a subset for validating the cost model
sc = SuffixDecodingCache(max_tree_depth=256)
recs = []; N = 200
for i, (_, r) in enumerate(df.head(N).iterrows()):
    o_code, u, f = str(r["original_code"]), str(r["update_snippet"]), str(r["final_code"])
    ids = tok(TEMPLATE.format(o=o_code, u=u), return_tensors="pt").input_ids.to(dev)
    if ids.shape[1] > 6000: continue
    max_new = min(4096, len(tok(o_code).input_ids) + len(tok(u).input_ids) + 64)
    st = {}; sync(); t0 = time.perf_counter(); g = trim(aligned(ids, max_new, stats=st)); sync(); ta = time.perf_counter() - t0
    rec = {"i": i, "prompt": ids[0].tolist(), "out": g, "orig": o_code, "snip": u, "final": f, "aligned_s": ta, "aligned_fwd": st["fwd"]}
    if i < 30:   # validation subset: real suffix timing + pure vanilla timing
        st2 = {}; sync(); t0 = time.perf_counter(); g2 = trim(suffix_run(ids, max_new, sc, i, 256, 4.0, stats=st2)); sync(); rec["suffix_s"] = time.perf_counter() - t0; rec["suffix_fwd"] = st2["fwd"]; rec["suffix_same"] = g2 == g
        if i < 10:
            sync(); t0 = time.perf_counter(); g3 = trim(vanilla(ids, max_new)); sync(); rec["vanilla_s"] = time.perf_counter() - t0; rec["vanilla_same"] = g3 == g
    recs.append(rec)
    print(f"[{i+1}/{N}] out {len(g)} tok aligned {ta:.2f}s fwd {st['fwd']}" + (f" suffix {rec['suffix_s']:.2f}s" if 'suffix_s' in rec else "") + (f" vanilla {rec['vanilla_s']:.2f}s same {rec['vanilla_same']}" if 'vanilla_s' in rec else ""), flush=True)
json.dump({"model": MODEL, "gpu": torch.cuda.get_device_name(0), "curve": curve, "records": recs}, open("oracle.json", "w"))
print("ORACLE DONE", len(recs), flush=True)
