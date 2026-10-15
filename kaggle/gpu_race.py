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




# Multi-branch copy drafter for edit-apply (lossless greedy verification happens outside).
# One class used by both the CPU simulator and the GPU loop, so both run the same logic.
from bisect import bisect_left


class Drafter:
    NS = (32, 16, 12, 8, 6, 4, 3, 2, 1)

    def __init__(self, src, nl_tokens, upd_ids, K=8, strong=64, weak=16, kmax=2048, k0=64, kmin=16):
        self.src = src
        self.K, self.strong, self.weak, self.kmax = K, strong, weak, kmax
        self.k = k0
        self.kmin = kmin
        self.nl = nl_tokens
        self.m = {n: {} for n in self.NS}
        for n in self.NS:
            d = self.m[n]
            for i in range(len(src) - n + 1):
                d.setdefault(tuple(src[i:i + n]), []).append(i + n)
        self.split = self._find(upd_ids)
        self.fo, self.fs = 0, self.split          # read-frontiers in original / snippet
        self.strength = {}
        self.rs = set()
        self.ptrs = []
        self.prev = 0

    def _find(self, pat):
        s = self.src
        for i in range(len(s) - len(pat) + 1):
            if s[i:i + len(pat)] == pat:
                return i + len(pat)
        return len(s)

    def _next_line(self, p):
        s = self.src
        while p < len(s) and s[p] not in self.nl:
            p += 1
        return p + 1

    def _cands(self, gen, prev, K):
        out = []
        for n in self.NS:
            if len(gen) < n:
                continue
            pos = self.m[n].get(tuple(gen[-n:]))
            if not pos:
                continue
            for p in sorted(pos, key=lambda q: (q < prev, abs(q - prev))):
                if p not in out:
                    out.append(p)
                    self.strength[p] = max(self.strength.get(p, 0), n)
                if len(out) >= K:
                    return out
        return out

    def start(self, gen):
        self.ptrs = self._cands(gen, 0, self.K)
        self.prev = self.ptrs[0] if self.ptrs else 0

    def propose(self, gen, remaining):
        """Return list of branches (token lists). Branch 0 is the long main guess."""
        src = self.src
        branches = []
        for j, p in enumerate(self.ptrs):
            if j == 0:
                L = self.k
            else:
                L = min(self.k, self.strong if (self.strength.get(p, 0) >= 4 or p in self.rs) else self.weak)
            branches.append(src[p:p + L][:remaining])
        if not branches:   # unanchored: copy what followed the latest earlier match in our own output
            for n in (4, 3, 2):
                if len(gen) <= n:
                    continue
                pat = gen[-n:]
                for i in range(len(gen) - n - 1, -1, -1):
                    if gen[i:i + n] == pat:
                        branches = [gen[i + n:i + n + 10][:remaining]]
                        self._self_draft = True
                        return branches
        self._self_draft = False
        return branches

    def update(self, gen, best_i, a, branches):
        """gen already includes the accepted tokens and the bonus token."""
        src = self.src
        chosen = self.ptrs[best_i] if (self.ptrs and not self._self_draft and best_i < len(self.ptrs)) else None
        if chosen is not None:
            if chosen < self.split:
                self.fo = max(self.fo, chosen + a)
            else:
                self.fs = max(self.fs, chosen + a)
        full = chosen is not None and branches and a == len(branches[best_i]) and a > 0
        if full and chosen + a < len(src) and src[chosen + a] == gen[-1]:
            self.prev = chosen + a + 1
            self.ptrs = [self.prev] + [p for p in self._cands(gen, self.prev, self.K) if p != self.prev][:self.K - 1]
            self.k = min(self.kmax, self.k * 2)
            return
        self.prev = (chosen if chosen is not None else self.prev) + a
        self.k = max(self.kmin, min(self.kmax, 2 * a + 16))
        rs = []
        if chosen is not None:
            t = gen[-1]
            if chosen + a + 1 < len(src) and src[chosen + a + 1] == t:
                rs.append(chosen + a + 2)
            for q in (chosen + a + 1, chosen + a, chosen + a + 2):
                if 0 <= q < len(src) and q not in rs:
                    rs.append(q)
        self.rs = set(rs)
        ptrs = rs + [p for p in self._cands(gen, self.prev, self.K) if p not in rs]
        fr = [p for p in (self.fo, self._next_line(self.fo), self.fs, self._next_line(self.fs)) if 0 <= p < len(src)]
        ptrs = [p for p in ptrs if p not in fr][:max(self.K, len(rs) + self.K)]
        self.ptrs = ptrs[:len(rs)] + fr + ptrs[len(rs):]

import pandas as pd
from huggingface_hub import hf_hub_download
df = pd.read_parquet(hf_hub_download("Kortix/FastApply-dataset-v1.0", "test.parquet", repo_type="dataset")).sample(frac=1, random_state=0)
TEMPLATE = ("<|im_start|>system\nYou are a coding assistant that helps merge code updates, ensuring every modification is fully integrated.<|im_end|>\n"
            "<|im_start|>user\nMerge all changes from the <update> snippet into the <code> below.\n- Preserve the code's structure, order, comments, and indentation exactly.\n"
            "- Output only the updated code, enclosed within <updated-code> and </updated-code> tags.\n- Do not include any additional text, explanations, placeholders, ellipses, or code fences.\n\n"
            "<code>{o}</code>\n\n<update>{u}</update>\n\nProvide the complete updated code.<|im_end|>\n<|im_start|>assistant\n")
VOC = tok.get_vocab()
NL = set(i for s, i in VOC.items() if "Ċ" in s)
UPD = tok.encode("<update>", add_special_tokens=False)
NEG = torch.finfo(model.dtype).min


def build_tree(last, branches):
    toks, parent, depth, children = [last], [-1], [0], [{}]
    for b in branches:
        cur = 0
        for t in b:
            nxt = children[cur].get(t)
            if nxt is None:
                nxt = len(toks); toks.append(t); parent.append(cur); depth.append(depth[cur] + 1); children.append({}); children[cur][t] = nxt
            cur = nxt
    return toks, parent, depth, children


import numpy as _np
def tree_mask(parent, base, mode):
    q = len(parent)
    allow = _np.zeros((q, q), dtype=bool)
    allow[0, 0] = True
    for i in range(1, q):
        allow[i] = allow[parent[i]]; allow[i, i] = True
    allow_t = torch.from_numpy(allow).to(dev)
    full = torch.cat([torch.ones((q, base), dtype=torch.bool, device=dev), allow_t], 1)[None, None]
    if mode == "bool":
        return full
    m = torch.zeros(full.shape, dtype=model.dtype, device=dev); m.masked_fill_(~full, NEG); return m


def keep_kv(cache, base, keep_rel):
    idx = torch.tensor([base + i for i in keep_rel], device=dev)
    if hasattr(cache, "layers"):
        for l in cache.layers:
            k, v = l.keys, l.values
            l.keys = torch.cat([k[:, :, :base], k.index_select(2, idx)], 2)
            l.values = torch.cat([v[:, :, :base], v.index_select(2, idx)], 2)
    else:
        for i in range(len(cache.key_cache)):
            k, v = cache.key_cache[i], cache.value_cache[i]
            cache.key_cache[i] = torch.cat([k[:, :, :base], k.index_select(2, idx)], 2)
            cache.value_cache[i] = torch.cat([v[:, :, :base], v.index_select(2, idx)], 2)


def argmax_rows(h):
    out = []
    W = model.lm_head.weight
    for c in range(0, h.shape[0], 256):
        out.append((h[c:c + 256] @ W.T).argmax(-1))
    return torch.cat(out).tolist()


MASK_MODE = "float"


@torch.no_grad()
def tree_forward(cache, toks, parent, depth):
    base = cache.get_seq_length()
    ids = torch.tensor([toks], device=dev)
    pos = torch.tensor([[base + d for d in depth]], device=dev)
    o = model.model(input_ids=ids, past_key_values=cache, attention_mask=tree_mask(parent, base, MASK_MODE), position_ids=pos, use_cache=True)
    return base, argmax_rows(o.last_hidden_state[0])


@torch.no_grad()
def self_test():
    global MASK_MODE
    ids = tok(TEMPLATE.format(o="def f(x):\n    return x + 1\n", u="def f(x):\n    return x + 2\n"), return_tensors="pt").input_ids.to(dev)
    for mode in ("float", "bool"):
        MASK_MODE = mode
        try:
            o = model(ids, use_cache=True); cache = o.past_key_values; last = int(o.logits[0, -1].argmax())
            branch = ids[0, -20:].tolist()
            o2 = model(ids, use_cache=True); c2 = o2.past_key_values
            ref = model(torch.tensor([[last] + branch], device=dev), past_key_values=c2, use_cache=True).logits[0].argmax(-1).tolist()
            toks, parent, depth, children = build_tree(last, [branch, branch[:5] + [11, 12, 13]])
            base, preds = tree_forward(cache, toks, parent, depth)
            got = [preds[n] for n in [0] + list(range(1, 1 + len(branch)))]
            ok = got == ref
            print("self-test mask", mode, "match", ok, sum(a == b for a, b in zip(got, ref)), "/", len(ref), flush=True)
            if ok:
                return mode
        except Exception as e:
            print("self-test mask", mode, "error", type(e).__name__, str(e)[:300], flush=True)
    raise SystemExit("tree mask self-test failed")


@torch.no_grad()
def tree_run(ids, max_new, stats=None, K=8, strong=32, weak=8, kmin=64, kmax=384, jacobi=False, jlen=128):
    src = ids[0].tolist()
    o = model(ids, use_cache=True); cache = o.past_key_values
    last = int(o.logits[0, -1].argmax()); gen = [last]; fwd = 1; jw = 0
    dr = Drafter(src, NL, UPD, K=K, strong=strong, weak=weak, kmin=kmin, kmax=kmax, k0=min(64, kmax)); dr.start(gen)
    residue = []
    while len(gen) < max_new and last not in EOS:
        br = dr.propose(gen, max_new - len(gen))
        nd = len(br)
        if jacobi and residue:
            br = br + [residue[:min(jlen, max_new - len(gen))]]
        toks, parent, depth, children = build_tree(last, br)
        base, preds = tree_forward(cache, toks, parent, depth); fwd += 1
        cur, path = 0, []
        while True:
            t = preds[cur]; nxt = children[cur].get(t)
            if nxt is None:
                bonus = t; break
            path.append(nxt); cur = nxt
        acc = [toks[n] for n in path]
        best_i, best_a = 0, -1
        for i, b in enumerate(br):
            a = 0
            while a < len(b) and a < len(acc) and b[a] == acc[a]:
                a += 1
            if a > best_a:
                best_a, best_i = a, i
        # Jacobi residue: model predictions AFTER the mismatch along the winning branch (free, already computed)
        residue = []
        if jacobi:
            b = br[best_i]; node = 0; nodes = []
            for t in b:
                node = children[node].get(t)
                if node is None: break
                nodes.append(node)
            if len(nodes) > best_a:
                residue = [preds[n] for n in nodes[best_a:]]   # predictions after the mismatched token (substitution assumption)
        if best_i >= nd and best_a > 0: jw += 1
        if path == list(range(1, len(path) + 1)):
            cache.crop(base + 1 + len(path))
        else:
            keep_kv(cache, base, [0] + path)
        for t in acc + [bonus]:
            gen.append(t)
            if t in EOS:
                break
        last = gen[-1]
        dr.update(gen, best_i if best_i < nd else len(dr.ptrs) + 99, max(best_a, 0), br[:nd] + [[]] * (best_i - nd + 1) if best_i >= nd else br)
    if stats is not None:
        stats["fwd"] = fwd; stats["jw"] = jw
    return gen[:max_new]


MASK_MODE = self_test()
import json
TABLE_JSON = r'''[[["ngram", "n1", "0", "orig", "mid"], [0.0, 99]], [["ngram", "n1", ">128", "orig", "mid"], [748.0, 993]], [["resync", "n0", "1-2", "orig", "mid"], [1529.0, 1194]], [["ngram", "n1", "neg_near", "orig", "mid"], [487.0, 599]], [["resync", "n1", "0", "orig", "mid"], [4.0, 117]], [["frontier", "n0", "3-16", "orig", "bol"], [0.0, 252]], [["frontier", "n0", ">128", "snip", "mid"], [545.0, 141]], [["frontier", "n0", ">128", "snip", "bol"], [1.0, 221]], [["resync", "n0", "0", "orig", "mid"], [556.0, 472]], [["resync", "n0", "1-2", "orig", "bol"], [25.0, 346]], [["frontier", "n1", "17-128", "orig", "mid"], [0.0, 109]], [["frontier", "n0", "17-128", "orig", "bol"], [0.0, 235]], [["ngram", "n2-3", "17-128", "orig", "mid"], [18671.0, 435]], [["ngram", "n1", "17-128", "orig", "mid"], [6339.0, 896]], [["continue", "n>=8", "0", "orig", "mid"], [134205.0, 545]], [["ngram", "n2-3", "3-16", "orig", "mid"], [1181.0, 103]], [["ngram", "n4-6", "neg_far", "orig", "mid"], [6745.0, 282]], [["ngram", "n1", "1-2", "orig", "mid"], [1.0, 10]], [["ngram", "n1", "3-16", "orig", "mid"], [841.0, 146]], [["ngram", "n>=8", ">128", "orig", "mid"], [10441.0, 170]], [["ngram", "n2-3", ">128", "orig", "mid"], [1446.0, 449]], [["ngram", "n1", ">128", "snip", "mid"], [170.0, 111]], [["ngram", "n2-3", "17-128", "snip", "bol"], [6.0, 6]], [["ngram", "n2-3", "neg_near", "orig", "bol"], [69.0, 27]], [["resync", "n0", "0", "orig", "bol"], [0.0, 245]], [["frontier", "n0", "3-16", "snip", "mid"], [401.0, 7]], [["frontier", "n0", "3-16", "snip", "bol"], [0.0, 66]], [["ngram", "n2-3", "neg_far", "orig", "bol"], [1281.0, 94]], [["continue", "n>=8", "0", "snip", "mid"], [5390.0, 53]], [["ngram", "n2-3", "17-128", "snip", "mid"], [224.0, 19]], [["ngram", "n2-3", "neg_far", "orig", "mid"], [6039.0, 719]], [["resync", "n0", "1-2", "snip", "mid"], [146.0, 191]], [["ngram", "n1", "neg_far", "orig", "mid"], [2732.0, 912]], [["frontier", "n0", "neg_far", "orig", "bol"], [4.0, 107]], [["resync", "n0", "0", "snip", "mid"], [0.0, 63]], [["resync", "n0", "1-2", "snip", "bol"], [1.0, 28]], [["frontier", "n0", "17-128", "snip", "mid"], [37.0, 16]], [["frontier", "n0", "17-128", "snip", "bol"], [3.0, 63]], [["ngram", "n1", "17-128", "snip", "mid"], [111.0, 75]], [["ngram", "n1", "3-16", "snip", "mid"], [98.0, 15]], [["continue", "n>=8", "0", "snip", "bol"], [261.0, 5]], [["ngram", "n>=8", "neg_near", "orig", "bol"], [553.0, 11]], [["ngram", "n1", "neg_near", "orig", "bol"], [132.0, 103]], [["frontier", "n1", "neg_far", "orig", "mid"], [72.0, 6]], [["resync", "n0", "0", "snip", "bol"], [0.0, 46]], [["ngram", "n1", "neg_far", "orig", "bol"], [2217.0, 118]], [["ngram", "n2-3", ">128", "snip", "mid"], [2370.0, 99]], [["ngram", "n2-3", "neg_near", "orig", "mid"], [279.0, 333]], [["ngram", "n>=8", "neg_far", "orig", "bol"], [2601.0, 44]], [["ngram", "n>=8", "neg_far", "orig", "mid"], [14111.0, 243]], [["frontier", "n0", "neg_far", "orig", "mid"], [1.0, 57]], [["ngram", "n2-3", "3-16", "snip", "mid"], [9.0, 9]], [["ngram", "n1", "neg_near", "snip", "mid"], [21.0, 52]], [["resync", "n2-3", "0", "orig", "mid"], [31.0, 22]], [["frontier", "n2-3", "neg_far", "orig", "mid"], [0.0, 5]], [["continue", "n>=8", "0", "orig", "bol"], [18145.0, 68]], [["ngram", "n2-3", ">128", "orig", "bol"], [213.0, 58]], [["ngram", "n2-3", "3-16", "orig", "bol"], [15.0, 9]], [["ngram", "n1", "3-16", "orig", "bol"], [18.0, 17]], [["ngram", "n1", "17-128", "orig", "bol"], [77.0, 77]], [["ngram", "n>=8", "neg_near", "orig", "mid"], [265.0, 47]], [["ngram", "n1", ">128", "orig", "bol"], [28.0, 92]], [["frontier", "n1", ">128", "snip", "bol"], [12.0, 9]], [["resync", "n1", "1-2", "orig", "mid"], [439.0, 120]], [["resync", "n1", "0", "orig", "bol"], [0.0, 13]], [["frontier", "n1", "3-16", "orig", "bol"], [0.0, 5]], [["frontier", "n0", ">128", "orig", "mid"], [9.0, 157]], [["frontier", "n1", ">128", "orig", "bol"], [4.0, 7]], [["frontier", "n0", ">128", "orig", "bol"], [2.0, 302]], [["ngram", "n4-6", "neg_near", "orig", "mid"], [321.0, 111]], [["ngram", "n4-6", "17-128", "orig", "mid"], [1136.0, 102]], [["resync", "n2-3", "0", "orig", "bol"], [0.0, 4]], [["frontier", "n2-3", ">128", "orig", "bol"], [0.0, 2]], [["ngram", "n2-3", "17-128", "orig", "bol"], [126.0, 31]], [["ngram", "n>=8", ">128", "snip", "mid"], [2643.0, 34]], [["ngram", "n4-6", ">128", "orig", "mid"], [1815.0, 131]], [["ngram", "n4-6", ">128", "snip", "mid"], [653.0, 34]], [["ngram", "n>=8", "neg_near", "snip", "mid"], [117.0, 3]], [["ngram", "n4-6", "17-128", "snip", "mid"], [328.0, 11]], [["frontier", "n>=8", ">128", "snip", "bol"], [74.0, 2]], [["ngram", "n4-6", "neg_far", "orig", "bol"], [1858.0, 75]], [["frontier", "n2-3", ">128", "orig", "mid"], [0.0, 1]], [["resync", "n2-3", "1-2", "orig", "mid"], [243.0, 85]], [["ngram", "n4-6", "3-16", "orig", "mid"], [14.0, 18]], [["frontier", "n0", "3-16", "orig", "mid"], [0.0, 10]], [["frontier", "n1", ">128", "orig", "mid"], [0.0, 43]], [["ngram", "n>=8", "17-128", "snip", "mid"], [73.0, 6]], [["ngram", "n1", "neg_far", "snip", "mid"], [3.0, 4]], [["ngram", "n1", "neg_far", "snip", "bol"], [0.0, 1]], [["ngram", "n1", "1-2", "snip", "mid"], [0.0, 1]], [["ngram", "n>=8", "neg_far", "snip", "bol"], [2.0, 1]], [["ngram", "n2-3", "1-2", "snip", "mid"], [0.0, 1]], [["ngram", "n2-3", "neg_near", "snip", "mid"], [34.0, 28]], [["frontier", "n0", "17-128", "orig", "mid"], [0.0, 60]], [["frontier", "n1", "17-128", "snip", "bol"], [0.0, 1]], [["ngram", "n>=8", "17-128", "orig", "mid"], [378.0, 39]], [["resync", "n1", "1-2", "orig", "bol"], [1.0, 4]], [["ngram", "n2-3", ">128", "snip", "bol"], [0.0, 1]], [["frontier", "n1", ">128", "snip", "mid"], [0.0, 29]], [["resync", "n4-6", "0", "snip", "mid"], [0.0, 1]], [["frontier", "n2-3", "17-128", "orig", "mid"], [0.0, 1]], [["frontier", "n4-6", ">128", "snip", "mid"], [0.0, 1]], [["ngram", "n2-3", "neg_far", "snip", "mid"], [4.0, 5]], [["resync", "n1", "0", "snip", "mid"], [0.0, 3]], [["frontier", "n2-3", "3-16", "orig", "mid"], [0.0, 1]], [["frontier", "n1", "17-128", "snip", "mid"], [0.0, 2]], [["frontier", "n1", "3-16", "snip", "mid"], [0.0, 1]], [["resync", "n4-6", "1-2", "orig", "mid"], [63.0, 22]], [["resync", "n1", "17-128", "orig", "mid"], [0.0, 1]], [["ngram", "n4-6", "neg_near", "snip", "mid"], [6.0, 4]], [["ngram", "n>=8", ">128", "orig", "bol"], [603.0, 22]], [["resync", "n2-3", "neg_near", "orig", "mid"], [0.0, 1]], [["ngram", "n4-6", ">128", "orig", "bol"], [240.0, 18]], [["ngram", "n4-6", "17-128", "orig", "bol"], [34.0, 3]], [["frontier", "n1", "17-128", "orig", "bol"], [27.0, 3]], [["frontier", "n0", "neg_near", "orig", "bol"], [0.0, 31]], [["ngram", "n>=8", "17-128", "orig", "bol"], [28.0, 4]], [["frontier", "n0", "neg_near", "orig", "mid"], [0.0, 19]], [["resync", "n1", "1-2", "snip", "mid"], [28.0, 7]], [["ngram", "n2-3", "neg_near", "snip", "bol"], [4.0, 2]], [["ngram", "n1", "17-128", "snip", "bol"], [2.0, 4]], [["resync", "n2-3", "1-2", "snip", "bol"], [0.0, 1]], [["resync", "n2-3", "0", "snip", "bol"], [0.0, 1]], [["frontier", "n>=8", ">128", "orig", "bol"], [0.0, 4]], [["resync", "n2-3", "neg_far", "orig", "mid"], [0.0, 1]], [["resync", "n2-3", "1-2", "orig", "bol"], [0.0, 1]], [["resync", "n1", "neg_near", "orig", "mid"], [0.0, 1]], [["ngram", "n4-6", "neg_near", "orig", "bol"], [59.0, 9]], [["ngram", "n1", "neg_near", "snip", "bol"], [2.0, 2]], [["ngram", "n>=8", "3-16", "orig", "mid"], [0.0, 2]], [["resync", "n>=8", "0", "orig", "mid"], [0.0, 11]], [["resync", "n>=8", "1-2", "orig", "mid"], [5.0, 7]], [["resync", "n>=8", "1-2", "orig", "bol"], [0.0, 3]], [["resync", "n4-6", "0", "orig", "mid"], [0.0, 7]], [["ngram", "n2-3", "1-2", "orig", "mid"], [0.0, 1]], [["frontier", "n>=8", "neg_far", "orig", "bol"], [267.0, 2]], [["ngram", "n1", ">128", "snip", "bol"], [13.0, 2]], [["resync", "n2-3", "1-2", "snip", "mid"], [12.0, 1]], [["ngram", "n4-6", "17-128", "snip", "bol"], [0.0, 1]], [["frontier", "n1", "3-16", "snip", "bol"], [0.0, 1]], [["frontier", "n2-3", "neg_near", "orig", "bol"], [0.0, 1]]]'''

# ---------------- cascade attention: fast kernel over cached context + tiny masked attention inside the tree, merged exactly ----------------
import torch.nn.functional as F
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
try:
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    AttentionMaskInterface.register("cascade", sdpa_mask)
except Exception as e:
    print("mask interface register failed:", e, flush=True)
TREE_CTX = None   # (base, additive tree mask [q, q] float32) when a tree pass runs


def _rep(x, n):
    return x if n == 1 else x.repeat_interleave(n, dim=1)


def cascade_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    if TREE_CTX is None:
        return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
    base, tmask = TREE_CTX
    n = query.shape[1] // key.shape[1]
    q = query.shape[2]
    kc, vc = _rep(key[:, :, :base], n), _rep(value[:, :, :base], n)
    kt, vt = _rep(key[:, :, base:], n), _rep(value[:, :, base:], n)
    sc = scaling if scaling is not None else query.shape[-1] ** -0.5
    out_c, lse_c = torch.ops.aten._scaled_dot_product_efficient_attention(query.contiguous(), kc.contiguous(), vc.contiguous(), None, True, 0.0, False, scale=sc)[:2]
    lse_c = lse_c[:, :, :q].float()
    s = (query.float() @ kt.float().transpose(-1, -2)) * sc + tmask
    lse_t = torch.logsumexp(s, -1)
    out_t = torch.softmax(s, -1) @ vt.float()
    m = torch.maximum(lse_c, lse_t)
    wc, wt = torch.exp(lse_c - m)[..., None], torch.exp(lse_t - m)[..., None]
    out = (out_c.float() * wc + out_t * wt) / (wc + wt)
    return out.to(query.dtype).transpose(1, 2).contiguous(), None


AttentionInterface.register("cascade", cascade_attention)


def tree_small_mask(parent):
    q = len(parent)
    allow = _np.zeros((q, q), dtype=bool)
    allow[0, 0] = True
    for i in range(1, q):
        allow[i] = allow[parent[i]]; allow[i, i] = True
    m = torch.zeros((q, q), dtype=torch.float32, device=dev)
    m.masked_fill_(~torch.from_numpy(allow).to(dev), float("-inf"))
    return m


@torch.no_grad()
def tree_forward_cascade(cache, toks, parent, depth):
    global TREE_CTX
    base = cache.get_seq_length()
    ids = torch.tensor([toks], device=dev)
    pos = torch.tensor([[base + d for d in depth]], device=dev)
    TREE_CTX = (base, tree_small_mask(parent))
    model.config._attn_implementation = "cascade"
    try:
        o = model.model(input_ids=ids, past_key_values=cache, position_ids=pos, use_cache=True)
    finally:
        model.config._attn_implementation = "sdpa"; TREE_CTX = None
    return base, argmax_rows(o.last_hidden_state[0])


@torch.no_grad()
def cascade_self_test():
    """cascade predictions must equal the verified full-mask tree predictions on random trees."""
    import random
    random.seed(1); ok_all = True
    ids = tok(TEMPLATE.format(o=str(df.iloc[0]["original_code"]), u=str(df.iloc[0]["update_snippet"])), return_tensors="pt").input_ids.to(dev)
    src = ids[0].tolist()
    for trial in range(4):
        o1 = model(ids, use_cache=True); c1 = o1.past_key_values; last = int(o1.logits[0, -1].argmax())
        o2 = model(ids, use_cache=True); c2 = o2.past_key_values
        brs = []
        for b in range(1 + trial * 2):
            p = random.randrange(0, len(src) - 80); brs.append(src[p:p + random.choice([8, 16, 64])])
        toks, parent, depth, children = build_tree(last, brs)
        _, p_ref = tree_forward(c1, toks, parent, depth)
        _, p_cas = tree_forward_cascade(c2, toks, parent, depth)
        same = sum(a == b for a, b in zip(p_ref, p_cas))
        print(f"cascade self-test trial {trial}: tree size {len(toks)}, argmax agree {same}/{len(toks)}", flush=True)
        ok_all &= same >= len(toks) - 1
    if not ok_all:
        raise SystemExit("cascade self-test FAILED")


# ---------------- learned branch sizing ----------------
LEARNED_SRC = r"""
def dist_bucket(x):
    if x == 0: return "0"
    if x < 0: return "neg_far" if x < -128 else "neg_near"
    if x <= 2: return "1-2"
    if x <= 16: return "3-16"
    if x <= 128: return "17-128"
    return ">128"


class Learned(Drafter):
    TABLE = None   # key -> (sum, count)

    def feats(self, p):
        kind = self.kind.get(p, "ngram")
        n = self.strength.get(p, 0)
        nb = "n>=8" if n >= 8 else ("n4-6" if n >= 4 else ("n2-3" if n >= 2 else ("n1" if n == 1 else "n0")))
        region = "orig" if p < self.split else "snip"
        line = "bol" if p > 0 and self.src[p - 1] in self.nl else "mid"
        return (kind, nb, dist_bucket(p - self.prev), region, line)

    def score(self, p):
        if self.TABLE is None: return 0.0
        k = self.feats(p)
        s, c = self.TABLE.get(k, (0.0, 0))
        return (s + 1.0) / (c + 1.0)   # smoothed mean copy length

    def _tag(self):
        # remember how each candidate was produced
        pass

    def start(self, gen):
        self.kind = {}
        super().start(gen)
        for p in self.ptrs: self.kind.setdefault(p, "ngram")
        self._rank()

    def update(self, gen, best_i, a, branches):
        self.kind = {}
        old_rs = None
        super().update(gen, best_i, a, branches)
        for p in self.rs: self.kind[p] = "resync"
        fr = (self.fo, self._next_line(self.fo), self.fs, self._next_line(self.fs))
        for p in self.ptrs:
            if p not in self.kind: self.kind[p] = "frontier" if p in fr else "ngram"
        if self.ptrs and self.prev == self.ptrs[0] and a > 0 and best_i is not None:
            self.kind[self.ptrs[0]] = "continue"
        self._rank()

    def _rank(self):
        if self.TABLE is None or not self.ptrs: return
        keep_first = self.kind.get(self.ptrs[0]) == "continue"
        rest = self.ptrs[1:] if keep_first else self.ptrs
        rest = sorted(rest, key=lambda p: -self.score(p))
        self.ptrs = ([self.ptrs[0]] if keep_first else []) + rest
        # strong/weak lengths from the learned score instead of n-gram strength
        self.rs = set(p for p in self.ptrs if self.score(p) >= self.strong_thr)

    strong_thr = 1e9

"""
exec(LEARNED_SRC)
Learned.TABLE = {tuple(k): tuple(v) for k, v in json.loads(TABLE_JSON)}


class Sized(Learned):
    MULT, FLOOR, MAINCAP = 1.0, 8, 6.0

    def propose(self, gen, remaining):
        br = super().propose(gen, remaining)
        if self._self_draft or not br:
            return br
        out = []
        for j, (p, b) in enumerate(zip(self.ptrs, br)):
            e = self.score(p)
            if j == 0:
                L = len(b) if self.MAINCAP is None else min(len(b), max(self.FLOOR, int(self.MAINCAP * e)))
            else:
                L = min(len(b), max(self.FLOOR, int(self.MULT * e)))
            out.append(b[:L])
        return out


@torch.no_grad()
def fast_run(ids, max_new, stats=None, cascade=True, sized=True, hybrid=True, K=8, kmax=1024):
    src = ids[0].tolist()
    o = model(ids, use_cache=True); cache = o.past_key_values
    last = int(o.logits[0, -1].argmax()); gen = [last]; fwd = 1; conf = False; trees = 0
    cls = Sized if sized else Drafter
    dr = cls(src, NL, UPD, K=K, strong=64, weak=16, kmin=64, kmax=kmax, k0=64); dr.start(gen)
    while len(gen) < max_new and last not in EOS:
        br = dr.propose(gen, max_new - len(gen))
        if hybrid and conf:
            br = br[:1]
        if len(br) <= 1:   # linear check on the normal fast path
            draft = br[0] if br else []
            base = cache.get_seq_length()
            out = model(torch.tensor([[last] + draft], device=dev), past_key_values=cache, use_cache=True); fwd += 1
            preds = out.logits[0].argmax(-1).tolist()
            a = 0
            while a < len(draft) and draft[a] == preds[a]:
                a += 1
            acc, bonus, best_i = draft[:a], preds[a], 0
            if len(draft) > a:
                cache.crop(base + 1 + a)
        else:
            trees += 1
            toks, parent, depth, children = build_tree(last, br)
            base, preds = (tree_forward_cascade if cascade else tree_forward)(cache, toks, parent, depth); fwd += 1
            cur, path = 0, []
            while True:
                t = preds[cur]; nxt = children[cur].get(t)
                if nxt is None:
                    bonus = t; break
                path.append(nxt); cur = nxt
            acc = [toks[n] for n in path]
            if path == list(range(1, len(path) + 1)):
                cache.crop(base + 1 + len(path))
            else:
                keep_kv(cache, base, [0] + path)
            best_i, a = 0, -1
            for i, b in enumerate(br):
                x = 0
                while x < len(b) and x < len(acc) and b[x] == acc[x]:
                    x += 1
                if x > a:
                    a, best_i = x, i
        conf = bool(br) and a == len(br[best_i]) and a > 0
        for t in acc + [bonus]:
            gen.append(t)
            if t in EOS:
                break
        last = gen[-1]
        dr.update(gen, best_i, max(a, 0), br)
    if stats is not None:
        stats["fwd"] = fwd; stats["trees"] = trees
    return gen[:max_new]


cascade_self_test()
sc = SuffixDecodingCache(max_tree_depth=256)
names = ["new", "new_nocascade", "suffix256"]
tot = {n: 0.0 for n in names}; fwd = {n: 0 for n in names}; same = {n: 0 for n in names}; ntok = 0; vt = 0.0; nv = 0
w = tok(TEMPLATE.format(o="x = 1\n", u="x = 2\n"), return_tensors="pt").input_ids.to(dev)
fast_run(w, 8); fast_run(w, 8, cascade=False); suffix_run(w, 8, sc, "warm", 256, 4.0)
for i, (_, r) in enumerate(df.head(30).iterrows()):
    o_code, u = str(r["original_code"]), str(r["update_snippet"])
    ids = tok(TEMPLATE.format(o=o_code, u=u), return_tensors="pt").input_ids.to(dev)
    if ids.shape[1] > 6000:
        continue
    max_new = min(4096, len(tok(o_code).input_ids) + len(tok(u).input_ids) + 64)
    res = {}
    for n in names:
        st = {}; sync(); t0 = time.perf_counter()
        if n == "new":
            g = fast_run(ids, max_new, stats=st)
        elif n == "new_nocascade":
            g = fast_run(ids, max_new, stats=st, cascade=False)
        else:
            g = suffix_run(ids, max_new, sc, i, 256, 4.0, stats=st)
        sync(); dt = time.perf_counter() - t0
        res[n] = trim(g); tot[n] += dt; fwd[n] += st["fwd"]
    ref = res["suffix256"]
    if i < 5:
        sync(); t0 = time.perf_counter(); v = trim(vanilla(ids, max_new)); sync(); vt += time.perf_counter() - t0; nv += len(v); ref = v
        print(f"   vanilla check: new==vanilla {res['new'] == v}, suffix==vanilla {res['suffix256'] == v}", flush=True)
    ntok += len(ref)
    for n in names:
        same[n] += res[n] == ref
    print(f"[{i+1}/30] {len(ref)} tok " + " ".join(f"{n} {tot[n]:6.1f}s/{fwd[n]}fwd" for n in names) + "  same " + "/".join(str(int(res[n] == ref)) for n in names), flush=True)
print(f"TOTAL tokens {ntok}")
for n in names:
    print(f"  {n:14s} {tot[n]:7.1f} s  passes {fwd[n]:6d}  {ntok/tot[n]:6.0f} tok/s  identical {same[n]}/30")
print(f"NEW vs suffix256: {tot['suffix256']/tot['new']:.2f}x time, {fwd['suffix256']/fwd['new']:.2f}x fewer passes; no-cascade {tot['suffix256']/tot['new_nocascade']:.2f}x; vanilla first 5: {vt:.1f}s for {nv} tok")
print("CASCADE DONE", flush=True)
