import os as _os; _HERE = _os.path.dirname(_os.path.abspath(__file__)); _TRACE = _os.path.join(_HERE, "..", "traces", "oracle.json")
# Ceiling for ANY copy-based drafter: at each pass, propose the source span that matches the true continuation longest (cheating oracle).
import json, sys
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else _TRACE)); R = d["records"]
def longest_from(src_index, src, gt, pos, maxlen=4096):
    """longest run of gt[pos:] found anywhere in src, using a 4-gram index for candidates"""
    best = 0
    key = tuple(gt[pos:pos + 4])
    for start in src_index.get(key, []):
        a = 0
        while a < maxlen and pos + a < len(gt) and start + a < len(src) and src[start + a] == gt[pos + a]: a += 1
        if a > best: best = a
    if best == 0:   # shorter keys
        for n in (3, 2, 1):
            for start in src_index.get(("n", n, tuple(gt[pos:pos + n])), []):
                a = 0
                while pos + a < len(gt) and start + a < len(src) and src[start + a] == gt[pos + a]: a += 1
                best = max(best, a)
            if best: break
    return best
P = 0; P30 = 0; novel = 0
for i, r in enumerate(R):
    src, gt = r["prompt"], r["out"]
    idx = {}
    for j in range(len(src) - 3): idx.setdefault(tuple(src[j:j + 4]), []).append(j)
    for n in (3, 2, 1):
        for j in range(len(src) - n + 1): idx.setdefault(("n", n, tuple(src[j:j + n])), []).append(j)
    # also allow copying from own earlier output (self-lookup) - append as extra source
    pos = 1; passes = 1
    while pos < len(gt):
        a = longest_from(idx, src, gt, pos)
        own = gt[:pos]
        if a == 0: novel += 1
        pos += a + 1; passes += 1
    P += passes
    if "suffix_fwd" in r: P30 += passes
print(f"copy-oracle ceiling (knows the future): all 199 edits {P} passes, first 30 edits {P30} passes; passes with zero copy {novel}")
