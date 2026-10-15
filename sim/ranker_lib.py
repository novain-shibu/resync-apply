import os as _os; _HERE = _os.path.dirname(_os.path.abspath(__file__)); _TRACE = _os.path.join(_HERE, "..", "traces", "oracle.json")
# Learned copy-pointer: rank candidate source positions by the expected copy length learned from training traces.
# Train on oracle records 100..198, test on 0..99 (contains the 30 edits with real SuffixDecoding numbers).
import json, sys, time
from collections import defaultdict
from drafter import Drafter

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else _TRACE)); R = d["records"]; aux = json.load(open(_os.path.join(_HERE, "vocab_aux.json")))
NL, UPD = set(aux["nl"]), aux["upd"]


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


def run(rec, cls, log=None, **kw):
    src, gt = rec["prompt"], rec["out"]
    dr = cls(src, NL, UPD, **kw); gen = [gt[0]]; pos = 1; passes = 1
    dr.start(gen)
    while pos < len(gt):
        br = dr.propose(gen, len(gt) - pos)
        if log is not None and not dr._self_draft:
            for p in dr.ptrs:
                a = 0
                while p + a < len(src) and pos + a < len(gt) and src[p + a] == gt[pos + a] and a < 512: a += 1
                log.append((dr.feats(p), a))
        best_a, best_i = 0, 0
        for i, b in enumerate(br):
            a = 0
            while a < len(b) and b[a] == gt[pos + a]: a += 1
            if a > best_a: best_a, best_i = a, i
        new = gt[pos:pos + best_a + 1]; gen += new; pos += len(new); passes += 1
        dr.update(gen, best_i, best_a, br)
    return passes


