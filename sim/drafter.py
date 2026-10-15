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
