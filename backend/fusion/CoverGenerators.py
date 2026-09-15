# -*- coding: utf-8 -*-
"""
Additional starting-cover generators for the route-comparison experiment
(branch exp/starting-cover-routes).

Both functions return pools in the same format as BranchFusion.anf_pool:
[(CompleteBranch, root=0, key)] whose flip functions XOR to f.

- minterm_entries: one dimension-0 block (an n-control point) per assignment in
  supp(f). No intelligence at all; intended as the absolute lower anchor and as an
  engine stress test at maximal pool size (~2^(n-1) blocks for random f).
- pk_entries: a pseudo-Kronecker-style expansion with per-term polarity. The
  recursion fixes the variable order and, at each variable, greedily chooses among
  the three expansion rules (Shannon / positive Davio / negative Davio) by one-step
  support-size lookahead, with the deterministic tie order S < pD < nD. No
  cross-term rewrites (in particular no exorlink) are performed, so the cover is
  left uncommitted for the fusion engine.
"""
from backend.CompleteBranch import CompleteBranch


def minterm_entries(fn, n):
    """One point block per assignment in supp(f)."""
    full = (1 << n) - 1
    return [(CompleteBranch(n, mask=full, bitstr=x), 0, x)
            for x in range(1 << n) if (fn >> x) & 1]


def pk_entries(fn, n):
    """Pseudo-Kronecker-style cover with per-term polarity (greedy S/pD/nD choice)."""
    entries = []

    def rec(g, level, mask, pol):
        # g is the subfunction over variables level..n-1, stored as a 2^(n-level)-bit
        # integer whose index bit 0 corresponds to variable `level` (bit s to the
        # remaining variables in order).
        rows = 1 << (n - level)
        if g == 0:
            return
        if g == (1 << rows) - 1:
            entries.append((CompleteBranch(n, mask=mask, bitstr=pol), 0, len(entries)))
            return
        # interleaved cofactor split: g0 = g restricted to x_level = 0, g1 = x_level = 1
        half = rows >> 1
        g0 = g1 = 0
        for s in range(half):
            if (g >> (2 * s)) & 1:
                g0 |= 1 << s
            if (g >> (2 * s + 1)) & 1:
                g1 |= 1 << s
        d = g0 ^ g1
        # one-step support-size lookahead; ties resolved S < pD < nD (deterministic)
        s_s = g0.bit_count() + g1.bit_count()
        s_p = g0.bit_count() + d.bit_count()
        s_n = g1.bit_count() + d.bit_count()
        bit = 1 << level
        if s_s <= s_p and s_s <= s_n:      # Shannon: g = x̄ g0 ⊕ x g1
            rec(g0, level + 1, mask | bit, pol)              # literal x = 0
            rec(g1, level + 1, mask | bit, pol | bit)        # literal x = 1
        elif s_p <= s_n:                   # positive Davio: g = g0 ⊕ x (g0 ⊕ g1)
            rec(g0, level + 1, mask, pol)
            rec(d, level + 1, mask | bit, pol | bit)
        else:                              # negative Davio: g = g1 ⊕ x̄ (g0 ⊕ g1)
            rec(g1, level + 1, mask, pol)
            rec(d, level + 1, mask | bit, pol)

    rec(fn, 0, 0, 0)
    return entries
