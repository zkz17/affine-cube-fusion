# -*- coding: utf-8 -*-
"""
G-2.2: deterministic GL(n,2) local search (an **optional** switch on the production path, off by default).

Design (finalized by gl_det_probe measurements, de-randomized):
  - baseline: FS-selected shift cF (including c=0; n≤8 full enumeration, n≥9 two-start shift climbing);
  - start set: {identity on f, identity on f∘τ_cF} ∪ {all single-CNOT E_ij on f∘τ_cF},
    totaling 2 + n(n−1) starts, **fully deterministic**;
  - climbing: steepest ascent in the CNOT neighborhood, strict improvement, step cap max_steps;
  - candidates include cF itself (zero wrapper); on T ties the zero wrapper is preferred, so the result
    has pairwise-fused T cost ≤ FS (the chain FG ≤ FS ≤ F holds before macro-M2 post-processing).
    Selecting the zero wrapper on a T tie avoids additional wrapper CNOTs; this is not a
    guarantee that the final emitted CX count is minimal.

Cost composition (measured at n=7): ~2400 evaluations × 2.55ms, of which **94% is fuse_pool_global**
(every coordinate change requires a fresh Möbius + fusion, no reuse) -- this is why the switch is off
by default.

Returns (kept, standalone, shift_c, cnots): the winning pool is the pool of g(x) = f(Ax ⊕ c);
emission sandwich = X(c) · CNOT(A⁻¹) · [segments] · CNOT(A) · X(c) (T-free, CNOTs counted in CX).
"""
from backend.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn


def apply_linear(fn, A, n):
    """g(x) = f(Ax). A is given by row masks: (Ax)_i = parity(A[i] & x)."""
    out = 0
    for x in range(1 << n):
        y = 0
        for i, r in enumerate(A):
            if (r & x).bit_count() & 1:
                y |= 1 << i
        if (fn >> y) & 1:
            out |= 1 << x
    return out


def compose_transvection_right(A, c, t, n):
    """A ∘ E, E = CNOT(c → t) acting on the input: row i XORs bit c when bit t is set."""
    out = A[:]
    for i in range(n):
        if (out[i] >> t) & 1:
            out[i] ^= 1 << c
    return out


def make_cost(fn, n, cache):
    def cost(A):
        key = tuple(A)
        if key not in cache:
            kept, standalone = fuse_pool_global(anf_pool(apply_linear(fn, A, n), n), n)
            cache[key] = pool_t_cost(kept, standalone)
        return cache[key]
    return cost


def climb_best(A0, cnots0, cost, n, max_steps):
    """Steepest ascent in the CNOT neighborhood (strict improvement); returns (A, cnots, value)."""
    A, cnots = A0[:], list(cnots0)
    cur = cost(tuple(A))
    while True:
        best, best_A, best_move = cur, None, None
        for c in range(n):
            for t in range(n):
                if c == t:
                    continue
                A2 = compose_transvection_right(A, c, t, n)
                v = cost(tuple(A2))
                if v < best:
                    best, best_A, best_move = v, A2, (c, t)
        if best_A is None:
            break
        A = best_A
        cnots.append(best_move)
        cur = best
        if max_steps and len(cnots) - len(cnots0) >= max_steps:
            break
    return A, cnots, cur


def fprm_best(fn, n):
    """Select a shift using production's exhaustive/climb2 budget policy. Return (cost, c)."""
    full = 1 << n
    if full <= 256:
        shifts = range(full)
        best, best_c = None, 0
        for c in shifts:
            g = shift_fn(fn, c, n)
            kept, standalone = fuse_pool_global(anf_pool(g, n), n)
            v = pool_t_cost(kept, standalone)
            if best is None or v < best:
                best, best_c = v, c
        return best, best_c
    # n ≥ 9: two-start shift climbing (identical to production _fprm_entries, deterministic)
    cache = {}

    def eval_c(c):
        if c not in cache:
            g = shift_fn(fn, c, n)
            kept, standalone = fuse_pool_global(anf_pool(g, n), n)
            cache[c] = pool_t_cost(kept, standalone)
        return cache[c]

    for start in (0, full - 1):
        cur = start
        while True:
            b, bc = eval_c(cur), None
            for j in range(n):
                nb = cur ^ (1 << j)
                v = eval_c(nb)
                if v < b:
                    b, bc = v, nb
            if bc is None:
                break
            cur = bc
    best_c = min(cache, key=lambda c: cache[c])
    return cache[best_c], best_c


def gl_search(fn, n, max_steps=3):
    """
    Deterministic GL search (called when the optional switch is active).

    Returns (kept, standalone, shift_c, cnots):
      shift_c : the winning candidate's shift (FPRM baseline);
      cnots   : the CNOT decomposition of the winning linear map A (generation order,
                A = E1∘…∘Ek, so A⁻¹ applies cnots in forward order and A in reverse).
    """
    vF, cF = fprm_best(fn, n)
    ident = [1 << i for i in range(n)]
    best = (vF, cF, ident, [])          # (cost, shift_c, A, cnots)

    def consider(c, A, cnots, v):
        nonlocal best
        if v < best[0] or (v == best[0] and len(cnots) < len(best[3])):
            best = (v, c, list(A), list(cnots))

    cache = {}
    cost = make_cost(fn, n, cache)
    A, cnots, v = climb_best(ident, [], cost, n, max_steps)
    consider(0, A, cnots, v)

    fn_pre = shift_fn(fn, cF, n)
    cache2 = {}
    cost2 = make_cost(fn_pre, n, cache2)
    A, cnots, v = climb_best(ident, [], cost2, n, max_steps)
    consider(cF, A, cnots, v)
    for c in range(n):
        for t in range(n):
            if c == t:
                continue
            A0 = compose_transvection_right(ident, c, t, n)
            A, cnots, v = climb_best(A0, [(c, t)], cost2, n, max_steps)
            consider(cF, A, cnots, v)

    _, shift_c, A, cnots = best
    g = apply_linear(shift_fn(fn, shift_c, n) if shift_c else fn, A, n)
    kept, standalone = fuse_pool_global(anf_pool(g, n), n)
    return kept, standalone, shift_c, cnots
