# -*- coding: utf-8 -*-
"""
方案 A:FPRM 免融合代理预筛(探针 P1 已达标,top-8 保留率 71.7%~84.6%)。

流程:
  全枚举 2^n 个平移,每个只做 Möbius(不融合),算代理分
    S0 = Σ t(n−|S|)(融合前精确 ΣT)
    P  = 差一变量对计数(C3 必然可融对密度)
    S1 = S0 − P(越小越好)
  取 top-k 个(排序 (S1, c) 升序,确定性)并恒含 c=0,仅对这 k+1 个候选做完整融合。

设计定位(选择关系,非 min):fprm_proxy=True 时**取代**默认 F 路径
  - n ≥ 9:代理全枚举 + top-k 融合,意图升级现行平移 climb2(更全的枚举口径);
  - n ≤ 8:默认 F 是全枚举完整融合,代理版质量低 12%~28%(P1 实测)——选择关系
    允许按需切换,不做 min 以免拖慢测试并掩盖差异(用户指示,探索期纪律)。
恒含 c=0 ⇒ 代理结果 ≤ C 路径(支配链保持)。
"""
from config import config
from limbdd.fusion.BranchFusion import anf_pool, fuse_pool_global, pool_t_cost, shift_fn


def proxy_scores(fn, n):
    """全部 2^n 个平移的代理分,返回 [(S1, c, S0, P)]。"""
    out = []
    for c in range(1 << n):
        entries = anf_pool(shift_fn(fn, c, n), n)
        masks = [cb.mask for cb, _, _ in entries]
        mset = set(masks)
        s0 = sum(config['t_cost'](n - m.bit_count()) for m in masks)
        p = 0
        for m in masks:
            for j in range(n):
                if not (m >> j) & 1 and (m | (1 << j)) in mset:
                    p += 1
        out.append((s0 - p, c, s0, p))
    return out


def fprm_proxy_best(fn, n, k=8):
    """
    top-k 代理候选(恒含 c=0)的完整融合,返回 (kept, standalone, best_c)。

    确定性:候选按 (S1, c) 升序取前 k;完整融合结果按评估顺序比较、并列保留先者。
    """
    rows = sorted(proxy_scores(fn, n), key=lambda r: (r[0], r[1]))
    cands = [c for _, c, _, _ in rows[:k]]
    if 0 not in cands:
        cands.append(0)
    best = None
    for c in cands:
        kept, standalone = fuse_pool_global(anf_pool(shift_fn(fn, c, n), n), n)
        v = pool_t_cost(kept, standalone)
        if best is None or v < best[0]:
            best = (v, kept, standalone, c)
    return best[1], best[2], best[3]
