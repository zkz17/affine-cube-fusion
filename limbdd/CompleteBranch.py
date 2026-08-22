# -*- coding: utf-8 -*-
"""
CompleteBranch：一个分支的完全信息（压缩表示）。

设计见 CompleteBranch_fusion_plan.md §7.8（2025-08 修订版）。

主数据（不可变；绝对变量位空间，bit k = 变量 k）：
  mask      : int   走边位置（MCX 控制变量）掩码；0 位 = fallthrough
  bitstr    : int   走边极性（仅 mask 位上有效；mask=0 处恒 0）
  x_global  : int   全部无条件 X 的净掩码（走边 Lim 的 XOR + 各 ft 位置 lim0∩lim1 上提部分）
  ft_lims   : tuple[(ctrl, m0, m1), ...]  自顶向下；m0&m1==0；m0/m1 的目标均位于 ctrl 下方
  synthetic : SyntheticPlan|None          融合产物（Phase 3）；None = 走线型分支

派生（惰性缓存）：
  AffineInfo(base, basis)：支撑集 S = base + span(basis)，basis 为 K 的规范基（RREF）
  fn                      ：2^n 位真值表

正确性依据（无近似）：
  1. 任何一层出边 Lim 只作用于"下方"变量 → 走边 Lim 可整体 XOR 进 x_global；
  2. [c=0→m0]⊕[c=1→m1] = (m0∩m1) ⊕ [c=0→m0∖m1] ⊕ [c=1→m1∖m0] → 公共部分上提 x_global；
  3. ft 项条件只依赖更高层变量 → 按 ft_lims 自顶向下发射即精确复现语义。
"""
from collections import namedtuple

from config import config
from circuit.Control import Control
from circuit.MCXCircuit import MCXCircuit
from limbdd.LimBDD import LimBDD, SimpleBranch
from limbdd.Node import LimBDDNode, LimBDDSignal


def t_cost(k):
    """k 控 MCX 的 T 代价（单一事实源：config.py 的 't_cost'，近线性新模型）。"""
    return config['t_cost'](k)


AffineInfo = namedtuple('AffineInfo', 'base basis')


def _bits(mask):
    """返回 mask 中置位比特的升序列表。"""
    out = []
    while mask:
        low = mask & (-mask)
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def abs_lim(lim, var_order, anomalies=None):
    """把图内 Lim（层位空间）换算为绝对变量位空间：bit k = 变量 k。"""
    m = 0
    for j in range(lim.bitnum):
        if (lim.bitstr >> j) & 1:
            if j >= len(var_order):
                if anomalies is not None:
                    anomalies['lim_overflow'] += 1
                continue
            m |= 1 << var_order[j]
    return m


def rref(rows, n):
    """
    把约束行 (mask, const) 化为 RREF（主元 = 最低置位，完全消元）。
    返回 [(mask, const, pivot_col), ...]（按主元列升序；行值为**最终** RREF 形，
    数学依据见 CompleteBranch_algebra.md §1.2/§3.6）。
    """
    row_masks = [m for m, _ in rows]
    row_consts = [c for _, c in rows]
    nrows = len(row_masks)
    pivot_cols = []
    r = 0
    for col in range(n):
        sel = None
        for i in range(r, nrows):
            if (row_masks[i] >> col) & 1:
                sel = i
                break
        if sel is None:
            continue
        row_masks[r], row_masks[sel] = row_masks[sel], row_masks[r]
        row_consts[r], row_consts[sel] = row_consts[sel], row_consts[r]
        for i in range(nrows):
            if i != r and (row_masks[i] >> col) & 1:
                row_masks[i] ^= row_masks[r]
                row_consts[i] ^= row_consts[r]
        pivot_cols.append(col)
        r += 1
    # 注意：必须先完成全部消元再取行值（后续主元会继续清除先前行的该列）
    return [(row_masks[i], row_consts[i], pivot_cols[i]) for i in range(r)]


def is_affine_cube(fn, n):
    """
    fn（2^n 位真值表）的支撑是否为仿射子空间。
    返回 (True, dim) 或 (False, 0)；空支撑（fn==0）返回 (True, -1)。
    """
    bits = [i for i in range(1 << n) if (fn >> i) & 1]
    if not bits:
        return True, -1
    pc = len(bits)
    if pc & (pc - 1):
        return False, 0
    base = bits[0]
    tset = {e ^ base for e in bits}
    tlist = sorted(tset)
    for i in range(len(tlist)):
        for j in range(i, len(tlist)):
            if tlist[i] ^ tlist[j] not in tset:
                return False, 0
    return True, len(tlist).bit_length() - 1


def reduce_basis(vectors):
    """把向量组归约为规范基：各向量最低置位互异，按主元（最低位）升序返回。"""
    by_bit = {}
    for v in vectors:
        x = v
        while x:
            low = x & (-x)
            if low in by_bit:
                x ^= by_bit[low]
            else:
                by_bit[low] = x
                break
    return tuple(by_bit[low] for low in sorted(by_bit))


def in_span(x, basis):
    """x 是否在 span(basis) 中（basis 需为 reduce_basis 产出的规范基：主元升序）。"""
    for b in basis:
        if x & (b & (-b)):
            x ^= b
    return x == 0


def affine_from_fn(fn, n):
    """fn 支撑的 (base, basis)：base = 最低支撑点，basis = reduce_basis 规范基。"""
    bits = [i for i in range(1 << n) if (fn >> i) & 1]
    if not bits:
        return AffineInfo(0, ())
    base = bits[0]
    return AffineInfo(base, reduce_basis(e ^ base for e in bits))


class SyntheticPlan:
    """融合段的发射计划：X(x_trans) → CNOT 网络 → MCX(controls) → CNOT 网络(逆) → X(x_trans)。"""

    __slots__ = ('x_trans', 'cnot_ops', 'controls')

    def __init__(self, x_trans=0, cnot_ops=(), controls=()):
        self.x_trans = x_trans
        self.cnot_ops = tuple(cnot_ops)
        self.controls = tuple(sorted(controls))

    def __eq__(self, other):
        return (isinstance(other, SyntheticPlan)
                and self.x_trans == other.x_trans
                and self.cnot_ops == other.cnot_ops
                and self.controls == other.controls)

    def __hash__(self):
        return hash((self.x_trans, self.cnot_ops, self.controls))

    def __repr__(self):
        return f'SyntheticPlan(x_trans={self.x_trans:b}, cnot_ops={self.cnot_ops}, controls={self.controls})'


def plan_from_affine(base, basis, n):
    """
    RREF 规范构造（docs/CompleteBranch_algebra.md §3.6）：
    仿射立方 e + span(basis) → SyntheticPlan(x_trans=0, cnot_ops, controls)。

    约束行由自由列索引：x_f ⊕ Σ_p (r_p)_f x_p = c_f；
    CNOT 网络控制 = 主元变量（不被修改）、目标 = 自由变量（次序无关）；
    常数吸收进 MCX 极性。

    本函数保持**规范形**（唯一、可哈希），供融合池幂等重入；发射时的 CX 最小化
    由 plan_min_cx 完成（T 不变，只换基）。
    """
    pivots = rref([(b, 0) for b in basis], n)
    pivot_cols = {col for _, _, col in pivots}
    free_cols = [f for f in range(n) if f not in pivot_cols]
    cnot_ops = []
    controls = []
    for f in free_cols:
        c = (base >> f) & 1
        for mask, _, col in pivots:
            if (mask >> f) & 1:
                c ^= (base >> col) & 1
                cnot_ops.append((col, f))
        controls.append((f, c))
    return SyntheticPlan(0, tuple(sorted(cnot_ops)), tuple(sorted(controls)))


def _plan_for_basis(base, basis, n):
    """给定基构造发射计划,返回 (SyntheticPlan, CNOT 数)。plan_from_affine 的构造本体。"""
    pivots = rref([(b, 0) for b in basis], n)
    pivot_cols = {col for _, _, col in pivots}
    free_cols = [f for f in range(n) if f not in pivot_cols]
    cnot_ops = []
    controls = []
    for f in free_cols:
        c = (base >> f) & 1
        for mask, _, col in pivots:
            if (mask >> f) & 1:
                c ^= (base >> col) & 1
                cnot_ops.append((col, f))
        controls.append((f, c))
    return SyntheticPlan(0, tuple(sorted(cnot_ops)), tuple(sorted(controls))), len(cnot_ops)


def plan_min_cx(base, basis, n):
    """
    B-4:CX 最优发射计划。

    同一仿射立方有多种基表示;在「规范基 + 所有两向量互加」的确定性候选集上
    取 CNOT 数最小的构造(并列时保持规范基优先 ⇒ 确定性)。控制数不变 ⇒ T 不变,
    仅 CNOT 网络规模下降。规范 plan(plan_from_affine)仍用于融合池的幂等/哈希。
    """
    candidates = [tuple(basis)]
    k = len(basis)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            alt = list(basis)
            alt[i] = alt[i] ^ alt[j]
            candidates.append(tuple(alt))
    best, best_cx = None, None
    for cand in candidates:
        plan, ncn = _plan_for_basis(base, cand, n)
        if best is None or ncn < best_cx:
            best, best_cx = plan, ncn
    return best


class CompleteBranch:
    __slots__ = ('n', 'mask', 'bitstr', 'x_global', 'ft_lims', 'source_branch', 'origin',
                 'synthetic', '_affine', '_fn', '_hash')

    def __init__(self, n, mask=0, bitstr=0, x_global=0, ft_lims=(),
                 source_branch=None, origin=0, synthetic=None):
        assert n >= 0
        assert bitstr & ~mask == 0, 'bitstr 只允许在 mask 位上置位'
        assert synthetic is None or (mask == 0 and bitstr == 0 and x_global == 0 and not ft_lims), \
            '合成段与走线字段互斥'
        for ctrl, m0, m1 in ft_lims:
            assert m0 & m1 == 0, 'ft 项需规范化：m0 & m1 == 0'
            assert not ((mask >> ctrl) & 1), 'ft 控制变量不得同时是走边位置'
            assert not ((m0 | m1) >> ctrl) & 1, 'ft 目标不得包含自身控制变量'

        self.n = n
        self.mask = mask
        self.bitstr = bitstr
        self.x_global = x_global
        self.ft_lims = tuple(ft_lims)
        self.source_branch = source_branch
        self.origin = origin
        self.synthetic = synthetic
        self._affine = None
        self._fn = None
        self._hash = None

    # ------------------------------------------------------------ 结构 ----

    def is_walk(self):
        return self.synthetic is None

    def nctrl(self):
        """MCX 控制数（走线型 = mask 位数；合成段 = controls 数）。"""
        if self.synthetic is not None:
            return len(self.synthetic.controls)
        return self.mask.bit_count()

    def controls(self):
        """MCX 控制集：sorted tuple[(var, polar)]。"""
        if self.synthetic is not None:
            return self.synthetic.controls
        return tuple((v, (self.bitstr >> v) & 1) for v in _bits(self.mask))

    def t_cost(self):
        return t_cost(self.nctrl())

    def is_constant_one(self):
        return self.nctrl() == 0

    # ------------------------------------------------------------ 派生 ----

    def affine(self):
        """支撑集 S = base + span(basis) 的符号化推导（O(n^2)，无需 2^n 枚举）。"""
        if self._affine is None:
            self._affine = self._compute_affine()
        return self._affine

    def fn(self):
        """分支函数（2^n 位真值表）。"""
        if self._fn is None:
            base, basis = self.affine()
            fn = 0
            for s in range(1 << len(basis)):
                k = 0
                for j in range(len(basis)):
                    if (s >> j) & 1:
                        k ^= basis[j]
                fn |= 1 << (base ^ k)
            self._fn = fn
        return self._fn

    def direction_key(self):
        """方向子空间 K 的规范基元组（融合候选分桶键）。"""
        return self.affine().basis

    def cancel_key(self):
        """函数真值表（相等 ⇒ 可抵消）。"""
        return self.fn()

    def _constraint_rows(self):
        """分支函数的约束行 (mask, const)（方程 mask·x ≡ const）。"""
        n = self.n
        masks = [1 << v for v in range(n)]
        consts = [0] * n

        if self.synthetic is not None:
            plan = self.synthetic
            for v in range(n):
                consts[v] = (plan.x_trans >> v) & 1
            for c, t in plan.cnot_ops:
                masks[t] ^= masks[c]
                consts[t] ^= consts[c]
            return [(masks[v], consts[v] ^ polar) for v, polar in plan.controls]

        for v in range(n):
            consts[v] = (self.x_global >> v) & 1
        for ctrl, m0, m1 in self.ft_lims:
            cmask, cconst = masks[ctrl], consts[ctrl]
            for t in _bits(m0):
                masks[t] ^= cmask
                consts[t] ^= cconst ^ 1
            for t in _bits(m1):
                masks[t] ^= cmask
                consts[t] ^= cconst
        return [(masks[v], consts[v] ^ ((self.bitstr >> v) & 1)) for v in _bits(self.mask)]

    def affine_rows(self):
        """约束行的 RREF 规范形：tuple[(pivot, free_mask, const)]（主元升序）。"""
        pivots = rref(self._constraint_rows(), self.n)
        return tuple((col, mask ^ (1 << col), const) for mask, const, col in pivots)

    def _compute_affine(self):
        pivots = rref(self._constraint_rows(), self.n)
        pivot_cols = {col for _, _, col in pivots}

        basis = []
        for f in range(self.n):
            if f in pivot_cols:
                continue
            v = 1 << f
            for mask, _, col in pivots:
                if (mask >> f) & 1:
                    v |= 1 << col
            basis.append(v)

        base = 0
        for _, const, col in pivots:
            if const:
                base |= 1 << col
        return AffineInfo(base, tuple(basis))

    # ------------------------------------------------------------ 相等 ----

    def _key(self):
        return (self.n, self.mask, self.bitstr, self.x_global, self.ft_lims, self.synthetic)

    def __eq__(self, other):
        return isinstance(other, CompleteBranch) and self._key() == other._key()

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self._key())
        return self._hash

    def __repr__(self):
        if self.synthetic is not None:
            return f'CompleteBranch(n={self.n}, synthetic={self.synthetic})'
        ft = ','.join(f'({c},{m0:b},{m1:b})' for c, m0, m1 in self.ft_lims)
        return (f'CompleteBranch(n={self.n}, mask={self.mask:b}, bitstr={self.bitstr:b}, '
                f'x_global={self.x_global:b}, ft_lims=[{ft}])')

    # ------------------------------------------------------------ 发射 ----

    def to_circuit(self, qubit_mgr, target=None):
        """
        发射为 MCXCircuit（Phase 4 的接入点；此处用于与 compute_tt 对账）。
        调用方需已分配全部 n 个输入量子比特。
        """
        circ = MCXCircuit(qubit_mgr, target)
        if self.synthetic is not None:
            # B-4:发射用 CX 最优基(存储 plan 保持 RREF 规范形,控制数不变 ⇒ T 不变)
            plan = plan_min_cx(*self.affine(), self.n)
            x_qubits = [qubit_mgr.get_input_qubit(v) for v in _bits(plan.x_trans)]
            circ.add_lim_operator(x_qubits)
            for c, t in plan.cnot_ops:
                circ.add_lim_operator([qubit_mgr.get_input_qubit(t)],
                                      Control(qubit_mgr.get_input_qubit(c), 1))
            circ.add_gate([Control(qubit_mgr.get_input_qubit(v), p) for v, p in plan.controls],
                          circ.target)
            for c, t in reversed(plan.cnot_ops):
                circ.add_lim_operator([qubit_mgr.get_input_qubit(t)],
                                      Control(qubit_mgr.get_input_qubit(c), 1))
            circ.add_lim_operator(x_qubits)
            return circ

        x_qubits = [qubit_mgr.get_input_qubit(v) for v in _bits(self.x_global)]
        circ.add_lim_operator(x_qubits)
        for ctrl, m0, m1 in self.ft_lims:
            c = qubit_mgr.get_input_qubit(ctrl)
            circ.add_lim_operator([qubit_mgr.get_input_qubit(t) for t in _bits(m0)],
                                  Control(c, 0))
            circ.add_lim_operator([qubit_mgr.get_input_qubit(t) for t in _bits(m1)],
                                  Control(c, 1))
        circ.add_gate([Control(qubit_mgr.get_input_qubit(v), p) for v, p in self.controls()],
                      circ.target)
        for ctrl, m0, m1 in reversed(self.ft_lims):
            c = qubit_mgr.get_input_qubit(ctrl)
            circ.add_lim_operator([qubit_mgr.get_input_qubit(t) for t in _bits(m0)],
                                  Control(c, 0))
            circ.add_lim_operator([qubit_mgr.get_input_qubit(t) for t in _bits(m1)],
                                  Control(c, 1))
        circ.add_lim_operator(x_qubits)
        return circ


# ---------------------------------------------------------------- 抽取 ----

def extract_complete_branches(limbdd, n, origin=0, anomalies=None):
    """
    与 branch2circ 走法逐层镜像的压缩抽取（见 CompleteBranch_fusion_plan.md §7.8）。

    对每个"根 → 终结结点"的路径生成一个 CompleteBranch：
      - 走边位置（含 one-child 与分叉结点的所走边）→ mask 置位 + bitstr 记极性，
        所走边 Lim 异或进 x_global；
      - fallthrough 位置 → 两出边 Lim 规范化（公共部分上提 x_global）后按序追加 ft_lims；
      - w0=w1=0 结点不应出现（记异常）；neg 边不应出现（记异常）；
      - ft 目标必须位于其控制变量下方（记异常，测试会兜底）。
    """
    assert isinstance(limbdd, LimBDD)
    assert n >= 0
    assert not limbdd.root.neg, 'split_neg 后根边不应有 neg'

    top_down = list(reversed(limbdd.var_order))
    pos_of = {v: i for i, v in enumerate(top_down)}
    terminator = limbdd.terminator
    results = []

    def walk(node, mask, bitstr, x_global, ft_lims, bits, bitnum):
        if node is terminator:
            results.append(CompleteBranch(n, mask, bitstr, x_global, ft_lims,
                                          source_branch=SimpleBranch(bits, bitnum),
                                          origin=origin))
            return
        if node.level == 0:
            if anomalies is not None:
                anomalies['zero_level'] += 1
            return
        w0, w1 = node.low.weight, node.high.weight
        if w0 == 0 and w1 == 0:
            if anomalies is not None:
                anomalies['zero_node'] += 1
            return
        if node.low.neg or node.high.neg:
            if anomalies is not None:
                anomalies['neg_edge'] += 1

        var = node.var_index
        if node.low.node is node.high.node and w0 == 1 and w1 == 1:
            # fallthrough：两出边 Lim 规范化（公共部分上提），项自顶向下追加。
            m0 = abs_lim(node.low.lim, limbdd.var_order, anomalies)
            m1 = abs_lim(node.high.lim, limbdd.var_order, anomalies)
            common = m0 & m1
            x_global ^= common
            m0 ^= common
            m1 ^= common
            if m0 or m1:
                for t in _bits(m0 | m1):
                    if pos_of.get(t, -1) <= pos_of[var]:
                        if anomalies is not None:
                            anomalies['ft_target_above'] += 1
                ft_lims = ft_lims + ((var, m0, m1),)
            walk(node.low.node, mask, bitstr, x_global, ft_lims, bits, bitnum)
        else:
            for b, succ in ((0, node.low), (1, node.high)):
                if succ.weight:
                    walk(succ.node,
                         mask | (1 << var),
                         bitstr | (b << var),
                         x_global ^ abs_lim(succ.lim, limbdd.var_order, anomalies),
                         ft_lims,
                         bits | (b << bitnum), bitnum + 1)

    walk(limbdd.root.node, 0, 0, 0, (), 0, 0)
    return results
