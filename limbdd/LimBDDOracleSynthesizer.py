from utils.Qubit import QubitManager
from utils.TruthTable import TruthTable
from limbdd.Node import LimBDDNode, LimBDDSignal
from limbdd.LimBDD import LimBDD, SimpleBranch
from limbdd.LimBDDManager import LimBDDManager
from limbdd.fusion.FusedBackend import limbdd2circ_fused
from limbdd.CompleteBranch import abs_lim, extract_complete_branches, _bits
from limbdd.fusion.BranchFusion import fuse_pool_global, pool_t_cost, anf_pool, shift_fn
from limbdd.fusion.GLSearch import gl_search
from limbdd.fusion.FprmProxy import fprm_proxy_best
from circuit.Control import Control
from circuit.MCXCircuit import MCXCircuit, order_segments_greedy
from utils.to_qiskit import to_qiskit
from config import config

class LimBDDOracleSynthesizer:
    def __init__(self, reorder=True, ancilla_num=1, fused_backend=True, gl_search=False,
                 fprm_proxy=False, paths=('F',)):
        self.limbdd = None
        self.limbdd_mgr = LimBDDManager()
        self.qubit_mgr = QubitManager()
        self.circ = MCXCircuit(self.qubit_mgr)
        self.nidx2circ = {}             # { LimBDDNode.idx : [ MCXCircuit, int ] }
        self.reorder = reorder
        self.ancilla_num = ancilla_num
        # Phase 5 起：融合发射后端为 no-ancilla 路径的默认后端（ancilla_num>0 时静默忽略）
        self.fused_backend = fused_backend
        # G-2.2 可选开关（默认关闭）：确定性 GL(n,2) 局部搜索,时间换质量
        # (n=7 约 +6s/例,94% 是融合引擎本身——见 Optimization_roadmap.md G-2.2 节)
        self.gl_search = gl_search
        # 方案 A 可选开关（默认关闭）：FPRM 免融合代理预筛 **取代** F 路径
        # (选择关系而非 min;n≥9 意图升级现行平移 climb2,n≤8 质量低 12~28%)
        self.fprm_proxy = fprm_proxy
        # 路径选择(2025-08-22,探索期纪律):生产后端**暂时只跑 F 路径**;
        # 之后按每条路径的优化效果/运行时间逐路径评测,再定最终组合。
        # paths 为 {'A','C','F'} 的子集,取其中最优者;gl_search=True 时改用
        # G 路径(min(A,G)),忽略 paths。
        assert set(paths) <= {'A', 'C', 'F'} and len(paths) >= 1, f'非法 paths: {paths}'
        self.paths = tuple(paths)

    def clear(self):
        self.limbdd = None
        self.limbdd_mgr = LimBDDManager()
        self.qubit_mgr.clear()
        self.nidx2circ.clear()
        self.circ.clear()
        self.circ = MCXCircuit(self.qubit_mgr)

    def construct(self, tt):
        var_order = self.determine_firsttime_order(tt.copy())
        self.limbdd_mgr.initialize(tt, var_order)
        self.allocate_input_qubit(tt.num_var)

    def synthesize(self, tt, path=None):
        assert(isinstance(tt, TruthTable))

        self.clear()
        if self.fused_backend and self.ancilla_num == 0:
            self.process_fused_global(tt, path)   # Phase 5+：min(BDD, ANF) 全局融合后端
            return

        self.construct(tt)
        if path: self.show(self.limbdd_mgr.limbdd, path, title=f'{tt.num_var} input function {tt.bitstr}\n{str(tt)}')

        self.process(path)

    def process(self, path=None):
        self.limbdd_mgr.try_neg(self.limbdd_mgr.limbdd)
        self.limbdd_mgr.split_neg(self.limbdd_mgr.limbdd)

        if self.reorder:
            self.variable_reorder()
            if path: self.show(self.limbdd_mgr.limbdd, path + '_reorder')
            #self.limbdd_mgr.split_neg(self.limbdd_mgr.limbdd)
        
        for limbdd_list in self.limbdd_mgr.limbdd_list.values():
            for limbdd in limbdd_list:
                self.circ.merge(self.limbdd2circ(limbdd, self.ancilla_num, collect_br=False))
            limbdd_list.clear()
        
        self.circ.merge(self.limbdd2circ(self.limbdd_mgr.limbdd, self.ancilla_num))

        if self.limbdd_mgr.limbdd.root.level in self.limbdd_mgr.limbdd_list:
            while len(self.limbdd_mgr.limbdd_list[self.limbdd_mgr.limbdd.root.level]):
                limbdd = self.limbdd_mgr.limbdd_list[self.limbdd_mgr.limbdd.root.level].pop()
                self.circ.merge(self.limbdd2circ(limbdd, self.ancilla_num))
                #limbdd = self.limbdd_mgr.fuse_level(tt.num_var)
                #if limbdd: self.circ.merge(self.limbdd2circ(limbdd, self.ancilla_num))

    def _collect_pieces_fused(self, path=None):
        """
        Phase 5 融合路径的图阶段（与旧 process() 的图阶段逐行同构）：
        try_neg / split_neg / reorder / 分片与吸收时序，返回处理顺序的非零子图列表。
        不生成电路；供 process_fused_global 与残差测量脚本复用。
        """
        mgr = self.limbdd_mgr
        mgr.try_neg(mgr.limbdd)
        mgr.split_neg(mgr.limbdd)

        if self.reorder:
            self.variable_reorder()
            if path: self.show(mgr.limbdd, path + '_reorder')

        pieces = []  # 处理顺序（吸收/拆分后仍非零的子图）
        for limbdd_list in mgr.limbdd_list.values():
            for limbdd in limbdd_list:
                if limbdd.root.weight:
                    mgr.perform_split(limbdd)
                mgr.append_temp_to_list()
                if limbdd.root.weight:
                    pieces.append(limbdd)
            limbdd_list.clear()

        main = mgr.limbdd
        if main.root.weight:
            mgr.collect_branches(main, main.root.level)
            mgr.append_temp_to_list()
            if main.root.weight:
                pieces.append(main)

        if main.root.level in mgr.limbdd_list:
            while len(mgr.limbdd_list[main.root.level]):
                limbdd = mgr.limbdd_list[main.root.level].pop()
                if limbdd.root.weight:
                    mgr.collect_branches(limbdd, limbdd.root.level)
                    mgr.append_temp_to_list()
                    if limbdd.root.weight:
                        pieces.append(limbdd)
        return pieces

    def _bdd_entries(self, tt, path):
        """一次完整的 BDD 分解（construct + 图阶段 + sifting），返回全局条目列表。
        construct 会重建 limbdd_mgr，可安全连续调用多次。"""
        self.construct(tt)
        pieces = self._collect_pieces_fused(path)
        n = tt.num_var
        entries = []
        for idx, piece in enumerate(pieces):
            r = abs_lim(piece.root.lim, piece.var_order)
            for cb in extract_complete_branches(piece, n):
                entries.append((cb, r, idx))
        return entries

    def process_fused_global(self, tt, path=None):
        """
        Phase 5+：全局融合后端（默认 no-ancilla 路径）。

        路径选择(探索期纪律,2025-08-22):生产后端**暂时只跑 F 路径**(paths 默认
        ('F',));paths 为 {'A','C','F'} 的任意子集,仅计算所选路径并取其中最优。
          A. BDD 路径：图阶段（try_neg / split_neg / reorder / 拆分,absorb 已移除）
             → 分支池 → 全局融合；
          C. ANF 路径：真值表 → Möbius → 单项式池（c=0）；
          F. FPRM 路径（G-2.1）：平移子群上搜索最优 c（n≤8 全枚举 2^n；
             n≥9 climb2 双起点,确定性）；电路 = X(c) 三明治（T 免费）。
        fprm_proxy=True 时 F 由免融合代理预筛版取代（选择关系,非 min）。
        并列优先级按 paths 中的给定顺序(默认 F 唯一,无并列问题)。

        可选（gl_search=True，G-2.2）：忽略 paths,改用 G 路径(min(A,G))——
        确定性 GL(n,2) 局部搜索（候选含 FPRM 最优 cF ⇒ G ≤ F ≤ C 支配链保持、
        零回归;T 并列优先零 wrapper）。代价:n=7 约 +6s/例。

        注（2025-08-21）：C′ 路径（ANF 差一变量预合并,1.2/T-3）经实测后**暂时关闭**
        （端到端收益 −0.22%~−0.53% ΣT、代价 +23%~36% wall,处于决策临界区）;
        `BranchFusion.anf_pool_premerged` 与 `tests/test_premerge.py` 保留,随时可重新接入。

        发射：按根掩码分组（每个 r 一个三明治 X(r) … X(r)），融合段（r=0）独立
        发射；F/FP 获胜时整体包 X(c) 三明治;G 获胜时包 CNOT(A⁻¹)/CNOT(A) 线性
        三明治;末尾 simplify_terms。
        """
        n = tt.num_var

        if self.gl_search:
            # ---- 路径 G：确定性 GL(n,2) 局部搜索(可选,时间换质量)----
            kept_a, standalone_a = fuse_pool_global(self._bdd_entries(tt, path), n)
            kept_g, standalone_g, shift_c, cnots = gl_search(tt.bitstr, n)
            candidates = [
                (pool_t_cost(kept_a, standalone_a), kept_a, standalone_a, 0, ()),
                (pool_t_cost(kept_g, standalone_g), kept_g, standalone_g, shift_c, tuple(cnots)),
            ]
        else:
            candidates = []
            if 'A' in self.paths:
                kept_a, standalone_a = fuse_pool_global(self._bdd_entries(tt, path), n)
                candidates.append((pool_t_cost(kept_a, standalone_a), kept_a, standalone_a, 0, ()))
            if 'C' in self.paths:
                entries_c = anf_pool(tt.bitstr, n)
                kept_c, standalone_c = fuse_pool_global(entries_c, n)
                candidates.append((pool_t_cost(kept_c, standalone_c), kept_c, standalone_c, 0, ()))
            if 'F' in self.paths:
                # 方案 A(fprm_proxy=True)时用代理预筛版**取代**默认 F(选择关系,非 min)
                if self.fprm_proxy:
                    kept_f, standalone_f, shift_c = fprm_proxy_best(tt.bitstr, n)
                else:
                    kept_f, standalone_f, shift_c = self._fprm_entries(tt, n)
                candidates.append((pool_t_cost(kept_f, standalone_f), kept_f, standalone_f, shift_c, ()))
            assert candidates, 'paths 为空(构造时已校验,不应到达)'
        best_cost = min(c[0] for c in candidates)
        _, kept, standalone, shift_c, cnots = next(c for c in candidates if c[0] == best_cost)

        # ---- 发射：平移三明治 X(c)(shift_c=0 时 no-op)----
        shift_qubits = [self.qubit_mgr.get_input_qubit(v) for v in _bits(shift_c)]
        self.circ.add_lim_operator(shift_qubits)

        # ---- GL 获胜时的线性三明治:CNOT(A⁻¹) = cnots 正序(A = E1∘…∘Ek)----
        for c_ctrl, t_tgt in cnots:
            self.circ.add_lim_operator([self.qubit_mgr.get_input_qubit(t_tgt)],
                                       Control(self.qubit_mgr.get_input_qubit(c_ctrl), 1))

        kept_by_r = {}
        for cb, r, idx in kept:
            kept_by_r.setdefault(r, []).append(cb)
        for r, segs in sorted(kept_by_r.items()):
            r_qubits = [self.qubit_mgr.get_input_qubit(v) for v in _bits(r)]
            self.circ.add_lim_operator(r_qubits)
            # B-4:段顺序只影响 CX(段集合不变 ⇒ T 不变);
            # 评估走真实 merge 管线(qubit_map 重映射),前缀为当前未化简电路。
            for seg_circ in order_segments_greedy([cb.to_circuit(self.qubit_mgr) for cb in segs],
                                                  self.circ if not self.circ.empty() else None):
                self.circ.merge(seg_circ)
            self.circ.add_lim_operator(r_qubits)
        for seg_circ in order_segments_greedy([cb.to_circuit(self.qubit_mgr) for cb in standalone],
                                              self.circ if not self.circ.empty() else None):
            self.circ.merge(seg_circ)

        # ---- GL 获胜时的线性三明治:CNOT(A) = cnots 逆序 ----
        for c_ctrl, t_tgt in reversed(cnots):
            self.circ.add_lim_operator([self.qubit_mgr.get_input_qubit(t_tgt)],
                                       Control(self.qubit_mgr.get_input_qubit(c_ctrl), 1))

        self.circ.add_lim_operator(shift_qubits)
        self.circ.simplify_terms()

    def _fprm_entries(self, tt, n):
        """
        G-2.1：平移子群上搜索最优 ANF 池。

        返回 (kept, standalone, best_c)。预算策略（阶段 3.1-C 决策）：
          n ≤ 8：全枚举 2^n（廉价，1.9s/例 以下）；
          n ≥ 9：climb2 —— 从 c=0 与 c=2^n−1 双起点做最陡上升（Hamming 邻域、
                 严格改进、缓存复用，确定性），实测兑现全枚举收益的 85%~100%，
                 优于原 rand64 采样（72%~81%）。
        已记录未实现的可选项（见 docs/Optimization_roadmap.md）：
          ① rand64 采样点作多起点爬山的混合方案；② evaluate.py 的全枚举离线开关。
        """
        full = 1 << n
        cache = {}

        def eval_c(c):
            if c not in cache:
                g = shift_fn(tt.bitstr, c, n)
                kept, standalone = fuse_pool_global(anf_pool(g, n), n)
                cache[c] = (pool_t_cost(kept, standalone), kept, standalone)
            return cache[c]

        if full <= 256:
            for c in range(full):
                eval_c(c)
        else:
            for start in (0, full - 1):
                cur = start
                while True:
                    best, best_c = eval_c(cur)[0], None
                    for j in range(n):
                        nb = cur ^ (1 << j)
                        v = eval_c(nb)[0]
                        if v < best:
                            best, best_c = v, nb
                    if best_c is None:
                        break
                    cur = best_c

        best_c = min(cache, key=lambda c: cache[c][0])
        _, kept, standalone = cache[best_c]
        return kept, standalone, best_c

    def limbdd2circ(self, limbdd, anc_num, collect_br=True):
        assert(isinstance(limbdd, LimBDD))

        if not limbdd.root.weight: return MCXCircuit()
        '''
        self.show(limbdd, path=f'figure/debug/fig_{config['debug_figure_counter']}')
        config['debug_figure_counter'] += 1
        '''
        if collect_br: self.limbdd_mgr.collect_branches(limbdd, limbdd.root.level)
        else: self.limbdd_mgr.perform_split(limbdd)
        self.limbdd_mgr.append_temp_to_list()
        if not limbdd.root.weight: return MCXCircuit()

        if self.fused_backend and anc_num == 0:
            # Phase 4：CompleteBranch 融合发射后端（抽取 → 融合池 → 逐段发射 + 根三明治）
            n = self.qubit_mgr.get_input_num()
            return limbdd2circ_fused(limbdd, self.qubit_mgr, n)

        assert(not len(self.nidx2circ))
        self.limbdd = limbdd

        o_qubit_list, _ = self.get_o_qubit_list(limbdd.root)

        circ = MCXCircuit(self.qubit_mgr)
        circ.add_lim_operator(o_qubit_list)
        if limbdd.root.level <= 3:
            circ.merge(self.generate_circ_wo_anc(limbdd.root.node))
        else:
            branch_list_map = LimBDDManager.branch_process(limbdd.root.node, [limbdd.terminator])
            if len(branch_list_map):
                branch_list = branch_list_map[limbdd.terminator]
                circ.merge(self.generate_circ_with_anc(limbdd.root.node, branch_list, anc_num=anc_num, control_list=[]))
            else:
                circ.merge(self.generate_circ_wo_anc(limbdd.root.node))
        circ.add_lim_operator(o_qubit_list)

        self.nidx2circ.clear()
        self.limbdd = None
        return circ

    def determine_firsttime_order(self, tt):
        assert(isinstance(tt, TruthTable))

        var_branch_distance = {}
        for var_index in range(tt.num_var): var_branch_distance[var_index] = 0

        self.compute_branch_distance(tt, var_branch_distance)
        return sorted(var_branch_distance, key=lambda k: var_branch_distance[k], reverse=True)

    def compute_branch_distance(self, tt, var_branch_distance):
        assert(isinstance(tt, TruthTable) and isinstance(var_branch_distance, dict))

        if tt.num_var == 0: return
        if tt.num_var > 1: tt.try_neg()
        tt0, tt1 = tt.split(tt.num_var - 1)
        var_branch_distance[tt0.num_var] += TruthTable.distence(tt0, tt1)

        self.compute_branch_distance(tt0, var_branch_distance)
        self.compute_branch_distance(tt1, var_branch_distance)

    def variable_reorder(self):
        limbdd = self.limbdd_mgr.limbdd

        # Skip reordering for tiny diagrams: no branch structure to improve.
        if limbdd.root.level <= 2 or len(limbdd.node_table) <= limbdd.root.level + 1: return

        top_level = limbdd.root.level
        while top_level > 1 and len(limbdd.level2nodes[top_level - 1]) == 1: top_level -= 1
        self.limbdd_mgr.sifting(limbdd, 0, top_level)

    def allocate_input_qubit(self, num_var):
        for var_index in range(num_var):
            self.qubit_mgr.get_input_qubit(var_index)
    
    def identify_cut(self, start, branch_list, anc_num, control_num=0):
        assert(isinstance(start, LimBDDNode))

        cut_map = {}
        for branch in branch_list:
            assert(isinstance(branch, SimpleBranch))

            if branch.bitnum + control_num == 5: target_length = 2
            else: target_length = max(2, (branch.bitnum + control_num + 1) // (anc_num + 1))
            top_branch, bot_branch = branch.cut(target_length)
            mid = LimBDDManager.locate_branch_end(start, branch, length=target_length)

            if mid in cut_map:
                if top_branch in cut_map[mid]: cut_map[mid][top_branch].append(bot_branch)
                else: cut_map[mid][top_branch] = [bot_branch]
            else: cut_map[mid] = {top_branch: [bot_branch]}
        return cut_map

    def get_pivot_to_circ(self, pivot_limbdd, x_map, cx_map, add_before):
        nested_synthesizer = LimBDDOracleSynthesizer(ancilla_num=0, fused_backend=False)
        nested_synthesizer.qubit_mgr = self.qubit_mgr
        nested_synthesizer.circ = MCXCircuit(self.qubit_mgr)
        nested_synthesizer.limbdd_mgr.limbdd = pivot_limbdd
        nested_synthesizer.limbdd_mgr.limbdd_list.clear()
        for height in range(nested_synthesizer.limbdd_mgr.limbdd.root.level + 1):
            nested_synthesizer.limbdd_mgr.limbdd_list[height] = []

        nested_synthesizer.process()

        if not add_before: circ = nested_synthesizer.circ
        else: circ = MCXCircuit(self.qubit_mgr)

        o_qubit_list = []
        for var_index in x_map:
            if x_map[var_index]: o_qubit_list.append(self.qubit_mgr.get_input_qubit(var_index))
        circ.add_lim_operator(o_qubit_list)

        for ctrl_var in cx_map:
            pos_qubit_list, neg_qubit_list = [], []
            for target_var in cx_map[ctrl_var]:
                if cx_map[ctrl_var][target_var]: pos_qubit_list.append(self.qubit_mgr.get_input_qubit(target_var))
                else: neg_qubit_list.append(self.qubit_mgr.get_input_qubit(target_var))
            if not len(pos_qubit_list) and not len(neg_qubit_list): continue

            control_qubit = self.qubit_mgr.get_input_qubit(ctrl_var)
            circ.add_lim_operator(pos_qubit_list, Control( control_qubit ))
            circ.add_lim_operator(neg_qubit_list, Control( control_qubit, polar=0 ))

        if add_before: circ.merge(nested_synthesizer.circ)

        return circ

    def generate_circ_with_anc(self, start, branch_list, anc_num=1, control_list=None):
        assert(isinstance(start, LimBDDNode))

        if control_list is None: control_list = []

        circ = MCXCircuit(self.qubit_mgr)
        if not anc_num:
            for branch in branch_list: circ.merge(self.branch2circ(start, self.limbdd.terminator, branch))
        else:
            short_branches, long_branches = [], []
            for branch in branch_list: 
                if branch.bitnum + len(control_list) <= 3:
                    circ.merge(self.branch2circ(start, self.limbdd.terminator, branch.copy()))
                    short_branches.append(branch)
                else:
                    long_branches.append(branch)
            '''
            if anc_num == self.ancilla_num:
                self.show(self.limbdd, path=f'figure/debug/fig_{config['debug_figure_counter']}')
                config['debug_figure_counter'] += 1
            print('short_branches: ', [str(branch) for branch in short_branches])
            print('long_branches : ', [str(branch) for branch in long_branches])
            '''
            if len(long_branches) == 1 and long_branches[0].bitnum + len(control_list) <= 4:
            #if False:
                circ.merge(self.branch2circ(start, self.limbdd.terminator, long_branches[0]))
            elif len(long_branches):
                ancilla = self.qubit_mgr.get_ancilla_qubit(anc_num)
                
                cut_map = self.identify_cut(start, long_branches, anc_num, control_num=len(control_list))
                last_top_branch, last_mid = None, None
                for mid, branch_map in cut_map.items():
                    #print(f'****** mid: {mid.idx} ******')
                    #print(f'self.anc: {self.ancilla_num}, anc_num: {anc_num}, ancilla: {str(ancilla)}')
                    for top_branch, bot_branch_list in branch_map.items():
                        #print(f'top_branch: {str(top_branch)}, bot_branches: {[str(branch) for branch in bot_branch_list]}')
                        bot_circ = self.generate_circ_with_anc(mid, bot_branch_list, anc_num - 1, [ Control(ancilla) ])
                        if not last_top_branch: top_circ = self.branch2circ(start, mid, top_branch.copy())
                        else:
                            fuse_limbdd, x_map, cx_map = LimBDDManager.pivot_to(start, last_top_branch, top_branch, self.limbdd)
                            #self.show(fuse_limbdd, path=f'figure/debug/fig_{config['debug_figure_counter'] - 1}_fuse')
                            top_circ = self.get_pivot_to_circ(fuse_limbdd, x_map, cx_map, last_mid.level > mid.level)
                            '''
                            self.show(fuse_limbdd, path=f'figure/debug/fig_{config['debug_figure_counter'] - 1}_fuse_syn')
                            fuse_circ = top_circ
                            top_circ = self.branch2circ(start, last_mid, last_top_branch)
                            top_circ.merge(self.branch2circ(start, mid, top_branch.copy()))
                            
                            print(f'fuse_circ:')
                            print(to_qiskit(fuse_circ))
                            print(f'direct_circ:')
                            print(to_qiskit(top_circ))
                            '''
                        circ.merge(top_circ, target=ancilla)
                        circ.merge_controlled_circuit(bot_circ, [ Control(ancilla) ])
                        '''
                        print('top_circ:')
                        print(to_qiskit(top_circ))
                        print('bot_circ:')
                        print(to_qiskit(bot_circ))
                        print('circ:')
                        print(to_qiskit(circ))
                        '''
                        last_top_branch, last_mid = top_branch, mid
                if last_top_branch: circ.merge(self.branch2circ(start, last_mid, last_top_branch), target=ancilla)
        return circ
    
    def branch2circ(self, start, end, branch, target=None):
        assert(isinstance(start, LimBDDNode) and isinstance(end, LimBDDNode) and isinstance(branch, SimpleBranch))

        start_index = start.idx
        key = (start_index, branch.bitstr, branch.bitnum)
        if key in self.nidx2circ: # Circuit already computed.
            circ = self.nidx2circ[key].copy()
        else:
            circ = MCXCircuit(self.qubit_mgr, target)
            if start is end: # terminator
                circ.add_gate([], circ.target)
                self.nidx2circ[key] = circ.copy()
            else:
                control_qubit = self.qubit_mgr.get_input_qubit(start.var_index)
                
                if start.low.node is start.high.node and start.low.weight and start.high.weight:
                    low_o_qubit_list , low_o_qubit_list_  = self.get_o_qubit_list(start.low , terminate_level=end.level)
                    high_o_qubit_list, high_o_qubit_list_ = self.get_o_qubit_list(start.high, terminate_level=end.level)

                    circ.add_lim_operator(low_o_qubit_list , Control(control_qubit, polar=0))
                    circ.add_lim_operator(high_o_qubit_list, Control(control_qubit, polar=1))
                    circ.merge(self.branch2circ(start.low.node, end, branch, target))
                    circ.add_lim_operator(low_o_qubit_list , Control(control_qubit, polar=0))
                    circ.add_lim_operator(high_o_qubit_list, Control(control_qubit, polar=1))
                    ''''''
                    circ.add_lim_operator(low_o_qubit_list_ , Control(control_qubit, polar=0))
                    circ.add_lim_operator(high_o_qubit_list_, Control(control_qubit, polar=1))
                    
                    self.nidx2circ[key] = circ.copy()
                else:
                    polar = branch.bitstr & 1
                    branch.rshift()

                    successor = start.successors[polar]
                    assert(successor.weight)

                    o_qubit_list, o_qubit_list_ = self.get_o_qubit_list(successor, terminate_level=end.level)

                    circ.add_lim_operator(o_qubit_list)
                    circ.merge_controlled_circuit(self.branch2circ(successor.node, end, branch, target), [ Control(control_qubit, polar=polar) ])
                    circ.add_lim_operator(o_qubit_list)
                    ''''''
                    circ.add_lim_operator(o_qubit_list_)
                    
                    self.nidx2circ[key] = circ.copy()
        return circ
    
    def generate_circ_wo_anc(self, node, target=None):
        assert(isinstance(node, LimBDDNode))

        node_index = node.idx
        if node_index in self.nidx2circ: # Circuit already computed.
            circ = self.nidx2circ[node_index][0].copy()
            self.nidx2circ[node_index][1] -= 1
            if not self.nidx2circ[node_index][1]: self.nidx2circ.pop(node_index)
            return circ
        else:
            circ = MCXCircuit(self.qubit_mgr, target)
            if not node.low: # terminator
                circ.add_gate([], circ.target)
                if len(node.parents) >= 2: self.nidx2circ[node.idx] = [circ.copy(), len(node.parents) - 1]
                return circ
            else:
                control_qubit = self.qubit_mgr.get_input_qubit(node.var_index)
                
                if node.low.node is node.high.node and node.low.weight and node.high.weight:
                    low_o_qubit_list , _ = self.get_o_qubit_list(node.low)
                    high_o_qubit_list, _ = self.get_o_qubit_list(node.high)

                    circ.add_lim_operator(low_o_qubit_list , Control(control_qubit, polar=0))
                    circ.add_lim_operator(high_o_qubit_list, Control(control_qubit, polar=1))
                    circ.merge(self.generate_circ_wo_anc(node.low.node, target))
                    circ.add_lim_operator(low_o_qubit_list , Control(control_qubit, polar=0))
                    circ.add_lim_operator(high_o_qubit_list, Control(control_qubit, polar=1))

                    if len(node.parents) >= 2: self.nidx2circ[node.idx] = [circ.copy(), len(node.parents) - 1]
                    return circ
                else:
                    for i in range(2):
                        if not node.successors[i].weight: continue
                        o_qubit_list, _ = self.get_o_qubit_list(node.successors[i])

                        circ.add_lim_operator(o_qubit_list)
                        circ.merge_controlled_circuit(self.generate_circ_wo_anc(node.successors[i].node, target), [ Control(control_qubit, polar=i) ])
                        circ.add_lim_operator(o_qubit_list)
                    if len(node.parents) >= 2: self.nidx2circ[node.idx] = [circ.copy(), len(node.parents) - 1]
                    return circ
    
    def get_o_qubit_list(self, signal, terminate_level=0):
        assert(isinstance(signal, LimBDDSignal))

        o_qubit_list, o_qubit_list_ = [], []
        bitstr = signal.lim.bitstr
        for level_minus_one in range(terminate_level):
            if not bitstr: break

            var_index = self.limbdd.var_order[level_minus_one - self.limbdd.root.level]

            pivot = (1 << level_minus_one)
            if pivot & bitstr:
                o_qubit_list_.append(self.qubit_mgr.get_input_qubit(var_index))
                bitstr -= pivot

        bitstr >>= terminate_level
        for level_minus_one in range(signal.lim.bitnum - terminate_level):
            if not bitstr: break

            var_index = self.limbdd.var_order[level_minus_one + terminate_level - self.limbdd.root.level]

            pivot = (1 << level_minus_one)
            if pivot & bitstr:
                o_qubit_list.append(self.qubit_mgr.get_input_qubit(var_index))
                bitstr -= pivot
        return o_qubit_list, o_qubit_list_
    
    def show(self, limbdd, path="figure/debug/fig", skip_zero=True, title='LimBDD'):
        limbdd.show(path, skip_zero=skip_zero, title=title)