from utils.Qubit import QubitManager
from utils.TruthTable import TruthTable
from limbdd.Node import LimBDDNode, LimBDDSignal
from limbdd.LimBDD import LimBDD, SimpleBranch
from limbdd.LimBDDManager import LimBDDManager
from limbdd.FusedBackend import limbdd2circ_fused
from backend.CompleteBranch import _bits
from limbdd.Extraction import abs_lim, extract_complete_branches
from backend.fusion.BranchFusion import fuse_pool_global, pool_t_cost, anf_pool, shift_fn
from backend.fusion.GLSearch import gl_search
from backend.fusion.FprmProxy import fprm_proxy_best
from backend.fusion.MacroM2 import m2_interleaved
from backend.fusion.CoverGenerators import (minterm_entries, pk_entries,
                                            pk_proxy_best, pk_shift_best)
from circuit.Control import Control
from circuit.MCXCircuit import MCXCircuit, order_segments_greedy
from utils.to_qiskit import to_qiskit
from config import config

class OracleSynthesizer:
    def __init__(self, reorder=True, ancilla_num=0, fused_backend=True, gl_search=False,
                 fprm_proxy=False, paths=('FS',), m2_post=True):
        self.limbdd = None
        self.limbdd_mgr = LimBDDManager()
        self.qubit_mgr = QubitManager()
        self.circ = MCXCircuit(self.qubit_mgr)
        self.nidx2circ = {}             # { LimBDDNode.idx : [ MCXCircuit, int ] }
        self.reorder = reorder
        self.ancilla_num = ancilla_num
        # Since Phase 5 the fused emission backend is the default backend of the no-ancilla path
        # (silently ignored when ancilla_num>0)
        self.fused_backend = fused_backend
        # G-2.2 optional switch (off by default): deterministic GL(n,2) local search, time for quality
        # (about +6s/case at n=7, 94% of which is the fusion engine itself -- see docs/architecture.md, section 3).
        # Exploration-phase discipline: FG is an **independent option** (it does not implicitly carry A);
        # only one route is selected; multi-path combinations are not supported.
        self.gl_search = gl_search
        # Plan A optional switch (off by default): the fusion-free proxy prescreen **replaces** the FS path
        # (a choice relation, not a min; intended to upgrade the current shift climb2 for n≥9, 12~28%
        # worse quality for n≤8)
        self.fprm_proxy = fprm_proxy
        # Path selection (exploration-phase discipline): **single-path only** -- paths is a
        # single route name 'A'/'F'/'FS' (or a 1-tuple thereof); multi-path combinations (explicit min) are
        # disabled and untested at this stage.
        # Actual test paths = A / F / FS / FP (proxy switch) / FG (gl switch).
        # With gl_search=True the FG option runs alone, ignoring paths.
        if isinstance(paths, (tuple, list)):
            assert len(paths) == 1 and paths[0] in ('A', 'F', 'FS'),                 f'illegal paths (only single-path A/F/FS allowed at this stage): {paths}'
            self.paths = tuple(paths)
        else:
            assert paths in ('A', 'F', 'FS'), f'illegal paths (only single-path A/F/FS allowed at this stage): {paths}'
            self.paths = (paths,)
        # Macro M2 interleaved post-processing switch (integrated after P9; **default on**): at the winning pool's fusion fixed point apply the strictly-decreasing macro
        # {A,B,C}→{B⊕h,(A⊕h)⊕C} (docs/architecture.md, Fusion engine highlights).
        # Accepted moves strictly lower T; CX is not part of the acceptance condition.
        # Note: with it on, the FG≤FS zero-regression chain becomes empirical (M2 opportunities on FG pools
        # are usually 0 and occasionally present on FS -- FG is a slow enhancement option rather than the
        # production default, so that chain is not a constraint; the structural comparison is kept in
        # tests/test_gl_search.py's explicit m2_post=False setting).
        self.m2_post = m2_post

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
            # The fused backend skips construct (no BDD build) and must explicitly allocate all input
            # qubits: otherwise a qubit whose input no leading segment touches is lazily missing and
            # compute_tt enumerates over a shrunken variable space (regression case: n=4 tt=64560,
            # cF=0 and the first segment only uses qubits 1/2/3).
            self.allocate_input_qubit(tt.num_var)
            self.process_fused_global(tt, path)   # Phase 5+: global fusion backend
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
        Graph phase of the Phase-5 fusion path (line-for-line isomorphic to the old process()'s graph
        phase): try_neg / split_neg / reorder / splitting, returning the non-zero
        subgraphs in processing order. No circuits are generated; shared by process_fused_global and
        the residual-measurement scripts.
        """
        mgr = self.limbdd_mgr
        mgr.try_neg(mgr.limbdd)
        mgr.split_neg(mgr.limbdd)

        if self.reorder:
            self.variable_reorder()
            if path: self.show(mgr.limbdd, path + '_reorder')

        pieces = []  # processing order (subgraphs still non-zero after splitting)
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
        """One complete BDD decomposition (construct + graph phase + sifting), returning the global
        entry list. construct rebuilds limbdd_mgr, so repeated consecutive calls are safe."""
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
        Phase 5+: global fusion backend (default no-ancilla path).

        Path selection: paths selects exactly one of A/F/FS, defaulting to ('FS',).
          A. BDD path: graph phase (try_neg / split_neg / reorder / splitting, absorb removed)
             → branch pool → global fusion;
          F. ANF path: truth table → Möbius → monomial pool (c=0);
          FS. FPRM path (G-2.1): search for c by pairwise-fused T cost (n≤8 full 2^n
             enumeration; n≥9 deterministic two-start climb2); circuit = X(c) sandwich (T-free).
        With fprm_proxy=True, FS is replaced by the fusion-free proxy prescreen version (choice relation,
        not min). No comparison across multiple paths is performed.

        Optional (gl_search=True, G-2.2): ignore paths and run the **FG option** alone -- deterministic
        GL(n,2) local search (candidates include the FS-selected cF, so FG ≤ FS before M2;
        zero wrapper preferred on T ties). Applying M2 to both winning pools does not preserve
        this dominance guarantee. Exploration-phase discipline: FG is an
        independent option, no implicit min(A,FG). Cost: about +6s/case at n=7.

        Note: the F' path (ANF one-variable-apart pre-merge, 1.2/T-3) was measured and
        **turned off for now** (end-to-end gain −0.22%~−0.53% ΣT, cost +23%~36% wall, sitting in the
        decision-critical zone); `BranchFusion.anf_pool_premerged` and `tests/test_premerge.py` are
        kept and can be re-enabled at any time.

        Optional (m2_post=True, macro M2 interleaved post-processing, **default on**): on the winning
        pool's fusion fixed point apply the strictly-decreasing macro {A,B,C}→{B⊕h,(A⊕h)⊕C}
        (docs/architecture.md, Fusion engine highlights). Per-case T never regresses;
        CX may increase. m2_post=False retains pairwise fusion alone, for structural comparisons.

        Emission: grouped by root mask (one X(r) … X(r) sandwich per r), fused segments (r=0) emitted
        independently; an overall X(c) sandwich when FS/FP wins; CNOT(A⁻¹)/CNOT(A) linear sandwich when
        FG wins; simplify_terms at the end.
        """
        n = tt.num_var

        if self.gl_search:
            # ---- option FG: deterministic GL(n,2) local search (independent option, no implicit min) ----
            # FG runs alone and includes the FS-selected cF among its internal candidates,
            # so its pairwise-fused T cost is no greater than FS before M2.
            kept, standalone, shift_c, cnots = gl_search(tt.bitstr, n)
            if self.m2_post:
                kept, standalone = m2_interleaved(kept, standalone, n)
        else:
            candidates = []
            if 'A' in self.paths:
                kept_a, standalone_a = fuse_pool_global(self._bdd_entries(tt, path), n)
                if self.m2_post:
                    kept_a, standalone_a = m2_interleaved(kept_a, standalone_a, n)
                candidates.append((pool_t_cost(kept_a, standalone_a), kept_a, standalone_a, 0, ()))
            if 'F' in self.paths:
                entries_c = anf_pool(tt.bitstr, n)
                kept_c, standalone_c = fuse_pool_global(entries_c, n)
                if self.m2_post:
                    kept_c, standalone_c = m2_interleaved(kept_c, standalone_c, n)
                candidates.append((pool_t_cost(kept_c, standalone_c), kept_c, standalone_c, 0, ()))
            if 'FS' in self.paths:
                # With Plan A (fprm_proxy=True) the proxy prescreen version **replaces** the default FS (choice relation, not min)
                if self.fprm_proxy:
                    kept_f, standalone_f, shift_c = fprm_proxy_best(tt.bitstr, n)
                else:
                    kept_f, standalone_f, shift_c = self._fprm_entries(tt, n)
                if self.m2_post:
                    kept_f, standalone_f = m2_interleaved(kept_f, standalone_f, n)
                candidates.append((pool_t_cost(kept_f, standalone_f), kept_f, standalone_f, shift_c, ()))
            assert candidates, 'paths is empty (already validated at construction; should be unreachable)'
            best_cost = min(c[0] for c in candidates)
            _, kept, standalone, shift_c, cnots = next(c for c in candidates if c[0] == best_cost)

        self._emit_pool(kept, standalone, shift_c, cnots)

    def _emit_pool(self, kept, standalone, shift_c=0, cnots=(), reorder=True):
        """
        Shared emission pipeline for a fused pool (kept, standalone):
        shift sandwich X(c) → GL wrapper CNOT(A⁻¹) → root sandwiches per r → B-4 segment
        reordering → GL wrapper CNOT(A) → shift sandwich → simplify_terms.

        reorder=False skips the B-4 segment reordering; it is used by the raw MT0/PK0
        baselines, whose pools may contain thousands of blocks and for which the O(m²)
        reordering pass would dominate the runtime (T is unaffected either way). The raw
        ANF baseline F0 and preprocessed DD baseline A0 follow the same emission policy.
        A0 retains root-mask grouping and DD variable reordering; neither is B-4 reordering.
        """
        # ---- emission: shift sandwich X(c) (no-op when shift_c=0) ----
        shift_qubits = [self.qubit_mgr.get_input_qubit(v) for v in _bits(shift_c)]
        self.circ.add_lim_operator(shift_qubits)

        # ---- linear sandwich when GL wins: CNOT(A⁻¹) = cnots in forward order (A = E1∘…∘Ek) ----
        for c_ctrl, t_tgt in cnots:
            self.circ.add_lim_operator([self.qubit_mgr.get_input_qubit(t_tgt)],
                                       Control(self.qubit_mgr.get_input_qubit(c_ctrl), 1))

        kept_by_r = {}
        for cb, r, idx in kept:
            kept_by_r.setdefault(r, []).append(cb)
        for r, segs in sorted(kept_by_r.items()):
            r_qubits = [self.qubit_mgr.get_input_qubit(v) for v in _bits(r)]
            self.circ.add_lim_operator(r_qubits)
            # B-4: segment order only affects CX (the segment set is unchanged, so T is unchanged);
            # evaluate through the real merge pipeline (qubit_map remapping), prefix = current unsimplified circuit.
            seg_circs = [cb.to_circuit(self.qubit_mgr) for cb in segs]
            if reorder:
                seg_circs = order_segments_greedy(seg_circs, self.circ if not self.circ.empty() else None)
            for seg_circ in seg_circs:
                self.circ.merge(seg_circ)
            self.circ.add_lim_operator(r_qubits)
        stand_circs = [cb.to_circuit(self.qubit_mgr) for cb in standalone]
        if reorder:
            stand_circs = order_segments_greedy(stand_circs, self.circ if not self.circ.empty() else None)
        for seg_circ in stand_circs:
            self.circ.merge(seg_circ)

        # ---- linear sandwich when GL wins: CNOT(A) = cnots in reverse order ----
        for c_ctrl, t_tgt in reversed(cnots):
            self.circ.add_lim_operator([self.qubit_mgr.get_input_qubit(t_tgt)],
                                       Control(self.qubit_mgr.get_input_qubit(c_ctrl), 1))

        self.circ.add_lim_operator(shift_qubits)
        self.circ.simplify_terms()

    def process_cover_route(self, tt, route):
        """
        Starting-cover routes:

          A0  : fully preprocessed DD pool emitted without the fusion engine (baseline);
          MT  : all-minterm pool (one dim-0 block per assignment in supp(f));
          PK  : pseudo-Kronecker cover (per-term polarity, no exorlink rewrites);
          PKS : PK cover with shift search (full enumeration n≤8, climb2 n≥9);
          PKP : PK cover with proxy-prescreened shift search (top-8 + c=0);
          F0  : raw ANF pool emitted **without** the fusion engine (baseline);
          MT0 : raw MT pool emitted **without** the fusion engine (baseline);
          PK0 : raw PK pool emitted **without** the fusion engine (baseline).

        MT/PK/F0/MT0/PK0 skip the transformation search entirely (T = identity); PKS/PKP
        wrap the emission in the X(c) sandwich. Macro M2 interleaving follows the
        regular `m2_post` switch for the engine routes; the raw baselines A0/F0/MT0/PK0 skip
        both the engine and the macro (there is no fixed point to apply it to) and are
        emitted without the B-4 reordering pass. A0 shares A's complete DD preprocessing
        (including `self.reorder`) and preserves root-mask sandwiches during emission;
        the other raw covers have zero root masks and are emitted in generator order.
        """
        n = tt.num_var
        assert route in ('A0', 'MT', 'PK', 'PKS', 'PKP', 'F0', 'MT0', 'PK0'), f'illegal cover route: {route}'
        if route == 'A0':
            # DD root masks are part of the cover semantics, unlike the zero-root raw
            # generators below. Keep them intact and skip fusion, M2, and B-4 ordering.
            self._emit_pool(self._bdd_entries(tt, None), [], reorder=False)
            return
        raw_generators = {'F0': anf_pool, 'MT0': minterm_entries, 'PK0': pk_entries}
        if route in raw_generators:
            entries = raw_generators[route](tt.bitstr, n)
            self._emit_pool([], [cb for cb, _, _ in entries], 0, (), reorder=False)
            return
        if route == 'MT':
            kept, standalone = fuse_pool_global(minterm_entries(tt.bitstr, n), n)
            shift_c = 0
        elif route == 'PK':
            kept, standalone = fuse_pool_global(pk_entries(tt.bitstr, n), n)
            shift_c = 0
        elif route == 'PKS':
            kept, standalone, shift_c = pk_shift_best(tt.bitstr, n)
        else:
            kept, standalone, shift_c = pk_proxy_best(tt.bitstr, n)
        if self.m2_post:
            kept, standalone = m2_interleaved(kept, standalone, n)
        self._emit_pool(kept, standalone, shift_c, ())

    def _fprm_entries(self, tt, n):
        """
        G-2.1: search shifted ANF pools by their pairwise-fused T cost.
        The search is exhaustive for n≤8 and heuristic for n≥9; M2 is applied later.

        Returns (kept, standalone, best_c). Budget policy (stage 3.1-C decision):
          n ≤ 8: full 2^n enumeration (cheap, under 1.9s/case);
          n ≥ 9: climb2 -- steepest ascent from the two starts c=0 and c=2^n−1 (Hamming neighborhood,
                 strict improvement, cache reuse, deterministic); measured to realize 85%~100% of the
                 full-enumeration gain, better than the old rand64 sampling (72%~81%).
        Recorded but unimplemented options (see docs/architecture.md):
          ① a hybrid of rand64 sample points as multi-start climbing; ② an evaluate.py full-enumeration offline switch.
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
            # Phase 4: CompleteBranch fused emission backend (extract → fusion pool → per-segment emission + root sandwich)
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
        nested_synthesizer = OracleSynthesizer(ancilla_num=0, fused_backend=False)
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
