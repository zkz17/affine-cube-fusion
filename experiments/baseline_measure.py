# -*- coding: utf-8 -*-
"""
阶段 0.3:1.1(T-1 闭式切换)的验收基线。

记录三类量,全部**非侵入式**(不改生产代码,仅在本脚本内 monkeypatch 计数/计时):

1. **逐例 ΣT**(质量不变量)——1.1 是纯性能改动,切换后必须与本基线**逐位一致**;
2. **融合阶段耗时**——1.1 要优化的对象,单独计时以便量化提速倍数;
3. **fn 路径的调用规模**——`is_affine_cube` / `shift_fn` 调用次数与其 O(2^n) 工作量,
   用于核对「消掉的 2^n 因子」是否与实测提速相符。

输出:
  experiments/phase0_results/baseline_tcost_{n}.txt   逐例 `tt:T`(与 evaluate.py 同口径,便于 diff)
  experiments/phase0_results/baseline_summary.txt     汇总表

用法:
  ./venv/bin/python experiments/baseline_measure.py            # 默认 n=4..7
  ./venv/bin/python experiments/baseline_measure.py 4 5 6 7 --cases 200
"""
import argparse
import os
import sys
import time

# 项目根目录（脚本位于 experiments/ 下，故上溯一级）；必须先于 limbdd/evaluate 导入加入 sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXPERIMENTS = os.path.join(ROOT, 'experiments')

import limbdd.fusion.BranchFusion as BF
import limbdd.LimBDDOracleSynthesizer as SYN
from limbdd.LimBDDOracleSynthesizer import LimBDDOracleSynthesizer
from limbdd.CompleteBranch import t_cost as mcx_t_cost
from limbdd.fusion.BranchFusion import anf_degree
from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit
from evaluate import count_t_cost

OUT_DIR = os.path.join(EXPERIMENTS, 'phase0_results')


class Probe:
    """融合阶段的计时与调用计数(monkeypatch,生产代码零改动)。"""

    def __init__(self):
        self.fuse_time = 0.0
        self.affine_calls = 0
        self.affine_work = 0      # Σ 2^n,即 is_affine_cube 的枚举总规模
        self.shift_calls = 0
        self.shift_work = 0
        self._orig = {}

    def install(self):
        # 注意:Synthesizer 以 `from ... import fuse_pool_global` 绑定了名字,
        # 因此必须替换 **Synthesizer 模块命名空间**里的那个名字,而不是 BF 里的。
        orig_pool = SYN.fuse_pool_global
        self._orig = {'pool': orig_pool}

        # 1.1 起生产融合路径为闭式,不再经过 is_affine_cube / shift_fn:
        # 若这些名字仍存在于 BranchFusion(例如回退到旧实现时),继续对其计数;
        # 否则保持 0,由输出列直接证明 2^n 路径已从生产链路上消失。
        if hasattr(BF, 'is_affine_cube'):
            orig_affine = BF.is_affine_cube
            self._orig['affine'] = orig_affine

            def affine(fn, n):
                self.affine_calls += 1
                self.affine_work += 1 << n
                return orig_affine(fn, n)

            BF.is_affine_cube = affine

        if hasattr(BF, 'shift_fn'):
            orig_shift = BF.shift_fn
            self._orig['shift'] = orig_shift

            def shift(fn, r, n):
                self.shift_calls += 1
                self.shift_work += 1 << n
                return orig_shift(fn, r, n)

            BF.shift_fn = shift

        def pool(entries, n, max_rounds=None):
            t0 = time.perf_counter()
            try:
                return orig_pool(entries, n, max_rounds)
            finally:
                self.fuse_time += time.perf_counter() - t0

        SYN.fuse_pool_global = pool

    def uninstall(self):
        for name, orig in self._orig.items():
            if name == 'pool':
                SYN.fuse_pool_global = orig
            else:
                setattr(BF, name, orig)


def load_cases(n, count):
    path = os.path.join(ROOT, f'data/exorcism_rand_tcost_{n}.txt')
    with open(path) as f:
        lines = f.readlines()
    out = []
    for line in lines[:count]:
        tt_str, t_exor = line.split(':')
        out.append((int(tt_str), int(t_exor)))
    return out


def measure(n, count, verify):
    cases = load_cases(n, count)
    synth = LimBDDOracleSynthesizer(reorder=True, ancilla_num=0, fused_backend=True)
    probe = Probe()
    probe.install()

    rows = []
    t_sum, exor_sum, bound_sum, wall = 0, 0, 0, 0.0
    failures = []
    try:
        for tt_int, t_exor in cases:
            tt = TruthTable(tt_int, n)
            t0 = time.perf_counter()
            synth.synthesize(tt)
            wall += time.perf_counter() - t0

            t_cost = count_t_cost(to_qiskit(synth.circ))
            rows.append((tt_int, t_cost))
            t_sum += t_cost
            exor_sum += t_exor
            # G-3:免费质量下界 t(deg f)
            bound_sum += mcx_t_cost(anf_degree(tt.bitstr, n))

            if verify and not synth.circ.verify(tt):
                failures.append(tt_int)
    finally:
        probe.uninstall()

    with open(os.path.join(OUT_DIR, f'baseline_tcost_{n}.txt'), 'w') as f:
        for tt_int, t_cost in rows:
            f.write(f'{tt_int}:{t_cost}\n')

    return {
        'n': n, 'cases': len(cases), 'sum_t': t_sum, 'sum_exor': exor_sum,
        'sum_bound': bound_sum,
        'wall': wall, 'fuse_time': probe.fuse_time,
        'affine_calls': probe.affine_calls, 'affine_work': probe.affine_work,
        'shift_calls': probe.shift_calls, 'shift_work': probe.shift_work,
        'failures': failures,
    }


def main():
    ap = argparse.ArgumentParser(description='阶段 0.3 基线量测')
    ap.add_argument('ns', type=int, nargs='*', default=[4, 5, 6, 7])
    ap.add_argument('--cases', type=int, default=200, help='每个 n 取前多少例')
    ap.add_argument('--no-verify', action='store_true', help='跳过 compute_tt 逐例验证')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for n in args.ns:
        print(f'--- n={n} ({args.cases} cases) ---', flush=True)
        r = measure(n, args.cases, not args.no_verify)
        results.append(r)
        print(f'  ΣT={r["sum_t"]} (exorcism {r["sum_exor"]})  G-3 下界 Σt(deg f)={r["sum_bound"]} '
              f'({r["sum_bound"] / max(r["sum_t"], 1):.0%} of ΣT)  '
              f'wall={r["wall"]:.2f}s  fusion={r["fuse_time"]:.2f}s '
              f'({r["fuse_time"] / max(r["wall"], 1e-9):.0%} of wall)', flush=True)
        print(f'  is_affine_cube: {r["affine_calls"]} calls / {r["affine_work"]:,} enum steps;  '
              f'shift_fn: {r["shift_calls"]} calls / {r["shift_work"]:,} steps', flush=True)
        if r['failures']:
            print(f'  !!! VERIFY FAILED for {len(r["failures"])} cases: {r["failures"][:5]}', flush=True)

    lines = []
    lines.append('阶段 0.3 基线(1.1 闭式切换的验收对照)')
    lines.append(f'命令: {" ".join(sys.argv)}')
    lines.append('')
    header = f'{"n":>3} {"cases":>6} {"ΣT":>10} {"ΣT_exor":>10} {"Σbound":>8} {"wall(s)":>9} {"fusion(s)":>10} {"fus%":>5} {"affine_calls":>13} {"affine_steps":>14} {"shift_calls":>12} {"verify":>7}'
    lines.append(header)
    lines.append('-' * len(header))
    for r in results:
        lines.append(f'{r["n"]:>3} {r["cases"]:>6} {r["sum_t"]:>10} {r["sum_exor"]:>10} {r["sum_bound"]:>8} '
                     f'{r["wall"]:>9.2f} {r["fuse_time"]:>10.2f} '
                     f'{r["fuse_time"] / max(r["wall"], 1e-9) * 100:>4.0f}% '
                     f'{r["affine_calls"]:>13,} {r["affine_work"]:>14,} {r["shift_calls"]:>12,} '
                     f'{"FAIL" if r["failures"] else "ok":>7}')
    lines.append('')
    lines.append('逐例 T 见 experiments/phase0_results/baseline_tcost_{n}.txt。')
    lines.append('1.1 验收:切换闭式后重跑本脚本,baseline_tcost_{n}.txt 必须逐位一致(diff 为空)。')
    text = '\n'.join(lines)

    with open(os.path.join(OUT_DIR, 'baseline_summary.txt'), 'w') as f:
        f.write(text + '\n')
    print()
    print(text)


if __name__ == '__main__':
    main()
