# 整体框架设计（系统当前状态）

> 本文档整合原 `docs/` 中散落的工作计划与进展记录，只保留**当前系统真实状态**的
> 设计参考。历史工作日志、已放弃方案与验收计划均已删除，不再维护。
>
> 数学细节仍以以下三份文档为准，本文不重述证明：
> - `docs/CompleteBranch_algebra.md` —— CompleteBranch 仿射立方代数、定理 2（C1–C4）与规范构造；
> - `docs/exorlink_unification.md` —— exorlink-δ 统一池重写理论（split / 融合 / 对旋转的关系）；
> - `docs/macro_m2_theory.md` —— 宏 M2（Π 中性对 + 有界恶化推广）的形式化。
>
> 本文所有路径、类名与开关以 2025-08-24 之后的仓库为准：类 `LimBDDOracleSynthesizer`
> 已更名 `OracleSynthesizer`，`limbdd/fusion/` 与 `limbdd/CompleteBranch.py` 已迁至
> `backend/`。

---

## 1. 模块布局（重构后）

```
backend/                     合成器与融合引擎
  OracleSynthesizer.py       OracleSynthesizer（原 LimBDDOracleSynthesizer）——五路径总入口
  CompleteBranch.py          CompleteBranch / SyntheticPlan / plan_from_affine / plan_min_cx /
                             t_cost / rref / reduce_basis / in_span / is_affine_cube / affine_from_fn
  fusion/
    BranchFusion.py          池融合入口：fuse_pool_global / fuse_pool / pool_t_cost /
                             anf_pool / anf_degree / shift_fn / four_case_kind
    ClosedForm.py            闭式判定与构造：canonical_affine / fuse_closed / closed_gain
    GLSearch.py              G 选项：gl_search / fprm_best / apply_linear / climb_best
    FprmProxy.py             FP 选项：proxy_scores / fprm_proxy_best
    MacroM2.py               宏 M2 后处理：m2_best_move / m2_interleaved / h_candidates
limbdd/                      BDD 核心与抽取
  Node.py / Lim.py / LimBDD.py / LimBDDManager.py
  patterns/                  SplitPattern / SplitLv2Pattern / SplitInterceptorPattern（split 模式）
  Extraction.py              abs_lim / extract_complete_branches（子图 → CompleteBranch 池）
  FusedBackend.py            limbdd2circ_fused（单子图融合发射，Phase 4 遗留入口）
circuit/
  MCXCircuit.py              MCXCircuit / STGate / order_segments_greedy / simplify_terms /
                             simplify_term_list / boundary_cx_savings
  Control.py                 Control / CtrlSequence（ControlTerm / LimTerm）
utils/
  Qubit.py                   Qubit / QubitManager
  TruthTable.py              TruthTable
  to_qiskit.py               MCXCircuit → Qiskit 电路（evaluate.py 计费用）
config.py                    t_cost / cx_cost 单一事实源（§4）
evaluate.py / test.py        评测入口 / 测试入口
```

要点：
- 生产路径是 **融合后端**（`OracleSynthesizer.fused_backend=True` 且 `ancilla_num=0`）；
  `ancilla_num>0` 的旧 ancilla 路径保留但已弃用，不测。
- `limbdd/patterns/` 只负责 BDD 图阶段的 split 模式；融合引擎不依赖 BDD，只消费
  `(CompleteBranch, root_mask, key)` 池条目。

---

## 2. 综合流水线与五项路径（A / C / F / FP / G）

### 2.1 总流程

```
TruthTable tt
  → OracleSynthesizer.synthesize(tt)
      fused_backend 且 ancilla_num==0:
        allocate_input_qubit(tt.num_var)   # 显式分配全部输入量子位（§6.4）
        process_fused_global(tt)
  → 路径选择（gl_search 时单独走 G，忽略 paths）
  → 融合到不动点（可选 m2_post 宏 M2 相间调度）
  → 发射管线（§2.3）
```

`OracleSynthesizer` 开关：
- `gl_search=False`（默认关）：G 选项，独立成项；
- `fprm_proxy=False`（默认关）：FP 选项，**取代** F（选择关系，非 min）；
- `m2_post=True`（默认开）：获胜池追加宏 M2 后处理；
- `paths=('F',)`（默认）：**单路径纪律**——只允许 `'A'/'C'/'F'` 之一；多路径组合
  （显式 min）在本阶段禁用。FP 与 G 由开关选择，不与其它路径隐式取 min。

### 2.2 五项路径

| 标签 | 入口 | 初始池 / 搜索 | 发射三明治 |
|---|---|---|---|
| **A**（BDD） | `paths=('A',)` | `construct`（变量序 + `try_neg`/`split_neg`/`reorder`/sifting）→ `_collect_pieces_fused`（split 模式；absorb 已移除）→ `extract_complete_branches` | 按片根掩码 `X(r)` 三明治 |
| **C**（ANF） | `paths=('C',)` | `anf_pool`：快速 Möbius，每个单项式 `x^S` 即一个 CompleteBranch（`c=0` 语义） | 无平移三明治 |
| **F**（FPRM，生产默认） | `paths=('F',)` | `_fprm_entries`：平移子群上搜索最优 `c`；`n≤8` 全枚举 `2^n`，`n≥9` 双起点 `climb2`（起点 `0` 与 `2^n−1`） | `X(c)` 三明治（T 免费） |
| **FP**（代理） | `paths=('F',)` 且 `fprm_proxy=True` | `fprm_proxy_best`：全枚举 `2^n` 个平移只做 Möbius，代理分 `S1 = Σt(n−|S|) − P`，`(S1,c)` 升序取 top-8 且恒含 `c=0`，仅对 ≤9 个候选完整融合 | `X(c)` 三明治 |
| **G**（GL） | `gl_search=True`（忽略 paths） | `gl_search`：先求 FPRM 最优 `cF`；起点集 = `{恒等 on f, 恒等 on f∘τ_cF} ∪ {全部单 CNOT on f∘τ_cF}`；CNOT 邻域最陡上升，`max_steps=3`；候选含 `cF`（零 wrapper）⇒ `G ≤ F` | `X(c) · CNOT(A⁻¹) · [段] · CNOT(A) · X(c)`（CNOT 计入 CX） |

路径选择语义：非 G 时按 `paths` 中给定顺序计算，`pool_t_cost` 最低者获胜，并列取先者。
所有路径最终都进入同一个 `fuse_pool_global` 融合引擎。

### 2.3 发射管线（`process_fused_global` 尾部）

1. **平移三明治**：若获胜路径含平移 `c`，先整体施加 `X(c)`（`shift_c=0` 时 no-op）；
2. **GL 线性三明治（仅 G）**：施加 `CNOT(A⁻¹)`（`cnots` 正序）；
3. **根三明治分组**：`kept` 按根掩码 `r` 分组，对每个 `r`（升序）施加 `X(r)`，
   组内段用 `order_segments_greedy` 排序后 `merge`，再闭合 `X(r)`；
4. **standalone 段**：融合产物（`key=None`，自包含）同样经 `order_segments_greedy` 后 `merge`；
5. **GL 线性三明治闭合**：`CNOT(A)`（`cnots` 逆序）；
6. **闭合平移三明治** `X(c)`；
7. `simplify_terms()`：相邻 `LimTerm` 合并（无条件 X 异或、同控同极条件翻转异或、
   同控异极同目标合并为无条件 X）。

每段自身由 `CompleteBranch.to_circuit` 发射为**对合三明治**：`X(x_trans) → CNOT 网络 →
MCX → CNOT 网络(逆) → X(x_trans)`。合成段发射用 `plan_min_cx`（CX 最优基，T 不变）；
存储的 `SyntheticPlan` 仍保持 RREF 规范形（融合池幂等/哈希）。

**B-4 段重排**（`order_segments_greedy`）：以相邻段边界抵消 `boundary_cx_savings` 为边权
做最近邻贪心（全起点），候选含原始顺序，按真实 `merge + simplify` 管线评估 CX 取最小——
**只影响 CX，不改变 T**（段集合不变）。

---

## 3. 融合引擎要点

- **表示**：每段 = 仿射立方指示函数 `base + span(basis)`；`CompleteBranch.affine()` 为
  O(n²) 符号推导（无 2^n 枚举）。跨片平移等价于 `base ^= root_mask`（`canonical_affine`）。
- **闭式判定 C1–C4**（`ClosedForm.fuse_closed`，O(n²) 位运算）：
  - `K_A == K_B`：`d ∈ K` → C1 抵消（恒接受）；否则 C2 融合；
  - `K_A ⊂ K_B` 或 `K_B ⊂ K_A` 余维 1 且同大陪集 → C3（集合差，余维 1 嵌套）；
  - 同维、交余维 1、陪集相交 → C4（`_build_joint`/`_solve_split` 一次消元求交点）。
- **`fuse_pool_global`**（生产引擎）：输入 `[(CompleteBranch, root_mask, key)]`；
  - **分桶**：按 `dim(K)` 分桶，只枚举同桶与相邻桶（定理 2 保证可融合 ⇒ dim 差 ≤1）；
  - **增量匹配**：候选键 `(gain, i, j, cancel)` 全局唯一；幸存候选只重映射索引，
    新候选仅涉及融合产物；
  - **纯立方快径**：方向子空间为坐标子空间时退化为 O(popcount) 掩码判定；
  - 接受准则：抵消恒接受；非抵消仅接受**严格正收益** `gain = t(A)+t(B)−t(F) > 0`。
  - 返回 `(kept, standalone)`：`kept` 保留片身份（在对应 `X(r)` 三明治内发射），
    `standalone` 为融合产物（自包含）。
- **FPRM 预算切换**（`_fprm_entries` / `fprm_best`）：
  - `n≤8`：全枚举 `2^n` 个平移（n=8 即 256 次完整融合）；
  - `n≥9`：`climb2` 双起点 `{0, 2^n−1}` 最陡上升（Hamming 邻域、严格改进、缓存），
    只评估几十次完整融合。**这解释 F 路径 n=9 wall < n=8 wall**（1368ms vs 1984ms）。
- **G-2.2 gl_search**：确定性 GL(n,2) 局部搜索，小 n 强化选项；n≤7 已验证，
  n=6 约 **1.05s/例**。时间大头是每次换坐标都要重跑 `fuse_pool_global`（~94%）。
- **fprm_proxy（方案 A）**：代理预筛取代 F。代理分 `S0 = Σt(n−|S|)`（融合前精确 ΣT）、
  `P = 差一变量对计数`、`S1 = S0 − P`；top-8 完整融合。恒含 `c=0` ⇒ 结果 ≤ C。
  n≥9 部署区：保留 F 的 **80–96%** T 缩减质量、快 **3–5×**。
- **MacroM2（`m2_post`，默认开启）**：
  - 调度：`fuse_pool_global` 到不动点 → 枚举最佳宏 M2 → 重融，循环 ≤ `max_rounds=30`；
  - 接受条件：真实总收益 **`imm + g_fuse > 0`**，其中
    `imm = t(A)+t(B)−t(A⊕h)−t(B⊕h)`（不动点处 ≤0）、
    `g_fuse = t(A⊕h)+t(C)−t((A⊕h)⊕C)`；Π 中性对（`imm==0`）是有界恶化推广
    （`imm<0` 但融合补偿）的特例；
  - C 预筛：`gain(A′,C)>0` 的必要条件是 `dim(K)` 差 ≤1，按 dim 分桶扫描；
  - XOR 不变量断言仅在 `LIMBDD_FUSION_DEBUG=1` 下执行；每步严格下降 ⇒ 逐例 T 零回归。

---

## 4. 成本模型与数据布局

### 4.1 成本函数（`config.py` 单一事实源）

```python
t_cost  = lambda x: 0 if x <= 1 else 8 * (x - 1) - 1
cx_cost = lambda x: 0 if x <= 0 else (1 if x == 1 else 6 * (x - 1))
```

口径：MCX/CCX 按其控制数计 `t_cost(k)` / `cx_cost(k)`；显式 CX 计 1 个 CX；X 免费。

### 4.2 数据目录

```
data/test_cases/test_cases_{n}                    # 每行一个真值表整数；n=3 全 256、n=4 全 65536、n≥5 各 10000
data/exorcism_cost/exorcism_{t,cx}_cost_{n}       # Exorcism 基线（按行号对齐）
data/sshr_cost/sshr_{t,cx}_cost_{n}               # SSHR 基线（n≥7 暂空则自动跳过）
data/our_cost/
  t_cost/    our_t_cost_{tag}_{n}_{start}_{stop}.txt
  cx_cost/   our_cx_cost_{tag}_{n}_{start}_{stop}.txt
  deg_bound/ our_deg_bound_{tag}_{n}_{start}_{stop}.txt   # G-3 下界 t(deg f)
  our_results_{tag}_{n}_{start}_{stop}.txt                # 汇总报告
```

- `tag ∈ {A, C, F, FP, G}`：`paths_tag` 生成——`gl_search` → `G`；`fprm_proxy` 且 F → `FP`；
  其余取单路径字符。
- 2025-08-24 数据口径：n=3 全 256、n=4 全 65536、n=5–7 各 10000、n=8–10 各 1000 例；
  A/C/F/FP 全 n，G 仅 n=3–6；融合后端含 `m2_post`（默认）。

---

## 5. 测试纪律与验收口径

1. **例行改动只测 n≤7**；n≥8 仅在最终汇总轮跑（时间预算约束）。
2. **单路径纪律**：只测 A / C / F / FP / G 五项独立路径，路径间不取 min；
   FP 与 G 为开关项，不与默认 F 隐式组合。
3. **确定性**：全枚举 / 确定性起点（climb2、GL 起点集、代理排序 `(S1,c)`）/
   确定性平局规则；不使用随机采样（旧 rand64 已被 climb2 取代）。
4. **XOR 不变量**：融合引擎与宏 M2 在 `LIMBDD_FUSION_DEBUG=1` 下逐轮断言
   池 XOR == 原函数（M2 重写 `h⊕h=0` 保证）。
5. **时间测量**：`time.perf_counter_ns()` 包住 `synthesize(tt)`，逐例累计取平均
   wall（`evaluate.py` 输出 `avg synthesize wall`）；同机、同提交、同进程数记录。
6. **发射验证**：端到端 `circ.verify(tt)` / `compute_tt() == tt.bitstr` 逐例通过
   （tests/test_* 各路径套件均含发射级 compute_tt 对账）。

---

## 6. 已定结论（final decisions）

### 6.1 图阶段与路径定位

- **absorb 已移除**：`LimBDDManager.collect_branches` 不再调用 `try_absorb_branch`
  （函数本体保留供研究）。P4 实测其单独效应严格为零（±8 噪声级）——它只做 C1 抵消，
  而 `fuse_pool_global` 无条件覆盖 C1。
- **split 模式保持激活**：split 是 BDD 路径的核心承诺机制（去掉后 n≤6 恶化 10–13%），
  代数本质见 `exorlink_unification.md`（δ=2 几何子类）。
- **BDD 路径是 n≤6 的廉价补充**：A 的小 n 优势来自 Shannon/LIM 结构化粗池 + split 承诺；
  n≥7 后 A 相对 F 明显退化（A 的 Exorcism CX 缩减转负 −3.1%→−9.2%），不作主力。
- **生产后端 = F（m2_post 开）**；**FP** 用于时间敏感运行（质量略降、更快）；
  **G** 仅小 n 质量模式（n≤6 质量最高，但 wall 1051.6ms/例 @n=6）。
- **A / C 在 n≥7 退化**：相对 F，二者 T 缩减明显收窄；A 的 Exorcism CX 缩减转负
  （−3.1%→−9.2%），C 的 Exorcism CX 缩减仅 +4~9% 量级。二者保留为路径对照与
  粗池研究价值，不作为生产默认。

### 6.2 最终数据摘要（2025-08-24，`experiments/data_summary.py`）

**F 路径 vs Exorcism T（n=3..10）**：`+30.9% / +38.4% / +41.4% / +38.4% / +32.2% /
+27.3% / +21.7% / +20.8%`，**higher 例数全 0**。

**F 路径 vs Exorcism CX（n=3..10）**：`+18.9% / +24.6% / +28.8% / +26.6% / +22.0% /
+19.0% / +14.9% / +15.5%`（n≥7 higher ≤10 例）。

**F 路径 vs SSHR（n=3..6）**：T `+20~29%`，CX `+18~27%`。

**wall（ms/例，F）**：

| n | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| F | 0.5 | 2.2 | 11.4 | 61.0 | 340.9 | 1984.4 | 1368.2 | 5411.5 |

（FP：0.5 / 1.3 / 3.7 / 12.0 / 41.6 / 155.2 / 639.5 / 2776.4；A/C 介于 0.3~1158。
F n=9 < n=8 的成因见 §3 FPRM 预算切换。）

### 6.3 质量/时间开关结论

- **GL**：n≤6 质量最高（G n=6 Exorcism T +50.0%），但 wall 显著（n=6 1051.6ms/例），
  仅作小 n 质量展示/强化选项。
- **FP**：保留 F 的 80–96% T 缩减质量，快 3–5×；n≥9 部署区初测双优，正式汇总后
  作为时间敏感场景的默认替代。
- **M2**：默认开启（T 逐例零回归、n≤7 全量 CX 劣化 0 例、时间 0~2.3ms/例）；在 F/G
  上边际 ≈0、在 A/C/FP 上 −0.4~−6.4%，但廉价且零回归，故并入默认融合后端。

### 6.4 输入分配修复（融合后端）

融合后端跳过 `construct` 时不建 BDD，此前输入量子位惰性分配——若某变量不出现在任何
发射算子（例如 f 与 x₀ 无关，回归例 n=4 tt=64560），`compute_tt` 会在缩小的变量空间上
枚举，导致发射 tt 错位。现 `synthesize` 在融合路径入口 **显式分配全部输入量子位**
（`allocate_input_qubit(tt.num_var)`）。**此前融合后端生成的旧数据文件可能含此类静默
错误条目，最终聚合前需重跑校验**（当前 `data/our_cost/` 为修复后重跑产物）。

---

## 7. 与保留理论文档的关系

- `docs/CompleteBranch_algebra.md`：融合判定的数学基础（定理 1/2、§3.6–3.7 规范构造、
  平移不变性），是 `backend/CompleteBranch.py` 与 `backend/fusion/ClosedForm.py` 的依据；
- `docs/exorlink_unification.md`：统一池重写算子族（δ=1 融合、δ=2 split、T-中性对旋转），
  是理解「split 已被构图期前置消费、引擎内 δ≥3 机会为 0」的理论框架；
- `docs/macro_m2_theory.md`：宏 M2 形式化，对应 `backend/fusion/MacroM2.py` 的生产实现。
