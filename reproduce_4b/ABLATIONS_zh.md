# OPD 论文消融实验正式运行指南（2×A100 / 可选 4B 扩展）

本指南对应论文 *Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe* 的 Figure 2、6、7、8、9、10、11(a)、12、13、15、16，以及本地的 4B 规模迁移检查。论文证据见[本地 PDF](../paper/rethinking_opd.pdf)和[中文精读](../paper/rethinking_opd_summary_zh.md)；机器可执行定义以 [`ablation_matrix.json`](./ablation_matrix.json) 为唯一注册表。

默认矩阵共有 **24 个 cell**。先做 `plan → smoke → calibration → 单组 pilot`，不要直接提交 24 个 `paper` 运行。本文中的“paper-exact”只表示论文披露的因子、模型或文本条件被准确映射；原论文使用 8×A800 80GB，本机是 2×A100 80GB，因此即使使用 `paper` 协议，也不宣称 bitwise reproduction。

## 1. 证据等级：先定标签，再看曲线

结果表必须同时记录“cell 证据等级”和“运行协议”。实际可声称的证据强度取两者中较弱的一项。

| 标签 | 本项目中的严格含义 | 可以声称 | 不可以声称 |
|---|---|---|---|
| `paper-exact` | 论文披露的比较因子、模型族、支持集或 prompt 文本被逐项落实；带 `*` 时仍存在双卡硬件、未披露随机性或公开资源差异 | “复现了论文实验设计/机制趋势” | “与作者 8×A800 数值或权重逐位一致” |
| `released-checkpoint` | 从作者公开的中间/最终 checkpoint 开始比较，而不是重做其上游训练 | “公开 SFT 初始化对 OPD 的影响” | “端到端复现了 200K 生成与 SFT” |
| `proxy` | 使用公开镜像、本地确定性匹配子集，或用 `smoke/calibration/pilot` 缩放论文设置 | “工程链路/硬件适配/方向性机制证据” | “论文精确曲线或最终分数复现” |
| `local-4B-extension` | 把学生扩到 4B 的本地外推，论文没有这组消融 | “4B 学生的双步可运行性与有限指标” | “论文结论中的 4B 原实验” |

协议会进一步降级证据：`smoke` 和 `calibration` 永远是 proxy；`pilot` 是硬件适配后的机制 proxy；只有 `paper-exact` cell 使用 `paper` 协议时，才可写成“论文披露设置的 2×A100 端口”，仍须披露硬件差异、seed 和 checkpoint 规则。

注册表中的原始 fidelity 字段应保留在产物中。例如 Fig. 6 是 `paper_models_with_public_author_checkpoint_proxy`，Fig. 8 是 `released_checkpoint_comparison`，Fig. 10 是 `paper_models_with_disclosed_local_matching_seed`。不要在汇总表中把这些字段统一改名为 `paper-exact`。

## 2. 默认 24-cell 矩阵

除表中唯一变量外，同组 cell 保持模型、数据、Top-k 和支持集语义不变。默认支持权重均为 `SUPPORT_WEIGHT_NORMALIZATION=author`。

| # | 论文位置 | cell ID | 分组内唯一变量 | 证据等级 |
|---:|---|---|---|---|
| 1 | Fig. 2 | `fig2-compatible-grpo` | `MODEL_PAIR=paper`，Qwen GRPO teacher | `paper-exact*` |
| 2 | Fig. 2 | `fig2-mismatch-nonthinking` | `MODEL_PAIR=mismatch`，Qwen Non-thinking teacher | `paper-exact*` |
| 3 | Fig. 6 | `fig6-success-justrl` | `MODEL_PAIR=r1_success`，JustRL-1.5B teacher | `proxy`：公开作者 checkpoint 镜像 |
| 4 | Fig. 6 | `fig6-failure-r1-7b` | `MODEL_PAIR=r1_failure`，R1-Distill-7B teacher | `proxy`：与 cell 3 成对比较 |
| 5 | Fig. 7 | `fig7-student-topk` | `TOP_K_STRATEGY=only_stu` | `paper-exact*`：author-code 语义 |
| 6 | Fig. 7 | `fig7-overlap-topk` | `TOP_K_STRATEGY=intersection` | `paper-exact*`：author-code 语义 |
| 7 | Fig. 7 | `fig7-nonoverlap-topk-author` | `TOP_K_STRATEGY=union-intersection` | `paper-exact*`：author raw-mass 语义 |
| 8 | Fig. 8 | `fig8-base-only-opd` | Qwen3-1.7B-Base 初始化 | `released-checkpoint` 对照 |
| 9 | Fig. 8 | `fig8-sft-then-opd` | `lllyx/Qwen3-1.7B-SFT` 公布权重初始化 | `released-checkpoint` |
| 10 | Fig. 9 | `fig9-template-original` | 原始 DAPO prompt 模板 | `paper-exact*`：原始文本 |
| 11 | Fig. 9 | `fig9-template-paper-aligned` | 严格重建 `...within \boxed{}.` 模板 | `paper-exact*`：文本重建 |
| 12 | Fig. 10 | `fig10-content-dapo-matched` | 14,116 条 matched DAPO prompts | `proxy`：本地 seed 42 匹配子集 |
| 13 | Fig. 10 | `fig10-content-deepmath` | 14,116 条去重 DeepMath prompts | `proxy`：与 cell 12 等量 |
| 14 | Fig. 11(a)/12/13 | `fig12-length-512` | `MAX_RESPONSE_LENGTH=512`（0.5K） | `paper-exact*`；pilot 为 proxy |
| 15 | Fig. 11(a)/12/13 | `fig12-length-1024` | `MAX_RESPONSE_LENGTH=1024`（1K） | `paper-exact*`；pilot 为 proxy |
| 16 | Fig. 11(a)/12/13 | `fig12-length-3072` | `MAX_RESPONSE_LENGTH=3072`（3K） | `paper-exact*`；pilot 为 proxy |
| 17 | Fig. 11(a)/12/13 | `fig12-length-7168` | `MAX_RESPONSE_LENGTH=7168`（7K） | `paper-exact*`；pilot 为 proxy |
| 18 | Fig. 11(a)/12/13 | `fig12-length-10240` | `MAX_RESPONSE_LENGTH=10240`（10K） | `paper-exact*`；pilot 为 proxy |
| 19 | Fig. 11(a)/12/13 | `fig12-length-15360` | `MAX_RESPONSE_LENGTH=15360`（15K） | `paper-exact*`；pilot 为 proxy |
| 20 | Fig. 15/16 | `fig16-topk-sampled` | `TOP_K=0`，sampled-token | `paper-exact*`；pilot 为 proxy |
| 21 | Fig. 15/16 | `fig16-topk-1` | Student Top-1 | `paper-exact*`；pilot 为 proxy |
| 22 | Fig. 15/16 | `fig16-topk-4` | Student Top-4 | `paper-exact*`；pilot 为 proxy |
| 23 | Fig. 15/16 | `fig16-topk-16` | Student Top-16 | `paper-exact*`；pilot 为 proxy |
| 24 | Fig. 15/16 | `fig16-topk-64` | Student Top-64 | `paper-exact*`；pilot 为 proxy |

几个不能从 cell 名称直接看出的边界：

- Fig. 6 使用可访问的 `hbx/JustRL-DeepSeek-1.5B` 固定 revision；它与论文提及但受限的 thunlp 镜像是否 byte-identical 尚未验证，所以是 checkpoint proxy。
- Fig. 8 只比较 Base 与已发布 SFT 权重作为 OPD 起点。
- Fig. 9 的 paper-aligned 数据把文字严格重建为单层花括号字面量 `\boxed{}`；不要用发布 processed 文件中的 `\boxed{{}}` 冒充这一条件。
- Fig. 10 的论文没有公布 matched DAPO 的 row IDs 或 seed；本项目固定从 paper-aligned DAPO 生成 14,116 条 seed-42 子集，并与 DeepMath 等量比较。
- Fig. 11(b) 不在这 24 个 cell 中，边界见第 10 节。

## 3. 200 steps、260 steps 与一轮 279 updates

论文存在三个不能静默合并的口径：正文把主要训练描述为 200 steps；长度图延伸到约 250–260 steps；Table 2 又给出 `epoch=1`。对 17,917 条 DAPO 数据和 global batch 64，完整 batch 数为 `17917 // 64 = 279`。因此本地协议明确选择：

| 协议 | 普通组 | 长度组 | 解释 |
|---|---:|---:|---|
| `smoke` | 2 | 2 | 只验分支、shape、有限指标 |
| `calibration` | 10 | 10 | 只测吞吐、显存和 NaN/OOM |
| `pilot` | 200 | 260 | 长度组延长到 260，以观察论文图中的后期崩溃；step 200 单独标记 |
| `paper` | 不强制 step，上限由 1 epoch 决定 | 同左 | 预计 279 个完整 batch；最终以实际日志 step 为准 |

报告长度实验时至少给出 `step=200` 和最终 `step=260/epoch-end` 两个读数，不能只挑最有利 checkpoint。默认长度 pilot 每 20 step 保存、只保留 2 个 actor checkpoint，因此训练结束后 step 200 的 **metrics 仍在 JSONL**，但 step-200 checkpoint 通常已被 step 240/260 淘汰。`paper` 协议也只保留最近 2 个 checkpoint。若研究问题要求对 step 200 做完整 avg@16，必须在开跑前用一份有审计记录的自定义 matrix 提高 checkpoint retention；训练结束后不能由 metrics 反推权重。

## 4. 资源分级与执行顺序

| 等级 | 操作 | GPU/产物 | 结论上限 |
|---|---|---|---|
| R0 | `plan`、`status`、聚合、已有日志绘图 | CPU；不训练 | 配置与产物审计 |
| R1 | `smoke` / `calibration` | 独占 2×A100；两种协议均 `SAVE_FREQ=-1`，没有可评估 checkpoint | L1 工程链路、显存、有限值 |
| R2 | 单组 `pilot` | 独占 2×A100；普通组 200 step，长度组 260 step | 硬件适配的机制趋势 |
| R3 | 单 cell `paper` | 2×A100，多日风险，尚未完成全协议实机验证 | 最接近论文披露训练设置，但非 bitwise |
| R4 | 完整 avg@16 评测 | 独占 2×A100，三套 benchmark、每题 16 次、最长 31,744 tokens | checkpoint 的论文口径绝对分数 |

已知实机 smoke 证据：1.7B 学生到 4B 教师曾记录约 50.80 GiB reserved；4B 学生到 4B 教师曾记录约 71.38 GiB reserved。后者离 80GB 上限很近，所以 4B 只开放 `smoke`，不开放 `pilot/paper`。10K/15K 长度 cell 也应先逐个做 calibration。日志字段由训练进程内 allocator 提供，不应在没有 `nvidia-smi` 同步采样时直接解释为每卡物理占用；历史峰值也不构成未来运行不 OOM 的保证。

runner 要求 `CUDA_VISIBLE_DEVICES` 恰好包含两个互异的数字，并对该 GPU 集合加非阻塞独占锁。同一时刻不要让训练和评测争用这两张卡。

### 4.1 实机进度（2026-08-03，smoke/seed-42）

当前默认矩阵完成 **4/24** 个 smoke cell；这表示工程 L1 通过，不是论文机制曲线或最终分数已经复现。

| 组 | cell | 状态 | step-2 摘要 |
|---|---|---|---|
| Fig. 7 支持集 | `fig7-student-topk` | completed，attempt 1 | overlap `0.7072`，Eq. (7) `-0.02393`，Eq. (8) `0.2662`，grad norm `3.133` |
| Fig. 7 支持集 | `fig7-overlap-topk` | completed，attempt 1 | overlap `0.6981`，Eq. (7) `-0.02381`，Eq. (8) `0.2641`，grad norm `2.799` |
| Fig. 7 支持集 | `fig7-nonoverlap-topk-author` | completed，attempt 3 | training proxy `0`（交集被排除），grad norm `0.4366`；仅属两步诊断 |
| Fig. 11(a)/12/13 长度 | `fig12-length-15360` | completed，attempt 1 | mean length `15360`，overlap `0.6174`，Eq. (8) `0.4065`，grad norm `4.108` |

Fig. 7 三格的[汇总图](../artifacts/ablations/rethinking-opd-formal-ablations-v1/smoke/seed-42/plots/fig7_support.png)、15K 的[两步位置熵热图](../artifacts/ablations/rethinking-opd-formal-ablations-v1/smoke/seed-42/plots/fig12_length_15360_position_entropy.png)、[JSON](../artifacts/ablations/rethinking-opd-formal-ablations-v1/smoke/seed-42/results.json)与[CSV](../artifacts/ablations/rethinking-opd-formal-ablations-v1/smoke/seed-42/results.csv)已归档。15K step 1 的 allocator 计数为 allocated/reserved `69.78/79.35 GiB`；step 2 日志变为 `76.41/86.24 GiB`，其中 reserved 已超过单卡物理容量，不能解释为“每卡实际占用”。该运行正常完成，结束后的两卡均为 4 MiB/0% utilization，但 15K 仍应判定为极紧张配置，禁止直接迁移到 4B 学生。

`fig7-nonoverlap-topk-author` 的 attempt 1 是本地代理不可用导致的 Hugging Face 元数据预检失败，attempt 2 是人工中止的直连等待，attempt 3 才是有效完成运行；这些基础设施重试不得计入科学失败率。更早一次 venv symlink 预检失败完整保存在 `artifacts/failed_runs/fig7-smoke-preflight-venv-symlink-20260802/`，未覆盖正式产物。

## 5. 正式训练工作流

以下命令从仓库根目录执行。所有 `plan` 都会检查注册表、数据行数与 SHA-256、组内单变量约束、源代码 hash 和 cell fingerprint。

### 5.1 审计默认矩阵

```bash
cd /vepfs-mlp2/queue010/20262202674/OPD
.venv-opd/bin/python reproduce_4b/run_ablations.py plan --protocol smoke
```

首行必须显示 `cells=24`。先检查一个目标组的完整解析命令：

```bash
.venv-opd/bin/python reproduce_4b/run_ablations.py plan \
  --protocol smoke --group fig7_support
```

### 5.2 Smoke 与 calibration

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv-opd/bin/python reproduce_4b/run_ablations.py run \
  --protocol smoke --group fig7_support --yes

CUDA_VISIBLE_DEVICES=0,1 .venv-opd/bin/python reproduce_4b/run_ablations.py run \
  --protocol calibration --cell fig12-length-15360 --yes
```

`smoke` 通过不代表趋势正确，`calibration` 通过也不代表 260-step 不会后期 OOM/发散；它们都不会保存 actor checkpoint，不能交给 `evaluate_ablation.py`。

### 5.3 单组 pilot

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv-opd/bin/python reproduce_4b/run_ablations.py run \
  --protocol pilot --group fig15_16_topk --yes --keep-going

.venv-opd/bin/python reproduce_4b/run_ablations.py status \
  --protocol pilot --group fig15_16_topk
```

`--keep-going` 只表示某个 cell 失败后继续下一个 cell；失败状态不会被改写为成功。建议按 Fig. 2 → Fig. 6 → Fig. 7 → Fig. 9/10 → Top-k → 长度的顺序逐组推进，并在每组结束后先聚合、再决定是否投入下一组。

### 5.4 Paper 协议

先只 plan 一个 cell：

```bash
.venv-opd/bin/python reproduce_4b/run_ablations.py plan \
  --protocol paper --cell fig16-topk-16
```

确认两张卡空闲、磁盘与运行窗口足够后才执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv-opd/bin/python reproduce_4b/run_ablations.py run \
  --protocol paper --cell fig16-topk-16 \
  --yes --acknowledge-multi-day
```

不要用 `--protocol paper` 不加 `--group/--cell` 直接启动全部 24 个多日实验。

### 5.5 失败重试不是 resume

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv-opd/bin/python reproduce_4b/run_ablations.py run \
  --protocol pilot --cell fig16-topk-16 --yes --retry-failed
```

`--retry-failed` 会新建 `attempt-000N`，从注册表固定的初始学生权重重新训练，并在 status 中写入 `resumed: false`。它不会从失败 checkpoint 继续，也不会覆盖旧 attempt。即使 seed 相同，分布式 GPU kernel 和 rollout 仍不保证 bitwise replay。已完成 cell 会被跳过；fingerprint 或 suite manifest 不一致时 runner 会拒绝复用目录，此时应换新的 `--run-root`，而不是删除旧证据。

## 6. 支持集语义：`author` 与 `selected`

默认 24 个 cell 都使用 `author`：

- 对 `only_stu`、`intersection` 等常规支持集，在选定支持上重归一化。
- 对 `union-intersection`（Non-Overlap），复现发布作者代码的特殊语义：保留原始概率质量，不在非重合支持上再次归一化。该集合的总质量通常很小，因此 reward/梯度尺度也随之变小。
- 论文公式本身没有清楚披露这个 Non-Overlap raw-mass 特例，所以它是“author-code exact”，不能写成唯一可能的论文公式解释。

`selected` 会在所有选定支持上重归一化。它改变的不仅是 token 覆盖，还改变梯度尺度，属于默认关闭的稳健性扩展。只能在固定 `union-intersection` 后成对比较：

```bash
.venv-opd/bin/python reproduce_4b/run_ablations.py plan \
  --protocol smoke --include-extensions \
  --group fig7_normalization_robustness

CUDA_VISIBLE_DEVICES=0,1 .venv-opd/bin/python reproduce_4b/run_ablations.py run \
  --protocol smoke --include-extensions \
  --group fig7_normalization_robustness --yes
```

这两个 cell 必须标为 `local extension`，不能并入默认 Fig. 7 三条论文曲线。

## 7. Eq. (7)、Eq. (8) 与指标判读

令学生和教师 Top-k 交集为 \(I_t\)，并分别在 \(I_t\) 上重归一化得到 \(p_{I,t}\) 与 \(q_{I,t}\)。本项目按论文 Eq. (7) 记录：

\[
M_{\mathrm{adv}}=
\mathbb E_t\!\left[
\frac{1}{|I_t|}\sum_{v\in I_t}
p_{I,t}(v)\bigl(\log q_{I,t}(v)-\log p_{I,t}(v)\bigr)
\right].
\]

对应字段是 `val-topk/adv_intersection`，要求 `opd/metric_schema_version=2`。它是**独立诊断量**：成功 OPD 中通常从负值向 0 靠近。`val-topk/training_adv_intersection` 则是实际训练 reward 在交集上的 proxy；对 Non-Overlap 策略，交集不参与训练，这个 proxy 可为 0，不能解释成 Eq. (7) 已收敛。Sampled-token (`TOP_K=0`) 没有显式 Top-k 交集，缺少 Eq. (7)/overlap 字段不是错误。

论文 Eq. (8) 的本地严格聚合是：

\[
M_{\Delta H}=\mathbb E_{t\in\mathrm{valid}}
\left|H(q_t)-H(p_t)\right|.
\]

对应字段是 `opd/abs_entropy_gap`，要求 `opd/entropy_gap_schema_version=1`。它是先对每个学生访问位置取熵差绝对值，再对有效位置平均；`|mean teacher entropy - mean student entropy|` 只是旧日志/旧图的 proxy，两者不能混用。

正式判读至少联合查看：

- `val-topk/overlap_ratio`、Eq. (7)、Eq. (8)；
- `val-topk/student_p_sum_intersection` 与 `teacher_p_sum_intersection`，防止“交集 token 数上升但主要质量不在交集”的误读；
- `actor/entropy`、`teacher/entropy`、`actor/grad_norm`；
- `response_length/mean` 与显存字段，排除截断或资源变化造成的假趋势；
- 三项 benchmark 的 avg@16，而不是从训练 proxy 推断最终正确率。

## 8. Checkpoint 评测：论文口径是 avg@16

`evaluate_ablation.py` 只接收某个成功 attempt 的 `run_dir`，自动发现/合并 FSDP actor checkpoint，并评估 AIME24、AIME25、AMC23。默认及论文口径为：每题 `n=16`、temperature 0.7、top-p 0.95、最大回复 31,744 tokens、thinking off、seed 42；三项总分是三个 benchmark `avg@16` 的非加权平均，**不是 pass@16**。

先从 cell status 读取真实 attempt 路径。以下以 pilot Top-16 为例：

```bash
ABLATION_SUITE_ROOT="$PWD/artifacts/ablations/rethinking-opd-formal-ablations-v1/pilot/seed-42"
CELL_RUN_DIR="$(.venv-opd/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["run_dir"])' \
  "$ABLATION_SUITE_ROOT/fig16-topk-16/status.json")"

.venv-opd/bin/python reproduce_4b/evaluate_ablation.py plan \
  --run-dir "$CELL_RUN_DIR"
```

未传 `--checkpoint-step` 时选择当前仍保留的最新 checkpoint。快速链路评测必须显式限制 prompts，并标为 proxy：

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv-opd/bin/python reproduce_4b/evaluate_ablation.py run \
  --run-dir "$CELL_RUN_DIR" --limit 2 --yes
```

完整 avg@16：

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv-opd/bin/python reproduce_4b/evaluate_ablation.py run \
  --run-dir "$CELL_RUN_DIR" \
  --n 16 --max-tokens 31744 --seed 42 \
  --yes --acknowledge-full-eval
```

如果所需 checkpoint 仍存在，可重复指定一个或多个 step：

```bash
.venv-opd/bin/python reproduce_4b/evaluate_ablation.py plan \
  --run-dir "$CELL_RUN_DIR" \
  --checkpoint-step 200 --checkpoint-step 260
```

评测器会核对每个 prompt 恰有 `n` 个不同 rollout ID，并检查 sampling metadata。已有但不完整的 generation 默认报错，先人工检查后才可使用 `--overwrite-incomplete`。任何 `--limit`、`n!=16`、较短 `max-tokens` 或只测单个 benchmark 的结果都不是论文 avg@16。当前 formal evaluator 只评 cell checkpoint；它不会自动补跑原始学生和教师 baseline，因此只凭一个 final checkpoint 的分数不能声称完成端到端分数复现。

## 9. 聚合与位置熵图

每个 suite root 固定一个协议和 seed。聚合器读取 manifest、status、最后一条 metrics、最新 evaluation summary，生成 `results.json`、`results.csv` 和每组对比图：

```bash
.venv-opd/bin/python reproduce_4b/aggregate_ablations.py \
  --suite-root "$ABLATION_SUITE_ROOT"
```

只有至少两个已完成且有 metrics 的同组 cell 才会生成组图。不要把不同 protocol、seed 或 fidelity 的目录手工拼成一张无标注曲线。若补做 seed 43/44，应分别运行独立 suite，并在外部统计时保留 seed，而不是只展示最优 seed。

Fig. 13 的 suffix→prefix 熵异常使用长度 pilot 的分位置日志。15K cell 在 pilot 中从 step 180 起每 10 step、按 256-token bin 记录。示例：

```bash
LONG_RUN_DIR="$(.venv-opd/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["run_dir"])' \
  "$ABLATION_SUITE_ROOT/fig12-length-15360/status.json")"

.venv-opd/bin/python reproduce_4b/plot_position_entropy.py \
  --input-jsonl "$LONG_RUN_DIR/metrics.jsonl" \
  --output "$LONG_RUN_DIR/position-entropy-step180-250.png" \
  --metric both --min-step 180 --max-step 250 \
  --title "15K response: positional entropy"
```

若对应 step 没有位置熵数组，绘图失败应视为日志缺失，不能用全序列平均熵伪造热图。

## 10. 当前没有实现的论文边界

### Fig. 11(b) 教师续写探针

论文的 Fig. 11(b) 不是普通 checkpoint avg@16：作者从约 2K prompts 的长 rollout 中筛选超过 16K 的样本，在 1K/4K/8K/16K prefix 处截断，再让教师续写并评分。当前 24-cell 长度组只实现 Fig. 11(a)、Fig. 12/13 的最大回复长度与位置熵训练诊断；`evaluate_ablation.py` 生成的是学生完整回答，**没有实现教师从学生 prefix continuation 的采样、配对和评分**。因此不能用长度 heatmap 或 final avg@16 代替 Fig. 11(b)。

### Fig. 8 端到端 SFT cold start

论文流程包含 Qwen3-4B Non-thinking 生成约 200K 条 OpenThoughts3 数学回答、过滤/整理、对 Qwen3-1.7B-Base 全参数 SFT，再在去重后的约 30K prompts 上 OPD。当前 Fig. 8 直接使用发布的 `lllyx/Qwen3-1.7B-SFT` revision，只实现“released checkpoint vs Base 的 OPD 起点比较”。200K rollout 生成、过滤和完整 SFT 没有纳入 formal ablation runner；而且论文、模型卡和仓库 SFT YAML 的超参数披露并不完全一致。结果必须标为 `released-checkpoint`，不能写成端到端 SFT 复现。

## 11. 本地 4B 扩展

4B 组默认关闭、只允许 `smoke`，包含 1.7B 与 4B 学生两个对照 cell。显式选择该组，避免裸用 `--include-extensions` 一次把默认 24 个和 4 个扩展 cell 全选中：

```bash
.venv-opd/bin/python reproduce_4b/run_ablations.py plan \
  --protocol smoke --include-extensions \
  --group local_4b_scale_transfer

CUDA_VISIBLE_DEVICES=0,1 .venv-opd/bin/python reproduce_4b/run_ablations.py run \
  --protocol smoke --include-extensions \
  --group local_4b_scale_transfer --yes
```

该组只验证 4B 学生在相同 Student Top-16、LR、temperature、数据和 teacher 条件下能否完成两步、指标是否有限，以及资源是否仍有余量。两步曲线不能判断模型质量、机制趋势或规模规律。由于既有 4B smoke 已 reserved 71.38 GiB/卡，任何 4B pilot/paper 都需要新的显存设计和授权，不在当前矩阵支持范围内。

## 12. 验收与结果记录

每个结论按三层验收：

1. **L1 工程通过**：runner state 为 `completed`，最后 step 符合协议，loss/entropy/grad/memory 为有限值；若需评测，pilot/paper checkpoint 能合并和生成。
2. **L2 机制复现**：预先指定整段曲线；成功 pair 相比失败 pair 呈现 overlap 上升、Eq. (7) 向 0、Eq. (8) 收窄；Overlap Top-16 接近 Student Top-16、Non-Overlap 更弱；Top-1 更不稳定；10K/15K 的后期异常与位置传播得到日志支持。单点、smoke 或 cherry-picked checkpoint 不够。
3. **L3 数值复现**：使用可审计的 `paper` 运行，报告训练 step/checkpoint 选择，完成三项完整 avg@16，并同时给 baseline/final/teacher。仍须披露 2×A100 对 8×A800、缺失随机种子/优化器细节等差异。

每个结果至少保存：suite/cell/attempt ID、cell fidelity、protocol、完整命令、git commit 与 diff、fingerprint/source hash、模型 revision/snapshot、数据 hash、seed、GPU 型号、metrics schema、完整 JSONL、checkpoint step、评测 sampling metadata、三项 avg@N、是否使用 `--limit`，以及失败/重试历史。`status.json` 中的 fresh attempt、`results.json/csv` 中的 fidelity 和 evaluation step 都应随图表一起归档。

论文 Fig. 15 的 R1-Distill→JustRL 参考分数为 Sampled `0.454/0.327/0.782`、Top-1 `0.446/0.310/0.772`、Top-4 `0.473/0.331/0.793`、Top-16 `0.458/0.338/0.791`、Top-64 `0.463/0.338/0.785`（AIME24/AIME25/AMC23）。这些值只适用于该论文 pair；不能拿来当 Fig. 2 的 Qwen pair、Fig. 8 的 SFT pair 或本地 4B 扩展的目标分数。
