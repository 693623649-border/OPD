# Rethinking OPD：完整论文复现总入口

本文是论文 *Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe* 的统一执行入口。原文见[本地 PDF](../paper/rethinking_opd.pdf)，中文方法与公式解读见[论文精读](../paper/rethinking_opd_summary_zh.md)。PDF 共 30 个物理页，且本版本的物理页与页脚页码一致；本文仍统一写“物理页”，避免更换 PDF 版本后混淆。

机器可执行定义以[完整 v3 训练矩阵](./paper_full_matrix.json)、[23 图 + 3 表 producer 台账](./paper_experiment_ledger.json)和[论文原始数值摘录](./paper_reported_results.json)为准。当前矩阵 SHA-256 为 `a45bc9aff21bc6c12a687660cd5d5953ce022f924f9c5fec7855f23c59818225`。历史 24-cell 文档仍在[消融指南](./ABLATIONS_zh.md)，但新实验应优先使用本页和 v3 矩阵。

> 当前交付的是“完整结构覆盖、可审计实现和分级运行入口”，不是“论文数值已经全部跑完”。截至 2026-08-03，本地动态台账为：结构覆盖 26/26，31 个注册 OPD cell role 中 27 个至少完成一种 smoke，论文可比结果 0/26，硬阻塞 3/26；renderer 已生成 15/23 张本地诊断图。以[动态覆盖报告](../artifacts/paper_reproduction/coverage/coverage.md)和[渲染 manifest](../artifacts/paper_reproduction/figures/render_manifest.json)为最新事实来源。任何两步 smoke、单题 `avg@1`、4-prompt rollout 或两步 SFT 都不得用于论文科学结论。

## 1. 三个必须分开的完成层级

| 层级 | 判定条件 | 当前状态 | 可以声称什么 |
|---|---|---:|---|
| 实现覆盖 | 23 张图、3 张表都有 producer；31 个注册 cell role（归一化为 29 个完整 env、28 个科学训练条件）以及上游训练、评测、probe、collector、coverage、renderer 均有明确入口或明确硬阻塞 | 26/26 条目已登记 | “论文实验结构已完整映射到可审计工程入口” |
| 工程 smoke | 固定模型/数据 revision，完成极短训练或生成，保存 manifest、fingerprint、日志和状态 | 27/31 个注册 cell role 至少有一个 smoke；两个 Figure 3 teacher 和 Table 1/3 上游链路也有缩短产物 | “该分支在当前机器上通过了有限链路检查” |
| 论文可比数值 | 使用 `paper` 训练口径、保留目标 checkpoint，并按三项 benchmark、每题 16 次、31,744 tokens 完成 `avg@16`；还必须披露硬件和所有本地重建项 | 0/26 | 才能比较论文报告值；仍不等于 bitwise reproduction |

证据标签统一如下：

- `P`：论文直接报告的文字、公式或数值，并给出物理页；只代表作者报告。
- `R`：公开、固定 revision 的模型或数据资源；只证明资源身份。
- `I`：实现和单元测试覆盖；不代表模型训练已经发生。
- `S`：本地 smoke/calibration 产物；只证明工程链路、形状、显存或有限值。
- `C`：本地论文可比运行，即 `paper` 协议加完整 `avg@16`；当前没有任何 `C` 结果。

实际结论强度取最弱证据。比如“公开 checkpoint + 两步 smoke”只能标 `R+S`，不能因模型身份精确而升级为 `C`。

## 2. v1/v2 历史证据与 v3 当前矩阵

| 项目 | v1 历史入口 | v2 不可变 smoke 证据 | v3 当前可执行入口 |
|---|---|---|---|
| 注册表/manifest | [`ablation_matrix.json`](./ablation_matrix.json) | [v2 suite manifest](../artifacts/ablations/rethinking-opd-complete-paper-v2/smoke/seed-42/suite_manifest.json) | [`paper_full_matrix.json`](./paper_full_matrix.json)与[`paper_experiment_ledger.json`](./paper_experiment_ledger.json) |
| suite ID | `rethinking-opd-formal-ablations-v1` | `rethinking-opd-complete-paper-v2` | `rethinking-opd-complete-paper-v3` |
| 范围 | 24 个默认核心 cell；另有 4 个默认关闭的本地扩展 | 27 个默认 cell 均完成 2 steps、exit 0 | 31 个注册 role、29 个完整 env、28 个科学条件，覆盖 11 组 |
| 默认/门禁 | 默认 24；4 个论文外扩展关闭 | 默认 27/27 已完成 | 默认 27；2 个 RL-Math cell 与 2 个 Figure 20 cell 关闭 |
| 用途 | 早期实现与 4B 扩展历史 | 完整默认分支的工程 smoke 证据 | 所有未来训练、paper checkpoint、评测注册和审计 |

`rethinking-opd-formal-ablations-v1`、full-matrix 草案 v1 和完整 v2 smoke 都必须原样保留。v3 runner 以 suite ID、真实 matrix SHA、源代码哈希和 cell fingerprint 隔离新运行；不能把 v2 目录改名伪装成 v3。v3 的代表性 [`fig6-deepseek-justrl-success` smoke](../artifacts/ablations/rethinking-opd-complete-paper-v3/smoke/seed-42/fig6-deepseek-justrl-success/status.json)也已完成 2 steps、exit 0，用于验证当前控制面与训练入口。collector 和 renderer 可以同时读取多个 suite，但输出必须保留 suite、protocol、seed、matrix/source identity 和 fingerprint。

本机 4B 学生迁移仍作为 v1 的论文外扩展保留，不计入这 31 个注册 cell role：

```bash
cd /vepfs-mlp2/queue010/20262202674/OPD
PY="$PWD/.venv-opd/bin/python"

"$PY" reproduce_4b/run_ablations.py plan \
  --matrix reproduce_4b/ablation_matrix.json \
  --protocol smoke --include-extensions \
  --group local_4b_scale_transfer
```

只有确认两张 80GB GPU 完全空闲后，才可在新的 run root 运行；这仍是 `local-4B-extension/S`，不是论文原实验：

```bash
export OPD_ACK_LOCAL_4B_SMOKE=YES
test "$OPD_ACK_LOCAL_4B_SMOKE" = YES || exit 2

CUDA_VISIBLE_DEVICES=0,1 "$PY" reproduce_4b/run_ablations.py run \
  --matrix reproduce_4b/ablation_matrix.json \
  --protocol smoke --include-extensions \
  --group local_4b_scale_transfer \
  --run-root artifacts/ablations-local-extensions \
  --yes --keep-going
```

## 3. 完整 31-cell 训练矩阵

v3 包含 11 组、31 个注册 OPD cell role；默认调度 27 个，4 个有显式门禁。31 是矩阵行/论文角色数，不能称作 31 条唯一科学轨迹：按完整 launcher environment 去重后是 29 个 env，再移除 `MIN_FREE_GIB` 等纯运维字段后是 28 个科学训练条件。完整逐 cell 模型、数据、长度、支持集和预算见[自动生成训练矩阵](../artifacts/paper_reproduction/coverage/training_matrix.md)。

| 组 | 数量 | 默认 | 主要比较 | 论文位置与证据边界 |
|---|---:|---:|---|---|
| `fig2_teacher_pattern` | 2 | 2 | Qwen GRPO-compatible teacher vs Non-thinking teacher | Fig. 2/17，物理页 6/23；公开 revision，双卡协议为硬件适配 |
| `fig4_6_deepseek_teachers` | 3 | 3 | Skywork-RL、R1-7B、JustRL-1.5B 教师 | Fig. 4/6/14/18/19，物理页 7/9/16/24/25；使用可访问的 `hbx` JustRL 镜像，但与受限 `thunlp` revision 的 byte identity 未独立验证 |
| `fig4_qwen_new_knowledge` | 2 | 0 | Qwen Non-thinking vs RL-Math teacher | Fig. 4，物理页 7；整组默认禁用，因为精确 RL-Math checkpoint 未公开 |
| `fig5_reverse_distillation` | 2 | 2 | JustRL-1.5B 学生反向蒸馏到 R1-1.5B/R1-7B | Fig. 5，物理页 8；`paper` 独立覆盖为 600 steps |
| `fig7_support` | 3 | 3 | Student Top-16、Overlap Top-16、Non-overlap Top-16 | Fig. 7，物理页 10；Non-overlap 保留作者 raw-mass 语义 |
| `fig8_cold_start` | 2 | 2 | Base→OPD vs 已发布 SFT→OPD | Fig. 8/21，物理页 12/28；重做 SFT 的上游另见 Table 3 |
| `fig9_prompt_template` | 2 | 2 | 原 DAPO prompt vs teacher-aligned prompt | Fig. 9/22，物理页 13/29 |
| `fig10_prompt_content` | 2 | 2 | matched DAPO vs 去重 DeepMath | Fig. 10，物理页 13；本地固定 seed 42，但作者未公开 row IDs/seed |
| `fig11_13_response_length` | 6 | 6 | 0.5K/1K/3K/7K/10K/15K | Fig. 11(a)/12/13/23，物理页 14/15/30；正文 200 steps 与图约 250–260 steps 的冲突须分别报告 |
| `fig15_16_topk` | 5 | 5 | Fig. 15：sampled-token、Top-1/4/16/64；Fig. 16：仅 Top-1/4/16/64 | Fig. 15/16，物理页 17 |
| `fig20_cross_model_large` | 2 | 0 | R1-7B 学生→Skywork-7B/R1-14B | Fig. 20，物理页 27；仅允许 `paper`，要求 8 张 80GB GPU，本机不调度 |
| **总计** | **31** | **27** |  | **默认关闭 4** |

`fig4_qwen_new_knowledge` 中两个 cell 同组关闭，是因为组内比较必须成对：公开的普通 Qwen teacher 不能单独代替缺失的 RL-Math 对照。`fig20_cross_model_large` 两个 cell 则因 7B 学生和 7B/14B 教师的原尺度资源需求而关闭。

所有使用 `hbx/JustRL-DeepSeek-1.5B@0637e4...` 的结果必须标作“可访问镜像”。论文关联的 `thunlp/JustRL-DeepSeek-1.5B@150339...` 受限，本地没有独立证明二者权重逐字节相同；固定 `hbx` revision 解决的是本地身份可追踪性，不等于恢复作者端精确 checkpoint。

## 4. 23 图 + 3 表 producer 台账

下面的“状态快照”只描述 2026-08-03 已观察到的本地产物；运行后以[动态覆盖工具](./audit_paper_coverage.py)输出为准。`I` 表示 producer 已实现，`S` 表示至少有工程 smoke，`C` 表示论文可比数值。

| 条目（物理页） | Producer | 当前状态/门禁 |
|---|---|---|
| Fig. 1（p.1） | Fig. 2/4/5/6/7 共 12 个组成 cell 的论文总览复用 | 10/12 组成 cell 有 `S`；两个 RL-Math cell 缺权重，作者 composite 仍阻塞 |
| Fig. 2（p.6） | `fig2-compatible-grpo`、`fig2-mismatch-nonthinking`；训练曲线 + 三基准非加权 `avg@16` | 两格均有两步 `S`；没有 paper 曲线或 `avg@16` |
| Fig. 3（p.6） | 两个固定 4B teacher 的 AIME24/AIME25/AMC23 `avg@16`，不训练 | 两个 teacher 均有三基准 `I+S`；仅 `n=1/max_tokens=256/avg@1=0`，不是论文分数 |
| Fig. 4（p.7） | DeepSeek 两 teacher + Qwen 两 teacher；初始 overlap、最终分数和 gap recovery | DeepSeek 两格有两步 `S`；Qwen RL-Math checkpoint 未公开，整体仍阻塞 |
| Fig. 5（p.8） | `fig5-reverse-r1-1p5b`、`fig5-reverse-r1-7b` 的 600-step 分 benchmark 轨迹 | v2 两格均完成 attempt 1、2 steps、exit 0 的 `S`；尚无 600-step 曲线或 `avg@16` |
| Fig. 6（p.9） | `fig6-deepseek-justrl-success`、`fig4-6-deepseek-r1-7b`；分数、overlap、Eq. (7)、Eq. (8) | 两格均有两步 `S`；其中 JustRL 格另有 v3 代表 smoke；没有 paper 曲线或 `avg@16` |
| Fig. 7（p.10） | `fig7-student-topk`、`fig7-overlap-topk`、`fig7-nonoverlap-topk-author` | 三格均有两步 `S`；没有 200-step/paper 曲线或 `avg@16` |
| Fig. 8（p.12） | `fig8-base-only-opd`、`fig8-sft-then-opd` | 两格均有两步 `S`；公开 SFT checkpoint 可用，上游缩短重建另见 Table 3 |
| Fig. 9（p.13） | `fig9-template-original`、`fig9-template-paper-aligned` | 两格均有两步 `S`；没有 paper 曲线或完整评测 |
| Fig. 10（p.13） | `fig10-content-dapo-matched`、`fig10-content-deepmath` | 两格均有两步 `S`；本地匹配数据已固定，但 row identity 不是作者精确数据 |
| Fig. 11(a)（p.14） | 六个 response-length cell 的最终 `avg@16` | 六格均有两步 `S`；没有 paper `avg@16` |
| Fig. 11(b)（p.14） | 严格 2K prompts、学生完整 rollout、筛选 `>16K`，按 1K/4K/8K/16K prefix 配对教师续写 | probe `I`；科学 workload 未运行，采样/checkpoint 多项未披露 |
| Fig. 12（p.15） | 六个长度 cell 的 overlap、学生熵、gradient norm | 六格均有两步 `S`；不能替代 200-step/图示后期窗口 |
| Fig. 13（p.15） | 15K cell 的 student position-entropy heatmap | 两步分箱 `S`；不是论文后期传播趋势 |
| Fig. 14（p.16） | 同一固定 rollout batch 分别由 JustRL-1.5B、R1-7B 评分；sequence-mean reward + tie-safe AUROC | 两个依赖训练格有 `S`，analyzer/scorer 为 `I`；精确 2,828/1,451 batch 未公开且科学 probe 未运行 |
| Fig. 15（p.17） | sampled/Top-1/4/16/64 的三基准最终 `avg@16` bar | 五格均有两步训练 `S`；没有 `avg@16` |
| Fig. 16（p.17） | Top-1/4/16/64 四个 cell 的 overlap、entropy、gradient norm；不含 sampled-token | 四格均有两步诊断 `S` |
| Fig. 17（p.23） | 复用 Fig. 2 两个 checkpoint 的三基准 breakdown | 两个训练格有 `S`；仍缺完整评测 |
| Fig. 18（p.24） | 复用 Fig. 6 两格的 student/teacher overlap probability mass | 两格均有两步指标 `S` |
| Fig. 19（p.25） | 复用 Fig. 6：policy loss、gradient norm、最大绝对 advantage token 上的 `p_s-p_t` | 两格均有 schema-v1 两步 `S`；旧 v1 日志仍不可追溯重建 |
| Fig. 20（p.27） | `fig20-r1-7b-skywork-7b`、`fig20-r1-7b-r1-14b` | 原模型 revision 已登记；2×A100 硬阻塞，8×80GB 专用门禁 |
| Fig. 21（p.28） | 复用 Fig. 8 的双方 overlap probability mass | 两格均有两步指标 `S` |
| Fig. 22（p.29） | 复用 Fig. 9 的三基准 breakdown | 两个训练格有 `S`；仍缺完整评测 |
| Fig. 23（p.30） | 复用 15K cell 的 teacher position-entropy heatmap | 两步分箱 `S`；缺完整后期窗口 |
| Table 1（p.23） | Qwen3-4B-Base 经 processed DAPO 一轮 GRPO 得到 teacher | `I/R+S`；upstream v1 smoke 已完成 2 steps 并保存 step-2 checkpoint；精确上游仍需 8×A800 且缺优化器/seed 信息 |
| Table 2（p.24） | OPD 默认训练合同，由 v3 registry 和 launcher 生成 | `I`；所有论文披露字段已编码，未披露字段不能猜作作者设置 |
| Table 3（p.28） | 200K teacher rollout、过滤、Qwen3-1.7B-Base 全参数 SFT | `I/R+S`；upstream v3 已完成 4-prompt rollout→过滤→两步全参数 SFT 与模型保存；精确 200K corpus identity 仍不可恢复 |

论文直接可读的 Fig. 3/4/11(b)/14/15/18 和 Table 1–3 数值单独保存在[`paper_reported_results.json`](./paper_reported_results.json)。renderer 不会用这些作者数值填充缺失的本地曲线；它们只能作为对照目标。

## 5. 运行前门禁

以下命令均从仓库根目录执行：

```bash
cd /vepfs-mlp2/queue010/20262202674/OPD
PY="$PWD/.venv-opd/bin/python"
MATRIX="$PWD/reproduce_4b/paper_full_matrix.json"
EXPECTED_MATRIX_SHA256=a45bc9aff21bc6c12a687660cd5d5953ce022f924f9c5fec7855f23c59818225

test -x "$PY"
test -f "$MATRIX"
test "$(sha256sum "$MATRIX" | awk '{print $1}')" = "$EXPECTED_MATRIX_SHA256"
nvidia-smi
ps -ef | grep -E '[r]ay|[v]llm|verl.trainer.main_ppo' || true
```

只有确认目标 GPU 上没有别人的进程后才运行。不要通过 `pkill`、`ray stop --force` 或删除 lock 文件清理不属于本实验的作业。v3 runner 先从所选 cell 的 `N_GPUS` 解析统一的 2 卡或 8 卡 allocation；混合 GPU-count selection、可见卡数量不符或重复物理卡编号都会在创建 suite/status/lock 或启动 subprocess 前 fail closed。通过后，runner 对每张可见物理卡分别持有非阻塞独占锁，因而重叠的 2/8 卡选择也互斥。已完成 cell 会跳过；训练失败后必须显式传 `--retry-failed`，并从固定初始权重新建 attempt，而不是覆盖或 resume 旧 attempt。

训练与评测的恢复语义不同。训练永远创建 fresh attempt；checkpoint 评测则由 write-once evaluation plan 与 target manifest 固定完整 grid 和 sampling identity，相同合同重跑时只补缺失 benchmark。paper checkpoint 评测禁止覆盖不完整文件；模型、tokenizer、grid、`n`、长度、seed 或 limit 任一改变都必须使用新的 evaluation identity。

先执行 CPU 验证：

```bash
"$PY" -m unittest discover -s reproduce_4b/tests -p 'test_*.py'

"$PY" reproduce_4b/run_ablations.py plan \
  --matrix "$MATRIX" --protocol smoke
```

默认 plan 的首行必须是 `cells=27`。查看全部 31 个结构 cell 只能使用不会加载模型的 paper plan：

```bash
"$PY" reproduce_4b/run_ablations.py plan \
  --matrix "$MATRIX" --protocol paper --include-extensions
```

此命令显示 31 不等于 31 个 role 均可运行：RL-Math sentinel 没有权重，Figure 20 要求 8 卡。不要对未替换的 `UNPUBLISHED/...` 模型调用 `run`。另外，显式 `--group`/`--cell` 会被视为人工选中，可能绕过“默认关闭”筛选，因此人工选择本身也是一次授权动作。

## 6. 2×A100：smoke → calibration → pilot

三种 2×A100 工程协议的科学上限如下；危险的 `paper` lane 单列在下一节：

| 协议 | 典型预算 | checkpoint | 用途 | 结论上限 |
|---|---:|---|---|---|
| `smoke` | 2 steps | 默认不保存 actor | 分支、shape、有限值、显存 | 工程 `S` |
| `calibration` | 10 steps | 默认不保存 actor | 吞吐、OOM、NaN 预警 | 工程 `S` |
| `pilot` | 普通组 200；长度组 260 | 每 20 steps，默认仅保留 2 个 | 本机趋势；长度组同时观察 step 200 与后期窗口 | 硬件适配趋势，不是论文绝对分数 |

先逐 cell smoke，尤其是 7B teacher 和 10K/15K 长序列。示例：

```bash
CUDA_VISIBLE_DEVICES=0,1 "$PY" reproduce_4b/run_ablations.py run \
  --matrix "$MATRIX" --protocol smoke \
  --cell fig5-reverse-r1-1p5b --yes

CUDA_VISIBLE_DEVICES=0,1 "$PY" reproduce_4b/run_ablations.py run \
  --matrix "$MATRIX" --protocol calibration \
  --cell fig12-length-15360 --yes
```

只有逐格验证、磁盘和下载预算都通过后，才运行默认 27-cell smoke。下面的环境变量是人为门禁，不由脚本自动设置：

```bash
export OPD_ACK_RUN_27_SMOKE=YES
test "$OPD_ACK_RUN_27_SMOKE" = YES || exit 2

CUDA_VISIBLE_DEVICES=0,1 "$PY" reproduce_4b/run_ablations.py run \
  --matrix "$MATRIX" --protocol smoke \
  --yes --keep-going
```

状态矩阵：

```bash
"$PY" reproduce_4b/run_ablations.py status \
  --matrix "$MATRIX" --protocol smoke
```

对 smoke 全部通过的组再做 calibration；失败 cell 不应静默跳过：

```bash
export OPD_ACK_RUN_27_CALIBRATION=YES
test "$OPD_ACK_RUN_27_CALIBRATION" = YES || exit 2

CUDA_VISIBLE_DEVICES=0,1 "$PY" reproduce_4b/run_ablations.py run \
  --matrix "$MATRIX" --protocol calibration \
  --yes --keep-going
```

pilot 应按组推进并在每组结束后评审日志。以下循环覆盖 9 个默认组，不包含两个 gated 组：

```bash
export OPD_ACK_RUN_PILOTS=YES
test "$OPD_ACK_RUN_PILOTS" = YES || exit 2

for group in \
  fig2_teacher_pattern \
  fig4_6_deepseek_teachers \
  fig5_reverse_distillation \
  fig7_support \
  fig8_cold_start \
  fig9_prompt_template \
  fig10_prompt_content \
  fig11_13_response_length \
  fig15_16_topk
do
  CUDA_VISIBLE_DEVICES=0,1 "$PY" reproduce_4b/run_ablations.py run \
    --matrix "$MATRIX" --protocol pilot \
    --group "$group" --yes --keep-going || exit $?
done
```

长度 pilot 将 position entropy 以 256-token bin 记录。这是本机显存适配。论文正文明确写 200 steps，而 Fig. 12/13/23 的横轴延伸到约 250–260；v3 保留双证据口径：`paper` 跟随正文 200，2×A100 length pilot 跑到 260 以观察图示窗口。任何报告都必须分别标出文本条件与图示后期窗口，不能合并成一个“作者默认步数”或只挑最好 checkpoint。

### R1-7B padded vocabulary 差异：已版本化解除门禁

旧草案 v1 的 `fig5-reverse-r1-7b` 曾因只比较 `config.vocab_size` 而在训练前失败；该[旧 preflight](../artifacts/ablations/rethinking-opd-complete-paper-v1/smoke/seed-42/fig5-reverse-r1-7b/attempt-0001/preflight.log)仅作为历史工程证据保留，不能代表当前实现的可运行性。

当前预检不再把 padded output head 尺寸相等误作 tokenizer identity。它先逐 ID 验证完整 tokenizer 映射，再检查实际最大 token ID 和当前 support 方向：两侧 tokenizer 均有 151,665 个映射项，最大真实 ID 为 151,664；JustRL-1.5B/R1-7B output heads 分别为 151,936/152,064。对本 cell 的 `TOP_K_STRATEGY=only_stu`，所有 student-support ID 都能被 teacher head 安全索引，因此允许 padded head 尺寸不同，同时打印 warning 并保存审计值。执行证据见[v2 R1-7B preflight](../artifacts/ablations/rethinking-opd-complete-paper-v2/smoke/seed-42/fig5-reverse-r1-7b/attempt-0001/preflight.log)。

这不是无条件放宽：若策略改为依赖 teacher-only support，或者 tokenizer 任一 ID 映射、最大真实 ID、支持方向安全性不满足，preflight 仍须 fail closed。v2 执行证据中的 1.5B 和 7B 反向蒸馏格均在 attempt 1 完成 2 steps、exit 0；它们只证明该安全规则和训练链路可执行，不证明 600-step 论文趋势。

## 7. v3 `paper`：2×A100 适配与 8×80GB Figure 20 lane

原论文 Table 1/2 披露的硬件为 8×A800-80GB。v3 runner 现在支持由 matrix 驱动的 2 卡或 8 卡 allocation，但这不改变证据边界：普通 OPD 组的当前 `paper` lane 仍是 2×A100 hardware adaptation；只有 Figure 20 组显式登记 `N_GPUS=8`、每卡至少 70 GiB 空闲，并且仍需独占 8×80GB 节点。先 plan 一个普通 paper cell：

```bash
"$PY" reproduce_4b/run_ablations.py plan \
  --matrix "$MATRIX" --protocol paper \
  --cell fig2-compatible-grpo

CUDA_VISIBLE_DEVICES=0,1 "$PY" reproduce_4b/run_ablations.py run \
  --matrix "$MATRIX" --protocol paper \
  --cell fig2-compatible-grpo \
  --yes --acknowledge-multi-day
```

`paper` 协议每 20 steps 保存一次 actor **model-only** shard，并设置 `MAX_ACTOR_CKPTS_TO_KEEP=0`；在当前 launcher 中 `0` 明确表示不删除任何 scheduled model checkpoint。这样可为预注册 checkpoint grid 保留权重，同时避免为每个时间点复制 optimizer/extra state。代价是这些 model-only checkpoint 用于 merge/evaluation，不构成可恢复优化器状态；训练失败仍应创建 fresh attempt。

Figure 20 必须单独选择，不能与 2 卡 cell 混在同一次 invocation。先确认节点身份并查看两格 plan：

```bash
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
NON_A800="$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -vc 'A800' || true)"
test "$GPU_COUNT" -eq 8 && test "$NON_A800" -eq 0 || {
  echo 'This lane requires exactly 8 visible A800 GPUs.' >&2
  exit 2
}

"$PY" reproduce_4b/run_ablations.py plan \
  --matrix "$MATRIX" --protocol paper --include-extensions \
  --group fig20_cross_model_large
```

确认 plan 中 7B student、7B/14B teacher 的固定 revision、`N_GPUS=8` 和 paper 合同后，再显式授权：

```bash
export OPD_ACK_8XA800_MULTI_DAY=YES
test "$OPD_ACK_8XA800_MULTI_DAY" = YES || exit 2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
"$PY" reproduce_4b/run_ablations.py run \
  --matrix "$MATRIX" --protocol paper --include-extensions \
  --group fig20_cross_model_large \
  --yes --acknowledge-multi-day --keep-going
```

该路径会自动写入 v3 suite manifest、matrix/source hashes、逐 cell fingerprint、模型 snapshot、resolved command、status 和不可覆盖 attempt。若选择同时含 2 卡与 8 卡的 cell，runner 会在任何产物写入前拒绝；不得绕过该检查或把手工目录移动进 v3 suite 冒充 runner 产物。当前本机没有执行 Figure 20 的 8 卡授权，因此它仍是硬阻塞而非失败实验。

## 8. 上游 Table 1：GRPO teacher

Table 1（物理页 23）披露：Qwen3-4B-Base、processed DAPO、1 epoch、global/mini batch 64、rollout 8、prompt/response 1024/7168、验证最大 31,744、LR `1e-6`、temperature/top-p/repetition penalty 为 1、KL 0、token-mean、8×A800-80GB。入口是[`run_upstream.py`](./run_upstream.py)和[`run_grpo_teacher_4b.sh`](./run_grpo_teacher_4b.sh)。

先检查本机 2×A100 适配计划：

```bash
"$PY" reproduce_4b/run_upstream.py plan \
  --protocol smoke --substage grpo-teacher

CUDA_VISIBLE_DEVICES=0,1 "$PY" reproduce_4b/run_upstream.py run \
  --protocol smoke --substage grpo-teacher --yes
```

本地 Table 1 smoke 已在 attempt 1 完成 2 steps、`exit_code=0`，并保存了完整的 [`global_step_2` checkpoint](../artifacts/upstream/rethinking-opd-upstream-v1/smoke/seed-42/grpo-teacher/attempt-0001/grpo/checkpoints/global_step_2)；其 [`status.json`](../artifacts/upstream/rethinking-opd-upstream-v1/smoke/seed-42/grpo-teacher/status.json)、[`upstream_manifest.json`](../artifacts/upstream/rethinking-opd-upstream-v1/smoke/seed-42/grpo-teacher/attempt-0001/upstream_manifest.json)和[`metrics.jsonl`](../artifacts/upstream/rethinking-opd-upstream-v1/smoke/seed-42/grpo-teacher/attempt-0001/metrics.jsonl)共同记录了成功状态、固定 revision、processed DAPO 指纹和两个训练 step。该产物明确标记为 `engineering-smoke; 2xA100 adaptation; not a paper result`，不能代替一轮、8×A800 的 Table 1 论文训练。

按论文超参数在 2×A100 上重做仍是硬件适配，runner 会要求多日确认：

```bash
"$PY" reproduce_4b/run_upstream.py plan \
  --protocol paper --substage grpo-teacher

CUDA_VISIBLE_DEVICES=0,1 "$PY" reproduce_4b/run_upstream.py run \
  --protocol paper --substage grpo-teacher \
  --yes --acknowledge-multi-day
```

当前 formal Table 1 launcher 明确限制 `N_GPUS=2`；原 8×A800 上游训练尚没有版本化 orchestrator，不能把上面命令称为原硬件复现。若研究目标只需下游 OPD，可直接使用公开且固定 revision 的 `lllyx/Qwen3-4B-Base-GRPO@1f3b...`，并把 Table 1 标为 released-checkpoint producer，而不是声称重新训练成功。

## 9. 上游 Table 3：200K rollout 与全参数 SFT

Table 3（物理页 28）披露的 rollout 为：Qwen3-4B Non-thinking、从 200K OpenThoughts math prompts 每题 1 条、temperature 0.7、top-p 0.95、top-k -1、最大 12,288 tokens，再过滤未完成和重复输出。SFT 为 Qwen3-1.7B-Base 全参数训练，qwen3 template、1 epoch、sequence 14,336、per-device batch 8、gradient accumulation 1、LR `1e-5`、cosine、warmup 0.05、BF16。

当前已建立隔离的 `.venv-sft`：Python 3.12、torch 2.11.0+cu128、transformers 4.57.6、LLaMA-Factory 0.9.5.dev0、DeepSpeed 0.18.4、Liger 0.8.1、torchaudio 2.11 与 torchvision 0.26，且 `pip check` 通过。仓库内 `datasets/OpenThoughts3_opd.parquet` 只有 30,000 行，可用于工程 smoke，不能冒充论文的 200K 源 prompts；正式 `paper` run 仍须显式提供至少 200,000 行的规范化源 parquet：

```bash
ROLLOUT_INPUT=/absolute/path/to/OpenThoughts3-math-at-least-200k.parquet
SFT_PY="$PWD/.venv-sft/bin/python"

"$PY" reproduce_4b/run_upstream.py plan \
  --protocol paper --substage cold-start-rollout \
  --rollout-input "$ROLLOUT_INPUT"

CUDA_VISIBLE_DEVICES=0,1 "$PY" reproduce_4b/run_upstream.py run \
  --protocol paper --substage cold-start-rollout \
  --rollout-input "$ROLLOUT_INPUT" \
  --yes --acknowledge-multi-day
```

rollout 完成后从状态文件读取唯一的过滤后 JSONL，再启动 SFT；不要手填猜测 attempt：

```bash
UPSTREAM_ROOT="$PWD/artifacts/upstream/rethinking-opd-upstream-v3/paper/seed-42"
ROLLOUT_RUN="$($PY -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["run_dir"])' \
  "$UPSTREAM_ROOT/cold-start-rollout/status.json")"
SFT_DATA="$ROLLOUT_RUN/rollout/cold_start_sft.jsonl"

test -s "$SFT_DATA"
test -x "$SFT_PY"

"$PY" reproduce_4b/run_upstream.py plan \
  --protocol paper --substage cold-start-sft \
  --cold-start-data "$SFT_DATA" --sft-python-bin "$SFT_PY"

CUDA_VISIBLE_DEVICES=0,1 "$PY" reproduce_4b/run_upstream.py run \
  --protocol paper --substage cold-start-sft \
  --cold-start-data "$SFT_DATA" --sft-python-bin "$SFT_PY" \
  --yes --acknowledge-multi-day
```

本地 upstream v3 smoke 已完成端到端缩短链路：

- [`cold-start-rollout/status.json`](../artifacts/upstream/rethinking-opd-upstream-v3/smoke/seed-42/cold-start-rollout/status.json)为 attempt 1、exit 0、239.818 s；[`filter_audit.json`](../artifacts/upstream/rethinking-opd-upstream-v3/smoke/seed-42/cold-start-rollout/attempt-0001/rollout/filter_audit.json)记录 4 个选中 prompt、5 次 generation、1 次因截断拒绝后重试，最终 4/4 接受。采样为固定 Qwen3-4B revision、temperature 0.7、top-p 0.95、top-k -1、max tokens 12,288 和本地 seed 42。
- 过滤后的 4 行 JSONL 随后进入 Qwen3-1.7B-Base 全参数 SFT；[`cold-start-sft/status.json`](../artifacts/upstream/rethinking-opd-upstream-v3/smoke/seed-42/cold-start-sft/status.json)为 attempt 1、2 steps、exit 0、65.887 s，[`train_results.json`](../artifacts/upstream/rethinking-opd-upstream-v3/smoke/seed-42/cold-start-sft/attempt-0001/sft/checkpoints/train_results.json)记录 `train_loss=0.7419069`，[`checkpoint-2`](../artifacts/upstream/rethinking-opd-upstream-v3/smoke/seed-42/cold-start-sft/attempt-0001/sft/checkpoints/checkpoint-2)含完整模型权重。

SFT 命令仍按 vendored YAML 请求 `--flash_attn fa2`，但当前隔离环境没有安装 FlashAttention-2；[真实训练日志](../artifacts/upstream/rethinking-opd-upstream-v3/smoke/seed-42/cold-start-sft/attempt-0001/sft/train.log)先给出未安装警告，随后明确使用 torch SDPA。因此该 smoke 验证的是 **SDPA 基础设施适配**，不能写成 FA2 已验证，也没有产生 paper 数值。

更早 upstream [v1 SFT 失败 status](../artifacts/upstream/rethinking-opd-upstream-v1/smoke/seed-42/cold-start-sft/status.json)与[v2 SFT 失败 status](../artifacts/upstream/rethinking-opd-upstream-v2/smoke/seed-42/cold-start-sft/status.json)仍保留为历史证据，不覆盖 v3 成功 attempt。本地 rollout 会记录 prompt 选择、retry、有效 generation seed、接受/拒绝原因和 shard；这些是可审计的本地选择，不是论文未披露信息的恢复。作者未公开精确 sampled prompt IDs、过滤后 row IDs、随机 seed、retry/filter 实现；4-prompt smoke、公开 SFT checkpoint 与公开数据集的行数/叙述都不能证明其训练 corpus 与 Table 3 逐行一致。

## 10. 论文评测：严格 `avg@16`

三项 benchmark 固定为 AIME 2024、AIME 2025、AMC 2023；每题 16 次，temperature 0.7、top-p 0.95、最大新 token 31,744、thinking off、seed 42。主指标是各题 16 次独立正确率再平均，即 `avg@16`；不是 `pass@16`。三项总分是三个 benchmark 的非加权平均。

### 10.1 固定模型/step-0/teacher

先 plan，再运行完整评测。完整 workload 必须同时给出 `--yes` 和 `--acknowledge-full-eval`：

```bash
"$PY" reproduce_4b/evaluate_model.py plan \
  --target-id fig3-grpo-teacher-paper \
  --model lllyx/Qwen3-4B-Base-GRPO \
  --revision 1f3b2966edfb75f2f98a00617588c1f748088422

CUDA_VISIBLE_DEVICES=0,1 "$PY" reproduce_4b/evaluate_model.py run \
  --target-id fig3-grpo-teacher-paper \
  --model lllyx/Qwen3-4B-Base-GRPO \
  --revision 1f3b2966edfb75f2f98a00617588c1f748088422 \
  --n 16 --max-tokens 31744 --seed 42 \
  --yes --acknowledge-full-eval
```

Non-thinking teacher 使用独立 target ID：

```bash
CUDA_VISIBLE_DEVICES=0,1 "$PY" reproduce_4b/evaluate_model.py run \
  --target-id fig3-nonthinking-teacher-paper \
  --model Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --n 16 --max-tokens 31744 --seed 42 \
  --yes --acknowledge-full-eval
```

带 `--limit`、`n!=16` 或较短 `max-tokens` 的 target manifest 会自动标为 `paper_comparable=false`。

### 10.2 OPD checkpoint

必须从 cell 的 `status.json` 读取实际 attempt 路径。对 `paper` training，第一次 checkpoint 评测必须显式重复 `--checkpoint-step` 注册**完整、排序且唯一**的 grid；没有已注册 plan 时，禁止隐式选择 latest：

```bash
SUITE_ROOT="$PWD/artifacts/ablations/rethinking-opd-complete-paper-v3/paper/seed-42"
CELL=fig16-topk-16
CELL_RUN="$($PY -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["run_dir"])' \
  "$SUITE_ROOT/$CELL/status.json")"

"$PY" reproduce_4b/evaluate_ablation.py plan \
  --run-dir "$CELL_RUN" --checkpoint-step 200

CUDA_VISIBLE_DEVICES=0,1 "$PY" reproduce_4b/evaluate_ablation.py run \
  --run-dir "$CELL_RUN" --checkpoint-step 200 \
  --n 16 --max-tokens 31744 --seed 42 \
  --yes --acknowledge-full-eval
```

`plan` action 只预览，不写注册文件；首次 `run` 会在任何 merge/generate 前 write-once 创建：

- `RUN/evaluation/evaluation_plan.json`：完整 checkpoint grid、training identity、sampling 和 benchmark 合同；
- `RUN/evaluation/global_step_N/target_manifest.json`：单步 source checkpoint、合并模型/tokenizer identity 和 plan SHA；
- 同目录 `summary.json`，以及累积的 `RUN/evaluation_status.json`。

plan、target manifest 和 summary 内容相同才可只读复用，不允许覆盖。相同参数中断重跑时，已有 plan 可自动恢复其 grid；完整且已验证的 benchmark 会跳过，只补 pending 项，status 保留历次 attempts/commands。只有 summaries **恰好覆盖整个注册 grid** 才能进入 `completed`，不得 cherry-pick subset。paper checkpoint 评测禁止 `--overwrite-incomplete`；grid、模型/tokenizer identity、`n`、max tokens、seed 或 limit 任一变化都要求新的 evaluation identity。

若要同时观察长度实验的文本条件 step 200 与图示窗口 step 260，两个 checkpoint 必须在训练前由一个单独版本化的 figure-window 条件保留下来，并在首次评测时一起注册：

```bash
LENGTH_RUN=/absolute/path/to/versioned-length-figure-window-run
test -d "$LENGTH_RUN"

"$PY" reproduce_4b/evaluate_ablation.py plan \
  --run-dir "$LENGTH_RUN" \
  --checkpoint-step 200 --checkpoint-step 260

CUDA_VISIBLE_DEVICES=0,1 "$PY" reproduce_4b/evaluate_ablation.py run \
  --run-dir "$LENGTH_RUN" \
  --checkpoint-step 200 --checkpoint-step 260 \
  --n 16 --max-tokens 31744 --seed 42 \
  --yes --acknowledge-full-eval
```

v3 标准 `paper` 长度条件遵循正文并在 step 200 结束；标准 length pilot 虽到 260，但只保留最近两个 checkpoint，通常不足以同时留下 200/260。v3 `paper` 的 model-only/unlimited retention 只保留实际运行范围内每 20 step 的权重，不能凭空产生 step 260。必须在训练前固定科学条件、grid 和 retention；evaluation plan 只是在评测前锁定已有 checkpoint，不能替代训练前预注册。训练结束后也不能从 metrics 反推出已删除或从未保存的权重。

## 11. Figure 11(b) teacher-continuation probe

[`probe_teacher_continuation.py`](./probe_teacher_continuation.py)实现了论文明确披露的结构不变量：恰好 2,000 个 DAPO prompts、完整学生 rollout、严格筛选 response token 数 `>16384`（不是 `>=`），并为每个选中 rollout 建立 1,024/4,096/8,192/16,384 四个 prefix 的配对教师续写工作量。当前已归档一个[非可运行 protocol template](../artifacts/paper_reproduction/probes/fig11b_protocol_template.json)；其中模型/revision 仍是显式 placeholder，不能被计作 probe workload 已运行。

如需创建独立工作副本，先生成非可运行模板；脚本默认拒绝覆盖已有文件：

```bash
PROBE11="$PWD/artifacts/paper_reproduction/probes/figure11b"
mkdir -p "$PROBE11"

"$PY" reproduce_4b/probe_teacher_continuation.py protocol-template \
  --output-json "$PROBE11/protocol.json"
```

必须人工把 student/teacher model 和 immutable revision 替换掉，并明确记录本地 prompt seed、generation seed、temperature、top-p、max tokens。论文未披露这些字段，因此默认 provenance 必须保留为 `paper-undisclosed-explicit-local`。准备符合脚本 schema 的 2K 学生 rollout 后：

```bash
STUDENT_ROLLOUTS=/absolute/path/to/fig11b_student_rollouts.jsonl

"$PY" reproduce_4b/probe_teacher_continuation.py plan \
  --student-rollouts "$STUDENT_ROLLOUTS" \
  --protocol-json "$PROBE11/protocol.json" \
  --output-jsonl "$PROBE11/continuation_plan.jsonl" \
  --dry-run

"$PY" reproduce_4b/probe_teacher_continuation.py plan \
  --student-rollouts "$STUDENT_ROLLOUTS" \
  --protocol-json "$PROBE11/protocol.json" \
  --output-jsonl "$PROBE11/continuation_plan.jsonl"
```

本脚本故意不隐式选择生成器。由固定 teacher generator 消费 plan、保留 `pair_id` 和 sampling metadata 后，再验证和汇总：

```bash
TEACHER_RESULTS=/absolute/path/to/fig11b_teacher_results.jsonl

"$PY" reproduce_4b/probe_teacher_continuation.py validate \
  --student-rollouts "$STUDENT_ROLLOUTS" \
  --protocol-json "$PROBE11/protocol.json" \
  --plan-jsonl "$PROBE11/continuation_plan.jsonl"

"$PY" reproduce_4b/probe_teacher_continuation.py summarize \
  --student-rollouts "$STUDENT_ROLLOUTS" \
  --protocol-json "$PROBE11/protocol.json" \
  --plan-jsonl "$PROBE11/continuation_plan.jsonl" \
  --results-jsonl "$TEACHER_RESULTS" \
  --output-dir "$PROBE11/summary"
```

论文报告的准确率增益为 1K/4K/8K/16K：0.3659/0.2709/0.1522/0.0237（`P`，物理页 14）。本地数值只有在 source checkpoint、2K prompt batch 和 sampling provenance 完整保存后才能比较；仍不能称为作者逐样本精确复现。

## 12. Figure 14 sequence-reward/AUROC probe

[`analyze_sequence_reward.py`](./analyze_sequence_reward.py)要求同一不可变 rollout batch、student token log-prob 和 teacher token log-prob 三者的 prompt/response token IDs、rollout ID 和 fingerprint 完全一致。每条序列计算

\[
r(x,y)=\frac{1}{|y|}\sum_t\left[\log p_T(y_t\mid x,y_{<t})-\log p_S(y_t\mid x,y_{<t})\right],
\]

再按 correctness 计算 tie-safe AUROC。下面先用 `--dry-run` 审计模型身份，再实际打分：

```bash
PROBE14="$PWD/artifacts/paper_reproduction/probes/figure14"
ROLLOUTS=/absolute/path/to/fixed_graded_rollouts.jsonl
SAMPLING=/absolute/path/to/fixed_rollout_sampling.json
mkdir -p "$PROBE14"

"$PY" reproduce_4b/analyze_sequence_reward.py score \
  --rollouts "$ROLLOUTS" --sampling-json "$SAMPLING" \
  --role student \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --revision ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
  --output-jsonl "$PROBE14/student.jsonl" \
  --device cuda:0 --dry-run

CUDA_VISIBLE_DEVICES=0 "$PY" reproduce_4b/analyze_sequence_reward.py score \
  --rollouts "$ROLLOUTS" --sampling-json "$SAMPLING" \
  --role student \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --revision ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
  --output-jsonl "$PROBE14/student.jsonl" --device cuda:0

CUDA_VISIBLE_DEVICES=1 "$PY" reproduce_4b/analyze_sequence_reward.py score \
  --rollouts "$ROLLOUTS" --sampling-json "$SAMPLING" \
  --role teacher \
  --model hbx/JustRL-DeepSeek-1.5B \
  --revision 0637e4096c789c67f9eecbe8355e0bdeddede1c2 \
  --output-jsonl "$PROBE14/teacher-justrl.jsonl" --device cuda:0

"$PY" reproduce_4b/analyze_sequence_reward.py analyze \
  --rollouts "$ROLLOUTS" \
  --student-logprobs "$PROBE14/student.jsonl" \
  --teacher-logprobs "$PROBE14/teacher-justrl.jsonl" \
  --sampling-json "$SAMPLING" \
  --student-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --student-revision ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
  --teacher-model hbx/JustRL-DeepSeek-1.5B \
  --teacher-revision 0637e4096c789c67f9eecbe8355e0bdeddede1c2 \
  --output-dir "$PROBE14/justrl-summary"
```

对 R1-7B teacher 重复 `score/analyze`，输出到独立目录。论文报告 correct/incorrect 数为 2,828/1,451，AUROC 为 0.7333/0.7511（JustRL-1.5B/R1-7B，`P`，物理页 16）。作者没有公开产生这 4,279 条样本的 checkpoint、prompt IDs、seed、sampling 和 grader identity；本地不同 batch 不应拿计数相等作为“精确复现”的充分条件。

## 13. Figure 19 新指标

新训练代码已实现论文需要的第三条诊断量。对每个有效 decoding state，在 student-side Top-k 与 teacher Top-k 的交集上选

\[
v^*=\arg\max_v |A(v)|,
\qquad \Delta p=p_S(v^*)-p_T(v^*),
\]

然后对有效状态平均，保留概率差的符号。实现见[`verl/utils/opd.py`](../verl/verl/utils/opd.py)和[`ray_trainer.py`](../verl/verl/trainer/ppo/ray_trainer.py)，测试见[`test_figure19_metric.py`](./tests/test_figure19_metric.py)。新日志必须同时包含：

- `opd/figure19_metric_schema_version=1`；
- `val-extrema/prob_diff_at_max_abs_adv_intersection`；
- policy-gradient loss 和 gradient norm。

旧 v1 smoke 只分别记录过正/负 extrema，无法还原“按 `argmax |adv|` 选择同一个 token 后的有符号 `p_s-p_t`”。因此 Figure 19 必须使用合入该 schema 后重新启动的 Fig. 6 两条轨迹；禁止从旧日志拼接或追溯填值。

## 14. 长表、动态覆盖和论文图 renderer

下面的 Bash 数组会发现所有真实 suite manifest，并把每个 root 显式传给工具；不会把不同 suite/protocol/seed 合并成无标注曲线：

```bash
mapfile -t SUITES < <(find "$PWD/artifacts/ablations" \
  -type f -name suite_manifest.json -printf '%h\n' | sort)

SUITE_ARGS=()
for suite in "${SUITES[@]}"; do
  SUITE_ARGS+=(--suite-root "$suite")
done

mapfile -t UPSTREAMS < <(find "$PWD/artifacts/upstream" \
  -mindepth 1 -maxdepth 1 -type d -name 'rethinking-opd-upstream-v*' | sort)

UPSTREAM_ARGS=()
for upstream in "${UPSTREAMS[@]}"; do
  UPSTREAM_ARGS+=(--upstream-root "$upstream")
done
```

### 14.1 训练/评测长表

[`collect_paper_results.py`](./collect_paper_results.py)保留每条 scalar 的 suite、protocol、seed、source hash、cell、fingerprint、attempt、step 和 fidelity；评测长表另外保留 checkpoint、benchmark、`n` 和 comparability：

```bash
"$PY" reproduce_4b/collect_paper_results.py \
  "${SUITE_ARGS[@]}" \
  "${UPSTREAM_ARGS[@]}" \
  --model-eval-root artifacts/evaluation/paper-models \
  --output-dir artifacts/paper_reproduction/tables
```

当前 collector 实际输出 `training_rows=9666`、`evaluation_rows=6`、`upstream_rows=7`：[`metrics_long.json`](../artifacts/paper_reproduction/tables/metrics_long.json)、[`metrics_long.csv`](../artifacts/paper_reproduction/tables/metrics_long.csv)、[`evaluation_long.json`](../artifacts/paper_reproduction/tables/evaluation_long.json)、[`evaluation_long.csv`](../artifacts/paper_reproduction/tables/evaluation_long.csv)、[`upstream_long.json`](../artifacts/paper_reproduction/tables/upstream_long.json)、[`upstream_long.csv`](../artifacts/paper_reproduction/tables/upstream_long.csv)。6 条评测记录来自两个 teacher × 三个 benchmark，均为 `n=1`；7 条 upstream 记录包含 v1/v2 历史状态与 v3 成功状态。三类当前记录的 `paper_comparable` 均为 false。

### 14.2 覆盖审计与训练矩阵

[`audit_paper_coverage.py`](./audit_paper_coverage.py)同时验证 26 个 ledger 条目、31 个 cell 引用、已完成 protocol、checkpoint evaluation、固定模型 evaluation 和 probe summary：

```bash
"$PY" reproduce_4b/audit_paper_coverage.py \
  --matrix reproduce_4b/paper_full_matrix.json \
  --ledger reproduce_4b/paper_experiment_ledger.json \
  "${SUITE_ARGS[@]}" \
  --model-eval-root artifacts/evaluation/paper-models \
  --probe-root artifacts/paper_reproduction/probes \
  --output-dir artifacts/paper_reproduction/coverage
```

主要输出是[动态覆盖报告](../artifacts/paper_reproduction/coverage/coverage.md)和[完整训练矩阵](../artifacts/paper_reproduction/coverage/training_matrix.md)，并同时生成 JSON/CSV。`smoke` 永远不会被升级成 paper-comparable。

当前覆盖审计为：结构 26/26、带至少一种训练 smoke 的注册 cell role 27/31、论文可比条目 0/26、硬阻塞条目 3/26。三个条目对应两个根因：未公开 Qwen RL-Math 权重同时阻塞 Figure 1 composite 与 Figure 4，本机没有原尺度 8×80GB 执行授权则阻塞 Figure 20。27/31 是跨 suite 的 role 映射，主要证据来自 v2 完整默认 smoke；不能误写成 v3 已跑 27/27。R1-7B padded-vocabulary 差异已不再是硬阻塞。

### 14.3 Figure 1–23 renderer

[`render_paper_figures.py`](./render_paper_figures.py)按 ledger 逐图选择 producer。缺 cell、缺 `avg@16`、probe 配对不完整、Figure 19 schema 不匹配或作者 composite 无法重建时，严格跳过并把原因写入 manifest；smoke 图会在图内写明 `NOT PAPER-COMPARABLE`。

每次写入新的时间戳目录，避免覆盖既有图：

```bash
RENDER_OUT="$PWD/artifacts/paper_reproduction/figures/$(date -u +%Y%m%dT%H%M%SZ)"

"$PY" reproduce_4b/render_paper_figures.py \
  --ledger reproduce_4b/paper_experiment_ledger.json \
  "${SUITE_ARGS[@]}" \
  --model-eval-root artifacts/evaluation/paper-models \
  --probe-root artifacts/paper_reproduction/probes \
  --output-dir "$RENDER_OUT"

"$PY" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d["summary"], indent=2))' \
  "$RENDER_OUT/render_manifest.json"
```

只有 manifest 中 `paper_comparable=true` 的 variant 才能进入论文数值对照表；“成功渲染”本身只表示有可视化输入。

当前[渲染 manifest](../artifacts/paper_reproduction/figures/render_manifest.json)记录 23 张图中渲染 15 张、跳过 8 张、论文可比 variant 为 0。现有输出均在图内标为 smoke/`NOT PAPER-COMPARABLE`：

- 训练/评测概览：[Fig. 2](../artifacts/paper_reproduction/figures/figure-02_smoke.png)、[Fig. 3](../artifacts/paper_reproduction/figures/figure-03_smoke.png)、[Fig. 5](../artifacts/paper_reproduction/figures/figure-05_smoke.png)、[Fig. 6](../artifacts/paper_reproduction/figures/figure-06_smoke.png)、[Fig. 7](../artifacts/paper_reproduction/figures/figure-07_smoke.png)、[Fig. 8](../artifacts/paper_reproduction/figures/figure-08_smoke.png)、[Fig. 9](../artifacts/paper_reproduction/figures/figure-09_smoke.png)、[Fig. 10](../artifacts/paper_reproduction/figures/figure-10_smoke.png)、[Fig. 12](../artifacts/paper_reproduction/figures/figure-12_smoke.png)、[Fig. 16](../artifacts/paper_reproduction/figures/figure-16_smoke.png)；
- 机制诊断：[Fig. 13](../artifacts/paper_reproduction/figures/figure-13_smoke.png)、[Fig. 18](../artifacts/paper_reproduction/figures/figure-18_smoke.png)、[Fig. 19](../artifacts/paper_reproduction/figures/figure-19_smoke.png)、[Fig. 21](../artifacts/paper_reproduction/figures/figure-21_smoke.png)、[Fig. 23](../artifacts/paper_reproduction/figures/figure-23_smoke.png)。

跳过的 8 张图仍缺完整 producer、严格评测或 probe；渲染成功本身不提升证据等级。renderer 当前为 Fig. 6 选择的是 v2 完整 suite 变体，不能用 v3 代表 smoke 的存在改写其 provenance。

## 15. 当前真实 smoke 产物

这些产物用于验证工程链路，不能支持论文机制或性能结论：

| 产物 | 状态 | 可验证内容 | 不可得出的结论 |
|---|---|---|---|
| v1 Fig. 7 三格：`student-topk`、`overlap-topk`、`nonoverlap-author` | 均完成 2 steps | 三种 support 分支、Eq. (7)/(8)、overlap/entropy/grad 日志链路 | 不证明完整 Student/Overlap 优于 Non-overlap |
| v1 15K response-length | 完成 2 steps | 15,360-token rollout 与 position entropy 分箱链路 | 不证明 suffix-to-prefix 后期熵扩散 |
| v2 默认完整矩阵 | 27/27 cell 均 completed、2 steps、exit 0；25 个 attempt 1，两个网络预检重试为 attempt 2 | 11 组中所有非门禁 cell 的训练/指标分支 | 不等于 v3 跑了 27 格，也不证明任何完整训练趋势 |
| v2 `fig5-reverse-r1-1p5b` | attempt 1，2 steps，exit 0，113.9 s | JustRL-1.5B→R1-1.5B、Figure 19 新指标和 v2 provenance 链路 | 不证明 600-step 退化轨迹 |
| v2 `fig5-reverse-r1-7b` | attempt 1，2 steps，exit 0，522.1 s | 完整 tokenizer-ID/padded-head 安全验证、1.5B→7B 反向蒸馏和 Figure 19 新指标链路 | 不证明 600-step 退化轨迹或教师优劣 |
| v3 `fig6-deepseek-justrl-success` | attempt 1，2 steps，exit 0，115.2 s | v3 matrix SHA、动态 GPU allocation、逐卡锁和当前源码下的代表训练链路 | 不代表 v3 其余 26 个默认 cell 已重跑 |
| full-matrix 草案 v1 `fig5-reverse-r1-7b` | 旧 preflight 失败，无 step | 记录旧版仅比较 config vocab-size 的过严门禁 | 不是当前实现失败，也不是科学失败 |
| Fig. 3 GRPO teacher smoke evaluation | 三 benchmark 各 1 prompt × 1 response、最大 256 tokens，已完成 | 固定模型加载、生成、严格 grader 三 benchmark 链路 | `avg@1=0` 且截断，不能与论文 Fig. 3 数值比较 |
| Fig. 3 Non-thinking teacher smoke evaluation | 使用相同 manifest 和离线缓存续跑缺失 AMC23，三 benchmark 均完成；`n=1/max_tokens=256/avg@1=0` | 评测器保留完成项并恢复 pending command、最终聚合 summary 的链路 | 仍不是 `avg@16`，不能比较论文数值 |
| Table 1 upstream v1 GRPO | attempt 1，2 steps，exit 0，保存 `global_step_2` | processed DAPO→Qwen3-4B-Base GRPO 的 2×A100 缩短链路 | 不等于一轮、8×A800 的 Table 1 训练 |
| Table 3 upstream v3 | 4 prompts 经 5 次生成后 4/4 接受；SFT 2 steps、exit 0、loss 0.7419069、保存模型 | rollout/filter/retry→Qwen3-1.7B-Base 全参数 SFT 的 SDPA 链路 | 不等于 200K 生成/SFT，也不证明 FA2 或论文结果 |
| Figure 11(b) protocol template | 已生成，模型/revision 保持 placeholder | 2K prompts、`>16K` 和四个 prefix 的 schema/门禁 | 不代表学生 rollout 或教师 continuation 已生成 |

对应不可变证据：

- [v1 smoke 汇总 JSON](../artifacts/ablations/rethinking-opd-formal-ablations-v1/smoke/seed-42/results.json)、[CSV](../artifacts/ablations/rethinking-opd-formal-ablations-v1/smoke/seed-42/results.csv)、[Fig. 7 工程图](../artifacts/ablations/rethinking-opd-formal-ablations-v1/smoke/seed-42/plots/fig7_support.png)、[15K 两步热图](../artifacts/ablations/rethinking-opd-formal-ablations-v1/smoke/seed-42/plots/fig12_length_15360_position_entropy.png)；
- v2 完整默认 smoke：[suite manifest](../artifacts/ablations/rethinking-opd-complete-paper-v2/smoke/seed-42/suite_manifest.json)与[27-cell 状态目录](../artifacts/ablations/rethinking-opd-complete-paper-v2/smoke/seed-42)；其中 Figure 5 的 1.5B teacher 有[status](../artifacts/ablations/rethinking-opd-complete-paper-v2/smoke/seed-42/fig5-reverse-r1-1p5b/status.json)与[metrics](../artifacts/ablations/rethinking-opd-complete-paper-v2/smoke/seed-42/fig5-reverse-r1-1p5b/attempt-0001/metrics.jsonl)，7B teacher 有[status](../artifacts/ablations/rethinking-opd-complete-paper-v2/smoke/seed-42/fig5-reverse-r1-7b/status.json)、[preflight](../artifacts/ablations/rethinking-opd-complete-paper-v2/smoke/seed-42/fig5-reverse-r1-7b/attempt-0001/preflight.log)与[metrics](../artifacts/ablations/rethinking-opd-complete-paper-v2/smoke/seed-42/fig5-reverse-r1-7b/attempt-0001/metrics.jsonl)；
- v3 代表格：[suite manifest](../artifacts/ablations/rethinking-opd-complete-paper-v3/smoke/seed-42/suite_manifest.json)、[status](../artifacts/ablations/rethinking-opd-complete-paper-v3/smoke/seed-42/fig6-deepseek-justrl-success/status.json)与[metrics](../artifacts/ablations/rethinking-opd-complete-paper-v3/smoke/seed-42/fig6-deepseek-justrl-success/attempt-0001/metrics.jsonl)；
- 历史草案 v1：1.5B 的[完成 status](../artifacts/ablations/rethinking-opd-complete-paper-v1/smoke/seed-42/fig5-reverse-r1-1p5b/status.json)和 7B 的[失败 preflight](../artifacts/ablations/rethinking-opd-complete-paper-v1/smoke/seed-42/fig5-reverse-r1-7b/attempt-0001/preflight.log)，均不并入 v2 fingerprint；
- [GRPO teacher smoke status](../artifacts/evaluation/paper-models/fig3-grpo-teacher-smoke/status.json)与[summary](../artifacts/evaluation/paper-models/fig3-grpo-teacher-smoke/summary.json)；
- [Non-thinking teacher smoke status](../artifacts/evaluation/paper-models/fig3-nonthinking-teacher-smoke/status.json)与[summary](../artifacts/evaluation/paper-models/fig3-nonthinking-teacher-smoke/summary.json)，以及合并两者的[Figure 3 smoke 图](../artifacts/paper_reproduction/figures/figure-03_smoke.png)；
- Table 1 upstream v1 的[status](../artifacts/upstream/rethinking-opd-upstream-v1/smoke/seed-42/grpo-teacher/status.json)与[metrics](../artifacts/upstream/rethinking-opd-upstream-v1/smoke/seed-42/grpo-teacher/attempt-0001/metrics.jsonl)，Table 3 upstream v3 的[rollout status](../artifacts/upstream/rethinking-opd-upstream-v3/smoke/seed-42/cold-start-rollout/status.json)、[filter audit](../artifacts/upstream/rethinking-opd-upstream-v3/smoke/seed-42/cold-start-rollout/attempt-0001/rollout/filter_audit.json)、[SFT status](../artifacts/upstream/rethinking-opd-upstream-v3/smoke/seed-42/cold-start-sft/status.json)与[trainer state](../artifacts/upstream/rethinking-opd-upstream-v3/smoke/seed-42/cold-start-sft/attempt-0001/sft/checkpoints/trainer_state.json)；
- [Figure 11(b) protocol template](../artifacts/paper_reproduction/probes/fig11b_protocol_template.json)与[完整 renderer manifest](../artifacts/paper_reproduction/figures/render_manifest.json)。

当前全量单元测试为 **149/149 通过**。训练重试必须由 runner 创建新 attempt。固定模型评测在 target manifest 完全一致时可恢复 pending benchmark；paper checkpoint 评测则必须服从已注册的完整 grid，禁止 `--overwrite-incomplete`，且只有全 grid 完成才成功。模型/tokenizer、grid、`n`、长度、seed 或 prompt limit 任一变化都必须使用新的 evaluation identity。不要把失败的基础设施或历史预检 attempt 计入科学失败率。

## 16. 无法从论文公开信息精确恢复的项目

以下缺口必须出现在最终报告，不能用默认值或第三方近似静默填补：

1. **Qwen RL-Math teacher**：论文命名 `Qwen3-4B-Non-Thinking-RL-Math`，但作者未公开 checkpoint/revision。公开搜索到的第三方同名或近似 checkpoint 不能替代作者权重；因此 Fig. 1/4 的 Qwen new-knowledge 部分不能精确完成。
2. **Figure 10 匹配数据**：作者未公开 matched DAPO row IDs 或采样 seed。本地固定 14,116 行、seed 42 和 SHA-256 只是一项可审计重建。
3. **随机性与 checkpoint 选择**：论文未完整披露 OPD/GRPO 的 seeds、optimizer betas、weight decay、gradient clipping、precision 细节和 checkpoint-selection 规则。
4. **Figure 11(b)**：未披露 source student checkpoint、2K prompt IDs/seed、student/teacher generation seed、temperature、top-p 和 continuation sampling 细节。
5. **Figure 14**：未披露产生 2,828 correct + 1,451 incorrect 固定 rollout batch 的 checkpoint、prompt IDs、seed、sampling 和 grader identity。
6. **Table 3**：v3 的 4-prompt rollout 与两步全参数 SFT 已通过，但论文未披露精确 200K prompt IDs、retry/filter 实现、过滤后 row IDs 和 seed；公开数据行数、论文描述、模型卡设置之间的差异不能靠 smoke 推断消除。当前 smoke 实际走 torch SDPA，而不是命令请求但环境未安装的 FA2。
7. **Figure 5/12/13 训练横轴**：Fig. 5 展示 600 steps；长度正文写 200 steps，但图延伸到约 250–260。各自按显式 group override 运行并并列报告，不能合并成一个“作者默认步数”。
8. **Figure 19 历史值**：新指标需要在训练时知道同一 `argmax |adv|` token；旧日志无法重建。
9. **Figure 20 硬件**：v3 已能按 cell 动态分配并锁定 8 卡，但 7B student + 14B teacher 不在当前 2×A100 主机执行授权内；只能在独占 8×80GB 节点按单独协议运行。
10. **跨规模 padded vocabulary**：JustRL/R1-1.5B 与 R1-7B 的 output-head 尺寸不同。当前实现已通过完整 tokenizer ID 映射、最大真实 ID 和 `only_stu` 支持方向验证安全运行，v2 smoke 是实际执行证据；这解决的是本地工程门禁，不是论文的额外披露。换 support 方向或模型后必须重新验证，不能沿用本次豁免。
11. **JustRL checkpoint mirror**：本地固定的是可访问 `hbx` revision；它与受限 `thunlp` revision 的 byte identity 未独立验证。使用该镜像的 Fig. 5/6/14/15/16/18/19 结果必须保留 proxy caveat。

因此“bitwise 权重一致”“逐样本结果一致”以及“全部论文数值已经复现”都不能从现有披露和当前产物得出。可实现的最高目标是：公开资源上的参数化、可审计复现；对未披露项给出固定本地重建并单独标注。

## 17. 最终验收清单

每次阶段性交付至少检查：

- v3 plan 为 11 组、31 个注册 cell role、29 个完整 env、28 个科学训练条件、默认 27、gated 4，matrix SHA 为 `a45bc9aff21bc6c12a687660cd5d5953ce022f924f9c5fec7855f23c59818225`；
- 每个成功 run 都有 suite manifest、cell fingerprint、source hash、固定 model revision/data hash、status、metrics 和不可覆盖 attempt；
- smoke/calibration/pilot 与 paper 目录严格分开；
- paper 每 20 steps 保存 model-only checkpoint、retention 0 表示不限量保留；Figure 5 保存 600-step 预算，长度实验分别报告正文 step 200 与图示后期窗口；
- Figure 19 新 run 含 schema-v1 指标，旧 run 不混入；
- paper checkpoint 首次评测显式注册完整 grid，plan/target manifest/summary 不可覆盖；所有被比较 checkpoint 均完成 AIME24/AIME25/AMC23 的 `avg@16`，sampling metadata 与完整 prompt 数通过验证；
- Figure 11(b)/14 使用同一批次、ID、token 和 sampling fingerprint 的配对检查；
- collector 输出无跨 suite/seed 的匿名合并；
- coverage 为 26/26 结构条目，并分别显示训练、评测、probe 和 hard blocker；
- renderer manifest 中缺失 producer 为 `skipped`，smoke 为 `NOT PAPER-COMPARABLE`；
- Table 3 报告记录实际 attention backend；当前 smoke 是 SDPA，不得标作 FA2；
- 最终报告逐项标出 `P/R/I/S/C`，并列出全部未披露字段与硬件差异。

只有当动态覆盖报告显示所有非硬阻塞条目完成 `paper` 训练和完整 `avg@16`，probe 工作量也完成，才能写“完成了公开条件下的论文数值复现”。即使到那一步，RL-Math、未公开 row IDs/seeds、Figure 11(b)/14 source identity 和 Figure 20 硬件等缺口仍需保留，不能升级成作者端的精确复现。
