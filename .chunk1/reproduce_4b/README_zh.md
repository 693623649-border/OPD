# 2×A100 80GB 上复现 Rethinking OPD

这套入口把论文的 OPD 主实验压缩到单机 2×NVIDIA A100 80GB，并保留最关键的训练目标、数据模板与机制诊断。先读[论文精读](../paper/rethinking_opd_summary_zh.md)；原文在[本地 PDF](../paper/rethinking_opd.pdf)。当前机器的两步端到端实测见[验证记录](VALIDATION_2XA100.md)。

如果目标是在当前两张空闲 A100 上一夜内得到论文主线的本地趋势，优先使用新设计的[Core 7-cell / Extended 14-cell 轻量复刻版](LITE_REPRODUCTION_zh.md)；本页继续作为环境、底层参数和单次运行参考。

> 本指南区分“代码跑通”“机制趋势复现”和“论文分数复现”。脚本、日志或文档存在不代表完整训练已经实际完成；只有保存了对应日志、checkpoint 和评测 JSON 的运行才能计入结果。

> 本页主要解释环境与单次运行入口。论文 Figure 2/6/7/8/9/10/11(a)/12/13/15/16 的正式 24-cell 消融、协议门禁与当前实机进度，以[正式消融指南](ABLATIONS_zh.md)和 [`ablation_matrix.json`](ablation_matrix.json) 为准；下文 `run_matrix.sh` 是早期 6-cell Qwen 快捷矩阵，不代表完整论文消融。

## 先确定复现边界

| `MODEL_PAIR` | 学生 | 冻结教师 | 应怎样表述 |
|---|---|---|---|
| `paper` | `Qwen/Qwen3-1.7B-Base` | `lllyx/Qwen3-4B-Base-GRPO` | 论文 Fig. 2 的成功 pair；最大模型为 4B，也是默认主线 |
| `4b` | `Qwen/Qwen3-4B-Base` | `lllyx/Qwen3-4B-Base-GRPO` | 面向本机的同尺寸 4B 机制迁移 |
| `mismatch` | `Qwen/Qwen3-1.7B-Base` | `Qwen/Qwen3-4B` | 论文的 Non-thinking pattern-mismatch 对照，不是主结果 |

请在正式命令中显式写 `MODEL_PAIR=paper`，避免把实验身份弄混。`MODEL_PAIR=4b` 没有出现在论文的这个主实验中；它可以验证 reverse-KL/Top-k 机制是否迁移到 4B 学生，但**不能**宣称复现了论文数值或原曲线。

启动器默认值就是 `PRESET=smoke`、`MODEL_PAIR=paper`。也就是说，无环境变量执行时只会尝试 Fig. 2 pair 与原始 DAPO 模板的最小链路检查，不会意外启动完整训练。

本地改造针对 2×A100 80GB；论文原实验使用 8×A800 80GB。双卡 preset 会改变同步 batch、更新时序和部分序列长度，因此即便 `MODEL_PAIR=paper` 也应完整记录 preset 和所有 override。

## 方法与数据不变量

学生在自己的 rollout prefix 上查询教师，最小化逐 token reverse KL：

\[
\mathbb E_{y\sim\pi_\theta}\sum_t
D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot\mid x,y_{<t})\,\|\,\pi_T(\cdot\mid x,y_{<t})\right).
\]

主配置保持：

- Student Top-16：`TOP_K=16`（传入 verl 后为 `log_prob_top_k=16`）；
- 从学生 Top-k 取支持集：`TOP_K_STRATEGY=only_stu`；
- 按学生概率加权：`REWARD_WEIGHT_MODE=student_p`；
- `algorithm.adv_estimator=token_reward_direct`，不用 outcome reward，也不额外加 KL penalty；
- 与论文入口一致使用 `data.shuffle=false`；可通过 `DATA_SHUFFLE` 显式改动，但这会改变逐 step 曲线；
- `SEED` 同时写入数据配置和 vLLM rollout engine；如需重放旧产物，可用 `ROLLOUT_SEED` 单独覆盖后者；
- `paper` preset 使用 rollout temperature 1.0、每题 4 条、学习率 `1e-6`。

三个内置 pair 的学生都是 Base 模型，训练时均保持 `ENABLE_THINKING=auto`。`MODEL_PAIR=mismatch` 的 “Non-thinking” 是冻结教师权重的属性；`data.apply_chat_template_kwargs` 实际作用于学生 tokenizer，不能因为教师 non-thinking 就把它设为 false。只有自定义的可训练学生本身是 Non-thinking 模型时才显式使用 `ENABLE_THINKING=false`。这与后文评测不同：论文评测统一显式使用 `--thinking off`。

Fig. 2 及论文此前未特别说明的实验使用 [`datasets/dapo-math-17k.parquet`](../datasets/dapo-math-17k.parquet) 中的原始 DAPO 模板，启动器也默认使用它：

```text
Solve the following math problem step by step. ... {Question}
Remember to put your answer on its own line after "Answer:".
```

论文 §5.2 的改进 recipe 才使用 [`datasets/dapo-math-17k-processed.parquet`](../datasets/dapo-math-17k-processed.parquet) 中与教师 RL 对齐的模板：

```text
{Question} Please reason step by step, and put your final answer within \boxed{}.
```

两份 parquet 含相同的 17,917 道题，但模板会改变学生访问的状态分布。要验证 §5.2 recipe 或 4B 迁移中的 aligned 变体，显式设置 `TRAIN_DATA="$PWD/datasets/dapo-math-17k-processed.parquet"`，并在实验名中标注 `aligned`；不能把它与 Fig. 2 原始模板结果混为一谈。生成和评测脚本直接使用 parquet 内已存的 `prompt`，不会再追加指令。

### 为什么学生初始化是 FP32、前向却是 BF16

这里的精度设置不是“全程 BF16”：

- 可训练学生以 FP32 载入，使优化器/master 参数保持 FP32；
- FSDP mixed precision 在前向/反向计算时把参数转换为 BF16，梯度归约与 buffer 保持 FP32；
- 冻结且只做前向的教师直接以 BF16 载入。

这是仓库内 vendored verl worker 的要求：其实现明确要求可训练 actor 先用 FP32 创建，否则优化器状态会错误地落到 BF16。不要为了省显存把 actor 的初始化 dtype 强改成 BF16；双卡适配依靠 FSDP 分片、activation checkpoint/offload 和缩小 preset，而不是牺牲优化器精度。

论文环境使用 FlashAttention/remove-padding 优化，但本机 535 驱动不能执行 PyTorch 2.8 配套 wheel 内的较新 CUDA cubin。此入口因此默认 `ATTN_IMPLEMENTATION=sdpa`、`USE_REMOVE_PADDING=false`，由 PyTorch SDPA 完成同一个 causal-attention 计算；这不改变 OPD 损失或 Top-k 支持集，但会降低吞吐，并且属于必须披露的基础设施差异。驱动升级并验证后，可显式设置 `ATTN_IMPLEMENTATION=flash_attention_2 USE_REMOVE_PADDING=true` 恢复论文的高效路径。

## 运行前：不要抢占正在使用的 GPU

先看 GPU 和进程：

```bash
nvidia-smi
ps -ef | grep -E '[r]ay|[v]llm|verl.trainer.main_ppo'
```

只在两张卡都确认为空闲时训练。若 GPU 正忙，停止本次运行并等待或换机器；不要用 `ray stop --force`、`pkill` 或 `kill` 清理不属于本实验的 Ray/vLLM/训练进程。启动脚本不会替你杀掉现有进程，preflight 对繁忙 GPU 的报错也不应被随意绕过。

以下命令都从仓库根目录执行：

```bash
cd /path/to/OPD
```

## 1. 安装与预检

安装隔离环境：

```bash
# 可先只打印安装动作
DRY_RUN=1 bash reproduce_4b/setup_env.sh

# 实际创建仓库内的 .venv-opd
bash reproduce_4b/setup_env.sh
source .venv-opd/bin/activate
```

安装器默认使用公共 PyPI，以避开宿主机可能注入但不可用的私有镜像；若环境要求使用内部镜像，请显式设置 `OPD_PIP_INDEX_URL=https://your-trusted-mirror/simple`。已验证的 Python/CUDA 依赖全部固定在 [`constraints-2xa100-cu128.txt`](constraints-2xa100-cu128.txt)，`setup_env.sh` 每次安装都会消费该约束；安装后仍额外保存 freeze 供差异审计。这只覆盖 PyPI 包；FlashAttention wheel 默认仍来自 GitHub，其默认 URL 已携带已验证的 SHA-256。离线或受限网络下先把 wheel 放到本机，再设置 `OPD_FLASH_ATTN_WHEEL=/absolute/path/to/flash_attn.whl`。

然后做 Fig. 2 pair 的预检：

```bash
python reproduce_4b/preflight.py \
  --repo-root . \
  --student Qwen/Qwen3-1.7B-Base \
  --student-revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
  --teacher lllyx/Qwen3-4B-Base-GRPO \
  --teacher-revision 1f3b2966edfb75f2f98a00617588c1f748088422 \
  --train-data datasets/dapo-math-17k.parquet \
  --gpus 2 --min-free-gib 70
```

预检会确认 Python/CUDA、依赖版本契约、数据 schema、原始或 teacher-aligned DAPO 模板、固定 revision 的 tokenizer/词表兼容性，并要求 Fig. 2 pair 的两张卡各至少空闲 70 GiB 且利用率低于阈值；4B 学生路径默认要求 72 GiB。必须恰好暴露两张唯一的数字编号 GPU；多卡机器先设置如 `CUDA_VISIBLE_DEVICES=2,3`，空值、重复编号和无法安全映射的 UUID/MIG selector 都会 fail closed。这个阈值的目的不是估算最低可运行显存，而是避免与别人的作业共享 GPU；同时在输出中人工确认型号确为 A100 80GB。它只做检查，不启动训练。

正式启动强制 `PIN_MODEL_SNAPSHOTS=true`：先按指南内固定 commit 下载不可变 Hub snapshot，再把本地绝对路径交给 VERL，并在 run 目录保存 `model_snapshots.json`。自定义模型默认把当时的 Hub HEAD 解析为 commit；也可显式提供 `STUDENT_REVISION` / `TEACHER_REVISION`。本地模型目录同样经过该入口验证并记录。由于当前 VERL 配置不会把 Hub revision 继续传给模型加载器，启动器会拒绝 `PIN_MODEL_SNAPSHOTS=false`，从而避免预检 revision 与训练权重悄悄分叉。

本机 NVIDIA 驱动与系统 `nvcc` 版本可能不一致。训练启动器和评测脚本均默认 `VLLM_USE_FLASHINFER_SAMPLER=0`，让 vLLM 使用等价的原生 PyTorch top-k/top-p sampler，避免 FlashInfer JIT 生成驱动无法加载的 cubin；训练侧同时使用上一节的 SDPA fallback。若主机驱动/toolkit 已正确配套，可显式恢复 FlashInfer/FlashAttention 做性能对比。

## 2. Dry-run、smoke、pilot 与 paper

先打印最终 Hydra 命令，不加载模型：

```bash
DRY_RUN=1 PRESET=smoke MODEL_PAIR=paper \
  bash reproduce_4b/run_opd_4b.sh
```

确认打印内容至少包含原始 DAPO、`token_reward_direct`、Top-16、`only_stu`、`student_p`、rollout seed、2 GPUs、actor FP32 初始化、teacher BF16，以及本机默认的 `sdpa/remove_padding=false`。Base 学生保留 `ENABLE_THINKING=auto`，dry-run 中没有 `enable_thinking=False` 是预期行为。然后按成本从低到高运行：

| preset | prompt batch | rollout/题 | 最大回复 | 数据/步数 | 用途 |
|---|---:|---:|---:|---:|---|
| `smoke` | 4 | 1 | 1,024 | 8 条、2 steps | 仅链路检查 |
| `pilot` | 8 | 2 | 4,096 | 1,600 条、200 steps | 机制曲线与消融 |
| `paper` | 64 | 4 | 7,168 | 全数据、1 epoch | 最接近论文主超参 |

```bash
# 一小步链路检查：只用于发现环境、显存和 checkpoint 问题
PRESET=smoke MODEL_PAIR=paper \
  bash reproduce_4b/run_opd_4b.sh

# 观察 overlap / advantage / entropy 动态的中等规模运行
PRESET=pilot MODEL_PAIR=paper \
  bash reproduce_4b/run_opd_4b.sh

# 双卡条件下最接近论文超参的完整主运行
PRESET=paper MODEL_PAIR=paper \
  bash reproduce_4b/run_opd_4b.sh
```

`smoke` 只能证明链路能跑，不能用于机制或分数结论；`pilot` 用于筛选配置和检查趋势；只有 `paper` 加论文评测设置才进入分数层验收。若出现 OOM，先降低当前 preset 的最大回复长度或 prompt batch，并在结果中标注 override；不要悄悄修改 Top-k、支持集或教师。

4B 学生迁移需另开实验名，并明确标为 migration：

```bash
PRESET=pilot MODEL_PAIR=4b EXPERIMENT_TAG=opd-4b-migration \
  bash reproduce_4b/run_opd_4b.sh
```

本机两步 4B smoke 实测峰值 allocated/reserved 约为 63.63/71.38 GiB 每卡。它证明当前最小链路能放入 80GB，但余量不大，不能据此保证更长回复、更大 batch 的 `pilot`/`paper` 不 OOM；扩展时持续观察显存并保留所有 override。

训练使用 file logger；日志是后续机制图的输入。不要依赖 vendored verl v0.7 的内置验证来报论文分数，本配置关闭训练期验证并用下面的独立评测脚本。

默认运行目录是 `artifacts/runs/<experiment_name>/`，其中 `metrics.jsonl` 是 file logger 输出，FSDP 分片在 `checkpoints/global_step_*/`。启动器还会保存实际命令、代码 revision、tracked diff、复现脚本/约束，以及本地新增的核心 `verl.utils.opd` helper 副本。它不会无差别打包其他 untracked 文件，以免将私有数据或密钥带入产物。可用 `RUN_ROOT`、`RUN_DIR`、`METRICS_FILE` 覆盖位置。

## 3. 检查训练机制

重点看这些原始 key：

| 指标 | 期望的成功动态 |
|---|---|
| `val-topk/overlap_ratio` | 逐步升高 |
| `val-topk/adv_intersection` | metric schema v2 严格按论文 Eq. (7) 在交集上对师生分布分别重归一化；成功时通常从负值靠近 0 |
| `val-topk/training_adv_intersection` | 实际策略 reward 在交集上的均值；`union-intersection` 对交集强制为 0，不能把该零值解释为收敛 |
| `actor/entropy`、`teacher/entropy` | 差距收窄，且不出现持续尖峰/塌缩 |
| `actor/grad_norm` | 保持有限且无后期爆炸 |
| `val-topk/student_p_sum_intersection`、`val-topk/teacher_p_sum_intersection` | 辅助确认交集覆盖双方主要概率质量 |

把一个或多个 file-logger JSONL 画在同一张图上：

```bash
SUCCESS_LOG=/path/to/paper-success/metrics.jsonl
MISMATCH_LOG=/path/to/paper-mismatch/metrics.jsonl
python reproduce_4b/plot_metrics.py \
  --input-jsonl "$SUCCESS_LOG" "$MISMATCH_LOG" \
  --labels success mismatch \
  --output artifacts/plots/success-vs-mismatch.png \
  --title 'Qwen3 OPD diagnostics'
```

用实际运行产生的日志路径替换示例路径；脚本会画 overlap ratio、overlap-token advantage、师生 entropy/gap 和 gradient norm。旧 smoke 产物中同名 `adv_intersection` 是作者代码的 training-reward proxy，没有 `opd/metric_schema_version=2`，不可与新 Eq. (7) 曲线直接混画；绘图脚本会默认拒绝这种混用，只有在明确接受口径差异时才使用 `--allow-mixed-metric-schema`。

## 4. 旧版 6-cell 机制快捷矩阵

所有对照应从同一学生初始化开始，固定数据、preset、rollout 数、学习率和评测设置，只改变表中的单一因素。本节只保留兼容旧工作流的 Qwen 快捷入口；新实验应使用 [`run_ablations.py`](run_ablations.py) 执行[正式 24-cell 矩阵](ABLATIONS_zh.md)。

仓库已经把下面三组对照汇总成矩阵。默认仅打印 6 条唯一的最终命令；`success_grpo_teacher` 同时作为 Top-16/Student-Top-k 基准，不会再重复跑一次。确认后才显式顺序执行：

```bash
ACTION=print PRESET=pilot bash reproduce_4b/run_matrix.sh
ACTION=run PRESET=pilot bash reproduce_4b/run_matrix.sh
```

`ACTION=run` 会顺序跑完整矩阵，耗时远高于单次 pilot，执行前再次确认 GPU 一直可独占。也可以用下面的单条命令逐项运行。

### 成功教师与 thinking-pattern mismatch

```bash
PRESET=pilot MODEL_PAIR=paper EXPERIMENT_TAG=paper-success \
  bash reproduce_4b/run_opd_4b.sh

PRESET=pilot MODEL_PAIR=mismatch EXPERIMENT_TAG=paper-mismatch \
  bash reproduce_4b/run_opd_4b.sh
```

第一条使用额外 GRPO 的 Base 教师；第二条换成 Qwen3-4B Non-thinking 权重，作为论文的 pattern-mismatch 对照；两条都保留同一个 Base 学生及其默认模板。不要仅凭教师 benchmark 更高就预测 OPD 会成功。

### 支持集消融

```bash
for strategy in only_stu intersection union-intersection; do
  PRESET=pilot MODEL_PAIR=paper TOP_K_STRATEGY="$strategy" \
    EXPERIMENT_TAG="support-${strategy}" \
    bash reproduce_4b/run_opd_4b.sh
done
```

`intersection` 只优化师生 Top-k 交集；`union-intersection` 是二者的对称差，即专门优化非重合 token。论文在 R1-Distill-1.5B→JustRL-1.5B 上观察到 overlap-only/完整 Student Top-k 明显优于 non-overlap；这里用 Qwen pair 运行属于跨模型机制迁移验证。

当前实现用 `SUPPORT_WEIGHT_NORMALIZATION` 显式区分两种口径。默认 `author` 与作者代码一致：`union-intersection` 保留未归一化的学生原始概率质量，因此同时改变 token coverage 与梯度尺度；`selected` 才会在选定支持集上重归一化，它属于默认关闭的稳健性扩展。正式 Fig. 7 使用 R1-Distill-1.5B→JustRL-1.5B pair 和 `author` 口径；本节 Qwen 循环只是跨模型快捷检查。同时保留了 `only_tch` 多 mini-batch 对齐修复，并在 PPO replay 前 fail closed 校验 shape。

### Top-k 大小

```bash
for k in 1 4 16; do
  PRESET=pilot MODEL_PAIR=paper TOP_K="$k" \
    EXPERIMENT_TAG="student-topk-${k}" \
    bash reproduce_4b/run_opd_4b.sh
done
```

论文在 R1-Distill-1.5B→JustRL-1.5B 上发现 Top-1 动态最不稳定，Top-16 是默认配置。这里用 Qwen pair 重跑同样属于跨模型机制迁移，而且旧循环不含论文的 sampled-token 与 Top-64；正式矩阵包含 `sampled/1/4/16/64` 五档。比较时同时看最终分数与训练曲线，不能只挑最高的单个 checkpoint。

## 5. 合并 checkpoint

verl 保存的是 FSDP 分片，不能把 actor 分片目录直接交给 vLLM。先选择要评估的 `global_step_*`，再合并：

```bash
STEP_DIR=artifacts/runs/opd-paper-paper-k16-only_stu-.../checkpoints/global_step_200
MERGED_MODEL=artifacts/models/paper-final-step200
bash reproduce_4b/merge_checkpoint.sh "$STEP_DIR" "$MERGED_MODEL"
```

保留原始分片、合并模型、训练命令和 metrics JSONL 的对应关系。若按验证结果选择 checkpoint，必须写下选择规则；论文没有充分披露 checkpoint 选择细节。

## 6. Baseline 与 final 的三项评测

论文评测是 AIME 2024、AIME 2025、AMC 2023 上的 **avg@16**：每题 `n=16`、temperature `0.7`、top-p `0.95`、最大生成 `31,744` tokens，并关闭 thinking。不要用 pass@16 替代论文主指标。

下面分别评估 Fig. 2 pair 的原始学生和合并后的最终学生。两卡必须空闲；评测不要与训练同时运行。

```bash
mkdir -p artifacts/eval/baseline-paper

for bench in AIME24 AIME25 AMC23; do
  python reproduce_4b/generate_eval.py \
    --model Qwen/Qwen3-1.7B-Base \
    --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
    --tokenizer-revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
    --input-parquet "datasets/test_data/${bench}/test.parquet" \
    --output-jsonl "artifacts/eval/baseline-paper/${bench}.jsonl" \
    --cuda-visible-devices 0,1 --tensor-parallel-size 2 \
    --n 16 --temperature 0.7 --top-p 0.95 --max-tokens 31744 \
    --thinking off --seed 42
  python reproduce_4b/grade_eval.py \
    --input-jsonl "artifacts/eval/baseline-paper/${bench}.jsonl" \
    --output-json "artifacts/eval/baseline-paper/${bench}.metrics.json" \
    --n 16 --strict-n
done

FINAL_MODEL=artifacts/models/paper-final-step200
FINAL_TOKENIZER=Qwen/Qwen3-1.7B-Base
FINAL_TOKENIZER_REVISION=ea980cb0a6c2ae4b936e82123acc929f1cec04c1
FINAL_EVAL_DIR=artifacts/eval/final-paper
mkdir -p "$FINAL_EVAL_DIR"
for bench in AIME24 AIME25 AMC23; do
  python reproduce_4b/generate_eval.py \
    --model "$FINAL_MODEL" \
    --tokenizer "$FINAL_TOKENIZER" \
    --tokenizer-revision "$FINAL_TOKENIZER_REVISION" \
    --input-parquet "datasets/test_data/${bench}/test.parquet" \
    --output-jsonl "$FINAL_EVAL_DIR/${bench}.jsonl" \
    --cuda-visible-devices 0,1 --tensor-parallel-size 2 \
    --n 16 --temperature 0.7 --top-p 0.95 --max-tokens 31744 \
    --thinking off --seed 42
  python reproduce_4b/grade_eval.py \
    --input-jsonl "$FINAL_EVAL_DIR/${bench}.jsonl" \
    --output-json "$FINAL_EVAL_DIR/${bench}.metrics.json" \
    --n 16 --strict-n
done
```

若评估的是 `MODEL_PAIR=4b` 的合并学生，必须同时换掉模型路径、tokenizer 与输出目录；不能把 1.7B checkpoint 配给 4B tokenizer：

```bash
FINAL_MODEL=artifacts/models/migration-4b-final-step200
FINAL_TOKENIZER=Qwen/Qwen3-4B-Base
FINAL_TOKENIZER_REVISION=906bfd4b4dc7f14ee4320094d8b41684abff8539
FINAL_EVAL_DIR=artifacts/eval/migration-4b
```

把这组赋值放在上方 final 评测循环之前，并确认 `FINAL_MODEL/config.json` 的模型类型与 4B checkpoint 一致。

直接评估教师仓库时必须显式提供 Base tokenizer；教师仓库自身可能没有完整 chat template：

```bash
for bench in AIME24 AIME25 AMC23; do
  python reproduce_4b/generate_eval.py \
    --model lllyx/Qwen3-4B-Base-GRPO \
    --revision 1f3b2966edfb75f2f98a00617588c1f748088422 \
    --tokenizer Qwen/Qwen3-4B-Base \
    --tokenizer-revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 \
    --input-parquet "datasets/test_data/${bench}/test.parquet" \
    --output-jsonl "artifacts/eval/teacher/${bench}.jsonl" \
    --cuda-visible-devices 0,1 --tensor-parallel-size 2 \
    --n 16 --temperature 0.7 --top-p 0.95 --max-tokens 31744 \
    --thinking off --seed 42
  python reproduce_4b/grade_eval.py \
    --input-jsonl "artifacts/eval/teacher/${bench}.jsonl" \
    --output-json "artifacts/eval/teacher/${bench}.metrics.json" \
    --n 16 --strict-n
done
```

完整论文评测很贵。快速回归只能显式缩放为 `n=4 / max_tokens=4096`，文件名和表格都必须标记 `scaled`，且不得与论文 avg@16 横向比较：

```bash
FINAL_MODEL=artifacts/models/paper-final-step200
FINAL_TOKENIZER=Qwen/Qwen3-1.7B-Base
FINAL_TOKENIZER_REVISION=ea980cb0a6c2ae4b936e82123acc929f1cec04c1
python reproduce_4b/generate_eval.py \
  --model "$FINAL_MODEL" \
  --tokenizer "$FINAL_TOKENIZER" \
  --tokenizer-revision "$FINAL_TOKENIZER_REVISION" \
  --input-parquet datasets/test_data/AIME24/test.parquet \
  --output-jsonl artifacts/eval/scaled-n4-max4096/AIME24.jsonl \
  --cuda-visible-devices 0,1 --tensor-parallel-size 2 \
  --n 4 --temperature 0.7 --top-p 0.95 --max-tokens 4096 \
  --thinking off --seed 42
python reproduce_4b/grade_eval.py \
  --input-jsonl artifacts/eval/scaled-n4-max4096/AIME24.jsonl \
  --output-json artifacts/eval/scaled-n4-max4096/AIME24.metrics.json \
  --n 4 --strict-n
```

4B 快速回归也要把这三个变量一起改为上一段的 `migration-4b` 模型/tokenizer 设置，并给输出目录保留 `scaled` 标识。

生成脚本默认值正是论文设置；上面仍显式写出，便于审计。示例对 Hub 模型和 tokenizer 传入不可变 commit，并显式记录 seed；也可以直接使用训练目录 `model_snapshots.json` 中的本地 snapshot path。脚本还与官方评测入口一致，把 `<|im_end|>` 和 `<|endoftext|>` 的 tokenizer ID 作为停止条件，避免 Base 模型越过 assistant turn 结束符继续生成。输出 JSONL 每条 response 都带 sampling metadata（含模型/revision、seed 和实际 stop IDs），评分脚本使用仓库原有数学答案抽取/等价判定，并同时输出 avg@N、pass@N 和格式率。

## 三层验收标准

### L1：工程跑通

- preflight 通过，dry-run 参数正确；
- `smoke` 无 OOM/NaN，loss、entropy、grad norm 为有限值；
- 至少保存一个 FSDP checkpoint，能合并并由 vLLM 生成答案。

这只能写成“链路跑通”。

### L2：机制复现

- 成功 pair 的 overlap ratio 上升、schema-v2 Eq. (7) `adv_intersection` 向 0 靠近、entropy gap 收窄；
- mismatch 教师的动态/增益弱于成功教师；
- `only_stu` 或 `intersection` 优于 `union-intersection`，Top-1 比 Top-16 更不稳定；
- 结论来自预先规定的多个指标与完整 pilot 区间，不是挑选单点。

这可以写成“复现了论文机制趋势”，仍不是论文分数复现。

### L3：分数复现

- 使用 `MODEL_PAIR=paper` 和记录完整 override 的 `paper` 运行；
- baseline/final/teacher 都在三项数据集上以 n=16、0.7/0.95、31,744 tokens 评测并报告 avg@16；
- 将最终结果与论文 Fig. 2/17 的 Qwen 学生曲线比较，同时披露双卡缩放、checkpoint 规则和运行次数；论文没有表列该 pair 的精确终点，数字化读图只能标为近似值，不能把 Fig. 15 的 R1-Distill→JustRL Top-16 数值 `0.458/0.338/0.791` 当作 Qwen 目标；
- 结果文件、日志、模型与命令可一一追溯。

只有达到这一层，才能讨论“数值复现程度”；未跑完整训练时必须明确写“尚未完成 full run”。`MODEL_PAIR=4b` 无论分数多高，都只能称 4B 迁移实验。

## 最小结果记录

每次运行至少保存：git commit/diff、完整启动命令、环境与 GPU 型号、`model_snapshots.json` 中的模型 revision/路径、数据文件校验值、preset/override、随机种子、metrics JSONL、checkpoint step、三项评测 JSON/JSONL，以及是否为 `scaled` 评测。建议结果表中单列：

```text
run_id | model_pair | preset | teacher | topk | support | train_steps
       | AIME24 avg@N | AIME25 avg@N | AMC23 avg@N | N | max_tokens | scaled
```

这能防止把 smoke、快速评测、4B migration 或 cherry-picked checkpoint 混入论文主结果。
