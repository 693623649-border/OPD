# Rethinking OPD：2×A100 轻量复刻版

本文给出针对当前机器的可执行轻量方案。论文原文见[本地 PDF](../paper/rethinking_opd.pdf)，完整论文结构复现见[总入口](./FULL_REPRODUCTION_zh.md)。轻量版的机器定义是[`lite_matrix.json`](./lite_matrix.json)，运行入口是[`run_lite.sh`](./run_lite.sh)。

> 这是一套用于验证论文主要方向和工程机制的 `local-lite/pilot` 实验，不是论文数值复现。40-step 曲线、`avg@4` 和缩短后的 response length 均不得与论文 `paper/avg@16` 结果直接比较。

## 1. 当前设备与设计结论

2026-08-03 实时探测到：

| 项目 | 当前状态 | 设计影响 |
|---|---:|---|
| GPU | 2×NVIDIA A100-SXM4-80GB | 所有训练串行独占两卡 |
| 每卡空闲显存 | 约 79.15 GiB，利用率 0% | 普通组要求至少 70 GiB；7B teacher 组要求 72 GiB |
| 共享盘剩余 | 约 508 TB | 磁盘不是当前瓶颈，但仍只保存 model-only endpoint |
| 已有 artifacts | 约 174 GB | lite suite 使用独立目录，不改写历史结果 |
| 模型缓存 | 约 66 GB | 主要公开模型已缓存，仍固定 revision 并执行 preflight |

已有 smoke 的实测边界如下。`perf/max_memory_reserved_gb` 是 PyTorch allocator 指标，不是同步的 `nvidia-smi` 单卡物理采样；它适合用于相对风险排序，不能解释为某一时刻的单卡真实占用。

| 条件 | allocator reserved | 实测 step 耗时 | 决策 |
|---|---:|---:|---|
| 普通 1.5B/1.7B student | 约 49–54 GiB | 约 11–20 s | 默认纳入 |
| 3K response | 约 54.45 GiB | 约 24.7 s | Extended 纳入 |
| 7K response | 约 65.0 GiB | 约 43.9 s | Extended 最后运行 |
| 10K response | 约 72.92 GiB | 约 65.7 s | 排除 |
| 15K response | allocator 约 86.24 GiB | 约 94 s | 视为危险信号，排除 |
| 4B student OPD | 约 71.38 GiB | smoke 级证据 | 不进入 40-step 默认矩阵 |

因此最稳妥的方案不是把 student 扩到 4B，而是保留论文原始 1.5B/1.7B student；teacher 默认不超过 4B，Extended 只保留一个已经通过 smoke 的 7B teacher 对照。

## 2. 两档训练矩阵

### Core：7 cells，默认运行

| 论文问题 | Cells | 固定不变量 | 能观察什么 |
|---|---|---|---|
| Fig. 2：Qwen teacher condition | `fig2-compatible-grpo`、`fig2-mismatch-nonthinking` | 同一 Qwen3-1.7B-Base student、DAPO、Student Top-16 | 两个论文复合 teacher 条件的早期 OPD 分离趋势 |
| Fig. 7：token support | `fig7-student-topk`、`fig7-overlap-topk`、`fig7-nonoverlap-topk-author` | 同一 R1-1.5B student、JustRL teacher、Top-16 | overlap 与 non-overlap support 对训练信号的差异 |
| Fig. 15/16：sampled vs Top-16 | `fig16-topk-sampled`、`fig16-topk-16` | 同一模型、数据和 Student support | sampled-token 与高概率 Top-16 的早期稳定性差异 |

Core 注册 7 个 cell role，全部使用两卡、40 steps、batch 4、rollout `n=1`、最大回复 2,048 tokens。它优先保留论文的“现象→token 机制”主线。`fig6-deepseek-justrl-success`（Extended）、`fig7-student-topk` 与 `fig16-topk-16` 是同一科学配置的重复物理执行，只用于各自组内配对，不能算作三个独立条件或 seed。因此 Core 7 cells 对应 6 个独立科学条件，Core + Extended 14 cells 对应 12 个。

### Extended：再增加 7 cells

| 论文问题 | Cells | 轻量化方式 | 边界 |
|---|---|---|---|
| Fig. 4/6/18/19：成功/失败 teacher | `fig4-6-deepseek-r1-7b`、`fig6-deepseek-justrl-success` | 40 steps、2K response；7B teacher 条件提高空闲显存门槛到 72 GiB | JustRL 使用可访问 `hbx` 镜像，未证明与受限作者 revision byte-identical |
| Fig. 8/21：cold start | `fig8-base-only-opd`、`fig8-sft-then-opd` | 直接比较公开 Base/SFT checkpoint，不重做 200K SFT | 只验证 released checkpoint 的早期效果 |
| Fig. 11–13/23：response length | `fig12-length-1024`、`fig12-length-3072`、`fig12-length-7168` | batch 2、`n=1`、40 steps；step 20 后每 5 steps 记录 position entropy | 只能得到三点早期长度趋势，不能证明 10K/15K instability 或 200–260 step 传播 |

Core + Extended 共 14 cells。Extended 不会替代 Core；同一 suite 下已完成的 Core cells 会被 runner 跳过，只执行剩余项。

## 3. 固定训练合同

| 参数 | Core 普通组 | Extended 长度组 | 论文口径差异 |
|---|---:|---:|---|
| Student | 1.5B/1.7B | 1.5B | 未做 4B scale transfer |
| GPUs | 2×A100-80GB | 2×A100-80GB | 论文部分上游使用 8×A800 |
| Steps | 40 | 40 | 论文通常约 200，Fig. 5 为 600 |
| Prompt batch | 4 | 2 | 论文默认 64 |
| Rollout `n` | 1 | 1 | 论文默认 4 |
| Response length | 2,048 | 1,024/3,072/7,168 | 论文默认 7,168，长度图还含 10K/15K |
| Train samples | 160 | 80 | 只覆盖小型固定窗口 |
| LR | `1e-6` | `1e-6` | 保留论文披露值 |
| Support | Student Top-16，按 cell 消融 | Student Top-16 | 保留作者语义 |
| Checkpoint | step 40，model-only，保留 1 个 | 同左 | 不保存 optimizer/RNG resume state |
| Seed | 42 | 42 | 单 seed，只能报告趋势 |

模型 revision、数据行数和 SHA、thinking mode、support direction、reward weighting 与作者 normalization 都沿用固定完整矩阵。轻量版只缩减计算预算，并使用独立 suite ID 与 matrix SHA，避免污染完整 v3 的 provenance。

当前 lite matrix SHA-256：

```text
72feb9598bf34d5aae1e266ffe52f8cd5b8bd8c227014d36cca98b55aa5925f5
```

### 3.1 已完成的代表 calibration

`fig7-student-topk` 已按 lite `calibration` 合同实际运行，不是 dry-run：

| 验收项 | 实测结果 |
|---|---:|
| 状态 | completed，exit 0，10/10 steps |
| 总耗时 | 270.445 s |
| Step timing | 中位 16.740 s，均值 17.257 s |
| 最大 allocator reserved | 52.7148 GiB |
| 指标有限性 | 10 行全部数值 finite |
| Step 10 overlap ratio | 0.748596 |
| Response clip ratio | 1.0 |
| 验收时 GPU | 两卡各 4 MiB used、0% util |

证据见[status](../artifacts/ablations/rethinking-opd-lite-2xa100-v1/calibration/seed-42/fig7-student-topk/status.json)、[metrics](../artifacts/ablations/rethinking-opd-lite-2xa100-v1/calibration/seed-42/fig7-student-topk/attempt-0001/metrics.jsonl)和[聚合结果](../artifacts/ablations/rethinking-opd-lite-2xa100-v1/calibration/seed-42/results.json)。缺少 `swanlab` 只导致候选计数辅助绘图警告；Ray 退出时 DataLoader worker teardown 也打印了 warning，但训练进程 exit 0、最终 step 10 完整落盘。GPU 数字是 calibration 完成后的验收时实时状态，并非 attempt 内固化的遥测。

`clip_ratio=1.0` 意味着这批 rollout 全部达到 2K 上限。因此 Core 的在线指标只能解释为“2K 截断训练目标下、学生实际访问的固定前缀机制趋势”，它不是论文 7K OPD 前 2K 的无偏代理，也不能解释为完整 reasoning trajectory、EOS 行为或解题正确率。输出效果由独立的 **4K-capped** `avg@4` 锚点评测观察，并须同时报告 finish reason/4K clip rate；长度敏感性则只由 Extended 内 batch 2、80 samples 的 1K/3K/7K 三点 sweep 报告，不能混入 Core 的 batch 4/160-sample 2K cell。

## 4. 推荐运行顺序

所有命令从仓库根目录执行。先确认两张卡仍为空闲：

```bash
cd /vepfs-mlp2/queue010/20262202674/OPD
nvidia-smi
ps -ef | grep -E '[r]ay|[v]llm|verl.trainer.main_ppo' || true
sha256sum reproduce_4b/lite_matrix.json
```

不要删除 lock、`pkill` 别人的进程或降低 70/72 GiB preflight 门槛。

### 4.1 只读计划

```bash
bash reproduce_4b/run_lite.sh plan core pilot
bash reproduce_4b/run_lite.sh plan extended pilot
```

首行应分别显示 `cells=7` 和 `cells=14`，suite 为 `rethinking-opd-lite-2xa100-v1`。

### 4.2 Core smoke

如果需要在新的 lite suite 下重新确认所有分支：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproduce_4b/run_lite.sh \
  run core smoke --yes --keep-going
```

这只运行 2 steps，且不保存 checkpoint。

### 4.3 高风险 Extended calibration

先校准 7B teacher 和 7K response；两项会串行执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproduce_4b/run_lite.sh \
  run extended calibration \
  --cell fig4-6-deepseek-r1-7b \
  --cell fig12-length-7168 \
  --yes --keep-going
```

每项 10 steps，不保存 checkpoint。只有二者状态均为 completed、指标有限且没有 OOM，才继续 Extended pilot。

`run_lite.sh` 会在真正启动 Extended pilot 前自动执行这一门禁：核对 calibration 的 matrix/source fingerprint、seed、exit code、最终 step 和所有数值有限性；任一条件缺失都会在创建 pilot 目录或占用 GPU 前退出。当前这两项仍为 pending，因此 Extended pilot 被有意锁定，Core pilot 不受影响。

### 4.4 Core pilot

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproduce_4b/run_lite.sh \
  run core pilot --yes --keep-going
```

### 4.5 Extended pilot

```bash
CUDA_VISIBLE_DEVICES=0,1 bash reproduce_4b/run_lite.sh \
  run extended pilot --yes --keep-going
```

若 Core 已完成，runner 会核对 fingerprint 后跳过它们。失败 cell 必须显式增加 `--retry-failed` 创建新 attempt；不会覆盖或 resume 旧 attempt。

### 4.6 状态与聚合

```bash
bash reproduce_4b/run_lite.sh status extended pilot

SUITE_ROOT="$PWD/artifacts/ablations/rethinking-opd-lite-2xa100-v1/pilot/seed-42"
"$PWD/.venv-opd/bin/python" reproduce_4b/aggregate_ablations.py \
  --suite-root "$SUITE_ROOT" \
  --output-json "$SUITE_ROOT/results.json" \
  --output-csv "$SUITE_ROOT/results.csv" \
  --plot-dir "$SUITE_ROOT/plots"
```

聚合表重点查看 overlap ratio、Eq. (7) advantage、Eq. (8) entropy gap、双方 overlap probability mass、gradient norm、response length 和显存；`score` 与 `actor/pg_loss` 需在各 attempt 的原始 `metrics.jsonl` 中查看。不能只凭一个最终分数判断论文机制是否复现。

runner 的 `completed` 状态只代表进程 exit 0 且至少写出一行 metric。正式验收还必须人工或脚本确认 `last_metric_step=40`、所有 metric finite、`global_step_40` model-only checkpoint 存在；缺少任一项都不能进入评测。Extended calibration 门禁已经自动执行对应的 10-step/finite 检查。

## 5. 轻量评测

默认只评测四个 Core 锚点：

- `fig2-compatible-grpo`
- `fig2-mismatch-nonthinking`
- `fig16-topk-sampled`
- `fig16-topk-16`

评测使用 AIME24、AIME25、AMC23 全量 143 题，每题 4 个样本，temperature 0.7、top-p 0.95、thinking off、最大回复 4,096 tokens，报告各 benchmark `avg@4` 和三项非加权 macro mean。DeepSeek-family 模板本身仍会以 `<think>` 起始，因而 `thinking off` 是评测合同元数据而非跨模型族统一的 non-thinking 模式；它不影响同模型族的成对比较。

```bash
PY="$PWD/.venv-opd/bin/python"
SUITE_ROOT="$PWD/artifacts/ablations/rethinking-opd-lite-2xa100-v1/pilot/seed-42"

for CELL in \
  fig2-compatible-grpo \
  fig2-mismatch-nonthinking \
  fig16-topk-sampled \
  fig16-topk-16
do
  RUN_DIR="$("$PY" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["run_dir"])' \
    "$SUITE_ROOT/$CELL/status.json")"

  CUDA_VISIBLE_DEVICES=0,1 "$PY" reproduce_4b/evaluate_ablation.py run \
    --run-dir "$RUN_DIR" \
    --checkpoint-step 40 \
    --n 4 \
    --max-tokens 4096 \
    --seed 42 \
    --yes --acknowledge-full-eval
done
```

评测计划和 sampling identity 是 write-once。不要先在同一个 pilot run-dir 注册 `--limit` 或不同 `n/max-tokens` 的临时计划，否则后续不能静默改成上述正式 lite 口径。

四个锚点合计最多生成 2,288 个响应、约 9.37M 输出 token 上限。保守估计需要约 4–10 小时墙钟；训练和评测不要并行抢占同一两张卡。

## 6. 时间与存储预算

| 阶段 | 预计墙钟 | 预计 checkpoint 存储 |
|---|---:|---:|
| Core 7 cells × 40 steps | 约 2–4 小时 | 约 50–60 GiB |
| Extended 额外 7 cells | 约 3–4 小时 | 再增加约 50–60 GiB |
| 全部 14-cell 训练 | 保守 6–8 小时 | 约 0.12 TiB |
| 4 个锚点 `avg@4` | 约 4–10 小时 | merged 权重约 15–20 GiB；响应远小于 1 GiB |
| 失败重试与余量 | — | 建议总预留 0.5 TiB |

这些是依据两步 smoke 的加载时间、step timing 和 allocator 指标做出的保守估计，不是正式 benchmark。实际时间会受生成长度、模型缓存、checkpoint 写盘和共享文件系统波动影响。

## 7. 能支持与不能支持的结论

如果 40-step 方向一致，轻量版最多可以报告：

- 当前两卡机器上观察到论文 teacher 条件的早期趋势差异；
- overlap/non-overlap token support 的训练信号不同；
- sampled-token 与 Student Top-16 的早期稳定性不同；
- Extended 中 released SFT cold start、成功/失败 teacher 和三点长度 sweep 的本地趋势。

它不能支持：

- 论文绝对分数、完整训练曲线或 `avg@16` 结论；
- 单独把 Fig. 2 差异归因于 thinking 开关；
- 多 seed 显著性、严格因果必要性或最佳超参数；
- 未公开 Qwen RL-Math 条件、Fig. 5 的 600-step reverse distillation；
- 10K/15K 长序列和 step 200–260 的 suffix-to-prefix entropy propagation；
- Top-1/4/64 的完整饱和曲线；
- prompt-content/template 全部 recipe、Fig. 11(b)、Fig. 14 或 Fig. 20；
- Table 1/3 上游训练的端到端论文级重建；
- `hbx` JustRL 镜像与受限作者 checkpoint 的逐字节一致性。

Fig. 7 三个 support cell 不在默认四锚点 `avg@4` 中，因此该组只能报告在线训练信号和前缀对齐动态，不能声称某种 support 带来更高解题性能。Extended 中的“successful/failing teacher”也只是沿用论文标签，本地没有默认 endpoint accuracy 去重新验证该标签或 cold-start 性能恢复。

任何成对 cell 若出现明显不同的 `clip_ratio` 或有效长度分布，总体 overlap、entropy 和 advantage 都会受到 token-position composition 混杂。此时只能比较共同覆盖的 0–1K chunk；若 chunk 级长度分布仍不匹配，就停止该对照的科学解释，而不是把位置组成差异归因于论文机制。4K 评测亦须披露 pair 间 clip rate，避免把长度效应误读成准确率效应。

结果统一标记为 `local-lite/pilot/seed-42` 和 `avg@4`，不得写成 `paper reproduction`。
