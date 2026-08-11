# 2×A100 实机验证记录

本记录只证明 OPD 的最小端到端链路（L1）在当前机器上可运行，不代表论文分数或机制趋势已经复现。对应运行时间为 2026-08-02 UTC。

## 共同环境与科学边界

- GPU：2×NVIDIA A100-SXM4-80GB；训练前每卡约 79.1 GiB 可用；
- 环境：Python 3.12、PyTorch 2.8.0+cu128、Transformers 4.57.6、vLLM 0.11.0、Ray 2.56.1、vendored verl 0.7.0.dev；
- 教师：`lllyx/Qwen3-4B-Base-GRPO@1f3b2966edfb75f2f98a00617588c1f748088422`；
- 数据：原始 DAPO 与 teacher-aligned processed DAPO 均为 17,917 行；SHA-256 分别为 `039f3afd689c846985bd2bf58e55a2210a8b08a1a9e1f60dbc07107a5f341925` 与 `500bd8c45eca355b98f9ba6f3213194a72bd42c73c5e9569c6fbbb1b51bd0b39`；
- 本机兼容路径：PyTorch SDPA、`use_remove_padding=false`、vLLM PyTorch sampler。

当前默认路径已用论文 Fig. 2 的原始 DAPO 模板完成一次 1.7B→4B 两步 smoke；更早的两次训练使用 §5.2 的 teacher-aligned processed prompt，只作为 aligned-recipe 工程验证。原始模板运行显式记录 `data.seed=42` 与 `rollout.seed=42`；两个历史 aligned 运行使用 `data.seed=42` 和当时 rollout engine 的内部默认 seed 0，当前入口可用 `SEED=42 ROLLOUT_SEED=0` 精确表达。三次运行都只是 L1 链路验证，不是 Fig. 2 数值或趋势复现。

## Fig. 2 原始模板：1.7B 学生 → 4B 教师

实跑命令：

```bash
PYTHON_BIN="$PWD/.venv-opd/bin/python" \
PRESET=smoke MODEL_PAIR=paper \
EXPERIMENT_TAG=fig2-raw-eq7-smoke-validation \
bash reproduce_4b/run_opd_4b.sh
```

产物目录：

```text
artifacts/runs/opd-fig2-raw-eq7-smoke-validation-Qwen3-1.7B-Base-to-Qwen3-4B-Base-GRPO-20260802T174424Z/
```

学生为 `Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1`；`preflight.log` 确认读取 17,917 条 original-DAPO prompt，且数据哈希与上文一致。训练完成 2/2 step，`metrics.jsonl` 含两条可解析记录，`global_step_2` 保存了完整 FSDP 模型/优化器分片与数据状态：

| step | overlap ratio | Eq. (7) advantage | training-reward proxy | student / teacher entropy | grad norm | max allocated / reserved GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.777057 | -0.015507 | -0.014520 | 1.642774 / 1.571740 | 8.369473 | 42.1947 / 47.5801 |
| 2 | 0.736289 | -0.011421 | -0.010565 | 0.767258 / 0.680959 | 8.562602 | 42.1947 / 50.8027 |

这些值只诊断 rollout、Top-16、冻结教师 forward、reverse-KL token advantage 与 actor 更新链路是否接通；两步数据不足以支持趋势或质量结论。`opd/metric_schema_version=2` 表明 Eq. (7) 列在交集上对师生分布分别重归一化；training-reward proxy 则是实际优化张量的交集均值，二者不混用。运行目录中的 `command.sh` 已解析为固定本地 snapshot 绝对路径，并保存训练所需环境变量；`tracked_changes.patch` 加 `reproduction_code/verl/verl/utils/opd.py` 可重建当时核心修补。诊断图已实际生成为该目录下的 `diagnostics.png`。

## 历史 aligned 验证：1.7B 学生 → 4B 教师

等价复跑命令：

```bash
PYTHON_BIN="$PWD/.venv-opd/bin/python" \
TRAIN_DATA="$PWD/datasets/dapo-math-17k-processed.parquet" \
SEED=42 ROLLOUT_SEED=0 PRESET=smoke MODEL_PAIR=paper \
EXPERIMENT_TAG=paper-smoke-final-validation \
bash reproduce_4b/run_opd_4b.sh
```

历史产物目录：

```text
artifacts/runs/opd-paper-smoke-final-validation-Qwen3-1.7B-Base-to-Qwen3-4B-Base-GRPO-20260802T164133Z/
```

训练完成 2/2 step，两个 step 均完成 rollout、Student Top-16、冻结教师打分、reverse-KL token advantage、actor 反向传播与参数更新。`metrics.jsonl` 有两条可解析记录：

| step | overlap ratio | legacy training-reward proxy | student entropy | teacher entropy | grad norm | max allocated GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.687149 | -0.013950 | 0.332933 | 0.268756 | 4.677499 | 42.1937 |
| 2 | 0.742991 | -0.010243 | 0.802602 | 0.744065 | 4.115252 | 42.1947 |

这些是 8 个样本、两步训练的链路诊断值，不能据此判断趋势或模型质量。该历史日志产生于 metric schema v2 修补之前，同名 `adv_intersection` 是 training-reward proxy，不能与上表 Eq. (7) 列直接比较。第二步最大 reserved memory 为 50.4316 GiB/卡。checkpoint 已保存并成功合并到标准 Hugging Face 目录；对合并权重的 BF16 CUDA 前向得到 1,720,574,976 个参数、`[1, 8, 151936]` logits，且所有 logits 为有限值。

## 历史 aligned 验证：4B 学生 → 4B 教师

等价复跑命令：

```bash
PYTHON_BIN="$PWD/.venv-opd/bin/python" \
TRAIN_DATA="$PWD/datasets/dapo-math-17k-processed.parquet" \
SEED=42 ROLLOUT_SEED=0 PRESET=smoke MODEL_PAIR=4b \
EXPERIMENT_TAG=4b-migration-smoke-validation \
bash reproduce_4b/run_opd_4b.sh
```

历史产物目录：

```text
artifacts/runs/opd-4b-migration-smoke-validation-Qwen3-4B-Base-to-Qwen3-4B-Base-GRPO-20260802T164955Z/
```

学生为 `Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539`。训练同样完成 2/2 step：

| step | overlap ratio | legacy training-reward proxy | student entropy | teacher entropy | grad norm | max allocated / reserved GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.915288 | -0.002065 | 0.794554 | 0.706020 | 2.568636 | 63.6267 / 70.3652 |
| 2 | 0.881885 | -0.002894 | 0.811853 | 0.727281 | 2.712518 | 63.6267 / 71.3809 |

`global_step_2` 包含完整 FSDP 模型/优化器分片，并成功合并为 4,411,424,256 参数、BF16 safetensors 的 Hugging Face checkpoint。两个权重分片的 SHA-256 分别为 `c89c53c59ddd13dff690346d0ac29b74d674bc536c3622ca4ee8533f0e296832` 与 `257fa392f2f43beb1396bc3a5bfb8fcee8dbf025cd931eba6883e9bb6dde5fdd`。

随后用当前 `generate_eval.py` 与 vLLM 0.11.0 对该合并目录实际生成 1 条 AIME24 非空 smoke response。[精确命令](../artifacts/runs/opd-4b-migration-smoke-validation-Qwen3-4B-Base-to-Qwen3-4B-Base-GRPO-20260802T164955Z/vllm_smoke_provenance_command.sh) 显式指向 `global_step_2/actor_hf`；[输出 JSONL](../artifacts/runs/opd-4b-migration-smoke-validation-Qwen3-4B-Base-to-Qwen3-4B-Base-GRPO-20260802T164955Z/vllm_smoke_provenance.jsonl) 内嵌同一模型绝对路径、4B tokenizer commit、`n=1`、temperature 0.7、top-p 0.95、seed 42、thinking off 与停止 token IDs `[151645, 151643]`。该 response 在 64 tokens 处因 `finish_reason=length` 被截断，所以它只证明“训练→保存→合并→加载→非空生成”的 L1 链路，不是完整答案或评测得分。

训练退出时 TorchData 在对象析构阶段打印过 worker 被终止的 `Exception ignored` 警告；它发生在进度 2/2、指标写入和 checkpoint 保存之后，没有造成训练、合并或 vLLM 生成失败。未安装 SwanLab 还会产生非致命的候选计数绘图提示；正式诊断图由 `plot_metrics.py` 从 JSONL 生成。

## 尚未验证的范围

- 没有执行 `pilot` 或完整 `paper` preset；
- 没有跑 AIME24/AIME25/AMC23 的 n=16 完整评测；
- Fig. 2 原始 DAPO 模板只跑了两步 smoke，没有跑长训练；
- 因此当前验收等级是 L1，而不是论文机制趋势（L2）或分数复现（L3）。
