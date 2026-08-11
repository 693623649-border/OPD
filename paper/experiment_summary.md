# Rethinking OPD — 科学复现实验总结

**生成时间**: 2026-08-07 18:10:21 UTC
**硬件**: 2×A100-80GB
**论文**: *Rethinking On-Policy Distillation of Large Language Models* (arXiv 2604.13016v2)
**套件**: `rethinking-opd-scientific-2xa100-v1`
**W&B Dashboard**: https://wandb.ai/693623649-/Rethinking-OPD-2xA100

---

## 1. 总览

| 阶段 | 完成/总计 | 状态 |
|------|-----------|------|
| 硬件门禁 | 5/5 | ✅ 全部通过 |
| 训练 | 28/28 | ✅ 全部完成 |
| 趋势评测 (trend-n4) | 0/28 | ❌ 脚本缺陷已修复，待重运行 |
| 精确评测 (exact-avg16) | 0/28 | 🔄 第 1 cell 运行中 |
| Fig.11(b)/Fig.14 探针 | — | ⏳ 排队 |
| 渲染 + 审计 | — | ⏳ 排队 |

**训练总时间**: ~59 小时 (2026-08-05 00:41 → 2026-08-07 10:48 UTC)

---

## 2. 硬件门禁 (5/5 ✅)

| 门禁单元 | 验证内容 | 退出码 |
|----------|----------|--------|
| gate-length-10k | 10K 响应长度内存 | 0 |
| gate-length-15k | 15K 响应长度内存上限 | 0 |
| gate-qwen-4b-teacher | 4B 教师模型加载 | 0 |
| gate-r1-7b-teacher | 7B 教师模型加载 | 0 |
| gate-topk-64 | Top-k=64 全词表覆盖 | 0 |

---

## 3. 全部训练单元 (28/28 ✅)

### 长度消融组 (Fig.11/13) — 6/6 ✅

| 单元 | 最后步 | 峰值/GPU | 里程碑 | 中止率 |
|------|--------|---------|--------|--------|
| fig12-length-1024 | 260/260 | 41.4 GiB | 8 | 0.0 |
| fig12-length-10240 | 260/260 | 51.3 GiB | 8 | 0.0 |
| fig12-length-15360 | 260/260 | 64.6 GiB | 8 | 0.0 |
| fig12-length-3072 | 260/260 | 41.4 GiB | 8 | 0.0 |
| fig12-length-512 | 260/260 | 41.4 GiB | 8 | 0.0 |
| fig12-length-7168 | 260/260 | 43.3 GiB | 8 | 0.0 |

### Top-k 消融组 (Fig.15/16) — 5/5 ✅

| 单元 | 最后步 | 峰值/GPU | 里程碑 | 中止率 |
|------|--------|---------|--------|--------|
| fig16-topk-1 | 260/260 | 43.3 GiB | 8 | 0.0 |
| fig16-topk-16 | 260/260 | 43.3 GiB | 8 | 0.0 |
| fig16-topk-4 | 260/260 | 43.3 GiB | 8 | 0.0 |
| fig16-topk-64 | 260/260 | 43.3 GiB | 8 | 0.0 |
| fig16-topk-sampled | 260/260 | 41.4 GiB | 8 | 0.0 |

### 教师模式组 (Fig.2/6) — 2/2 ✅

| 单元 | 最后步 | 峰值/GPU | 里程碑 | 中止率 |
|------|--------|---------|--------|--------|
| fig2-compatible-grpo | 200/200 | 42.5 GiB | 5 | 0.0 |
| fig2-mismatch-nonthinking | 200/200 | 42.5 GiB | 5 | 0.0 |

### DeepSeek 教师组 (Fig.4/6) — 3/3 ✅

| 单元 | 最后步 | 峰值/GPU | 里程碑 | 中止率 |
|------|--------|---------|--------|--------|
| fig4-6-deepseek-r1-7b | 200/200 | 46.1 GiB | 5 | 0.0 |
| fig4-deepseek-skywork-rl | 200/200 | 46.1 GiB | 5 | 0.0 |
| fig6-deepseek-justrl-success | 200/200 | 43.3 GiB | 5 | 0.0 |

### Qwen 公开教师组 (Fig.4) — 1/1 ✅

| 单元 | 最后步 | 峰值/GPU | 里程碑 | 中止率 |
|------|--------|---------|--------|--------|
| fig4-qwen-nonthinking | 200/200 | 42.5 GiB | 5 | 0.0 |

### Reverse OPD 组 (Fig.5) — 2/2 ✅

| 单元 | 最后步 | 峰值/GPU | 里程碑 | 中止率 |
|------|--------|---------|--------|--------|
| fig5-reverse-r1-1p5b | 600/600 | 43.3 GiB | 6 | 0.0 |
| fig5-reverse-r1-7b | 600/600 | 46.1 GiB | 6 | 0.0 |

### 支持组 (Fig.7) — 3/3 ✅

| 单元 | 最后步 | 峰值/GPU | 里程碑 | 中止率 |
|------|--------|---------|--------|--------|
| fig7-nonoverlap-topk-author | 200/200 | 43.3 GiB | 5 | 0.0 |
| fig7-overlap-topk | 200/200 | 43.3 GiB | 5 | 0.0 |
| fig7-student-topk | 200/200 | 43.3 GiB | 5 | 0.0 |

### 冷启动组 (Fig.8) — 2/2 ✅

| 单元 | 最后步 | 峰值/GPU | 里程碑 | 中止率 |
|------|--------|---------|--------|--------|
| fig8-base-only-opd | 200/200 | 42.5 GiB | 5 | 0.0 |
| fig8-sft-then-opd | 200/200 | 42.5 GiB | 5 | 0.0 |

### 提示模板组 (Fig.9) — 2/2 ✅

| 单元 | 最后步 | 峰值/GPU | 里程碑 | 中止率 |
|------|--------|---------|--------|--------|
| fig9-template-original | 200/200 | 43.3 GiB | 5 | 0.0 |
| fig9-template-paper-aligned | 200/200 | 43.3 GiB | 5 | 0.0 |

### 提示内容组 (Fig.10) — 2/2 ✅

| 单元 | 最后步 | 峰值/GPU | 里程碑 | 中止率 |
|------|--------|---------|--------|--------|
| fig10-content-dapo-matched | 200/200 | 42.5 GiB | 5 | 0.0 |
| fig10-content-deepmath | 200/200 | 42.5 GiB | 5 | 0.0 |

---

## 4. 基础设施质量

| 指标 | 结果 | 门限 |
|------|------|------|
| 退出码 | 全部 = 0 | = 0 |
| 中止率 | 0.0% | ≤ 1% |
| 内存峰值 | 64.6 GiB | ≤ 78 GiB |
| 非有限指标 | 0 | 0 |
| OOM/NCCL | 0 | 0 |
| 里程碑完整性 | 100% | 100% |
| 离线合约 | HF_HUB_OFFLINE=1 ✅ | — |

---

## 5. 评测状态

### 趋势评测 (trend-n4) — 已修复，待重运行

**缺陷**: (1) `--acknowledge-full-eval` 未转发 trend 层级; (2) trend_steps 含未保存的中间步

**修复**: 所有层级转发 flag; 新增 `_available_checkpoint_steps()` 自动过滤

**测试**: 194/194 通过

### 精确评测 (exact-avg16) — 运行中 🔄

**规格**: n=16, max_tokens=31744, temperature=0.7, thinking=off

**进度**: 第 1 cell (`fig10-content-dapo-matched`), AIME24/AIME25 完成, AMC23 进行中

**规模**: 28 cells × 3 数据集 × 16 样本 ≈ 4,800 条生成

---

## 6. W&B 上传

**Dashboard**: [Rethinking-OPD-2xA100](https://wandb.ai/693623649-/Rethinking-OPD-2xA100)

**上传**: 28 个 cell 的完整 metrics.jsonl, 配置, GPU 峰值, 里程碑计数
**指标**: actor/entropy, actor/pg_loss, opd/abs_entropy_gap, critic/rewards, perf/throughput 等

---

## 7. 论文覆盖范围

| 章节 | 图表 | 对应组 | 训练 | 评测 |
|------|------|--------|------|------|
| §3-5 | Fig.11/13 | fig11_13_response_length | ✅ 6/6 | ⏳ |
| §3-5 | Fig.15/16 | fig15_16_topk | ✅ 5/5 | ⏳ |
| §3-5 | Fig.2/6 | fig2_teacher_pattern | ✅ 2/2 | ⏳ |
| §3-5 | Fig.4/6 | fig4_6_deepseek_teachers | ✅ 3/3 | ⏳ |
| §3-5 | Fig.4 | fig4_qwen_public | ✅ 1/1 | ⏳ |
| §3-5 | Fig.5 | fig5_reverse_distillation | ✅ 2/2 | ⏳ |
| §3-5 | Fig.7 | fig7_support | ✅ 3/3 | ⏳ |
| §3-5 | Fig.8 | fig8_cold_start | ✅ 2/2 | ⏳ |
| §3-5 | Fig.9 | fig9_prompt_template | ✅ 2/2 | ⏳ |
| §3-5 | Fig.10 | fig10_prompt_content | ✅ 2/2 | ⏳ |

---

## 8. 待完成工作

| # | 任务 | 依赖 |
|---|------|------|
| 1 | 精确评测完成 | GPU (运行中) |
| 2 | 趋势评测重运行 | GPU 释放后 |
| 3 | Fig.11(b) 教师延续探针 | 评测完成 |
| 4 | Fig.14 序列奖励 AUROC | 评测完成 |
| 5 | 图表渲染 + 审计 | 全部数据 |
| 6 | 最终总结 | 全部完成 |

---

## 9. 总结

全部 **28/28** 训练单元完成 (100%)，覆盖论文 §3-§5 全部核心消融。
训练总耗时 ~59 小时，零失败，最高内存 64.6 GiB/GPU。
精确评测进行中，趋势评测脚本已修复待重运行。
W&B Dashboard 已上线: https://wandb.ai/693623649-/Rethinking-OPD-2xA100
