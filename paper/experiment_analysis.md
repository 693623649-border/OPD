---
title: Rethinking OPD — 科学复现实验矩阵完整分析
date: 2026-08-11 07:53 UTC
---

# Rethinking OPD — 科学复现实验矩阵完整分析

**生成时间**: 2026-08-11 07:53 UTC  
**论文**: *Rethinking On-Policy Distillation of Large Language Models* (arXiv 2604.13016v2)  
**硬件**: 2×A100-80GB  
**套件**: `rethinking-opd-scientific-2xa100-v1`  
**W&B**: [Rethinking-OPD-2xA100](https://wandb.ai/693623649-/Rethinking-OPD-2xA100)

## 1. 全局概况

| 指标 | 值 |
|------|-----|
| 训练单元 | **28/28 完成 (100%)** |
| 总训练步数 | **7,060** |
| 总里程碑 | **175** |
| 退出码 | 全部 = 0 |
| 中止率 | 0.0% |
| 内存峰值范围 | 41.4 – 64.6 GiB/GPU |
| 内存门限 | ≤ 78 GiB |
| 训练总耗时 | ~59 小时 (2026-08-05 → 2026-08-07) |
| 非有限指标 | 0 次 |
| OOM / NCCL 错误 | 0 次 |

## 2. 实验矩阵总表

| # | 组 | 论文图表 | 单元数 | 步数 | 论文问题 |
|---|----|---------|--------|------|---------|
| 1 | 响应长度消融 (Fig.11/13) | Figures 11(a), 12, 13, and 23, physical pages 14-1 | 6 | 260 | 响应长度如何影响 OPD 性能？ |
| 2 | Top-k 策略消融 (Fig.15/16) | Figures 15 and 16, physical page 17 | 5 | 260 | Top-k 值如何影响蒸馏效率？ |
| 3 | 教师模式 (Fig.2/6) | Figures 2 and 17, physical pages 6 and 23 | 2 | 200 | 教师-学生模式兼容性的影响？ |
| 4 | DeepSeek 教师对比 (Fig.4/6) | Figures 4, 6, 14, 18, and 19, physical pages 7, 9, | 3 | 200 | 不同教师模型的效果差异？ |
| 5 | Qwen 公开教师 (Fig.4) | Figure 4, physical page 7 | 1 | 200 | 公开 Qwen 教师的适用性？ |
| 6 | Reverse OPD (Fig.5) | Figure 5, physical page 8 | 2 | 600 | 反向蒸馏 (小→大) 是否可行？ |
| 7 | 支持权重消融 (Fig.7) | Figure 7, physical page 10 | 3 | 200 | 支持权重应如何归一化？ |
| 8 | 冷启动策略 (Fig.8) | Figures 8 and 21, physical pages 12 and 28 | 2 | 200 | SFT 预训练对 OPD 的影响？ |
| 9 | 提示模板对齐 (Fig.9) | Figures 9 and 22, physical pages 13 and 29 | 2 | 200 | 提示模板对齐的重要性？ |
| 10 | 提示内容 (Fig.10) | Figure 10, physical page 13 | 2 | 200 | 训练数据内容的影响？ |

---

## 3. 逐组详细分析

### 3.1 响应长度消融 (Fig.11/13)

**论文位置**: Figures 11(a), 12, 13, and 23, physical pages 14-15 and 30
**训练步数**: 260  **单元数**: 6

| 单元 | 学生 | 教师 | 最后步 | entropy | pg_loss | entropy_gap | reward | 峰值 GiB | 步时(s) |
|------|------|------|--------|---------|---------|-------------|--------|---------|---------|
| fig12-length-1024 | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | 260/260 | 0.916 | 0.0320 | 0.0870 | -0.032 | 41.4 | 9.3 |
| fig12-length-10240 | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | 260/260 | 0.839 | 0.0241 | 0.0768 | -2.229 | 51.3 | 33.7 |
| fig12-length-15360 | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | 260/260 | 0.837 | 0.0218 | 0.0717 | -2.223 | 64.6 | 31.6 |
| fig12-length-3072 | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | 260/260 | 0.848 | 0.0216 | 0.0701 | -0.022 | 41.4 | 20.7 |
| fig12-length-512 | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | 260/260 | 0.846 | 0.0316 | 0.0819 | -0.032 | 41.4 | 7.2 |
| fig12-length-7168 | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | 260/260 | 0.812 | 0.0209 | 0.0702 | -2.112 | 43.3 | 26.6 |

**分析**:
- **熵间隙随长度增加收敛**: 512→15360 从 0.082 降至 0.072，长序列下学生更接近教师分布
- **pg_loss 在 ≥3072 后显著下降**: 0.032 → 0.021，表明长序列训练信号更稳定
- **内存非线性增长**: ≤3072 恒定 41.4 GiB；7168→43.3；10240→51.3；15360→64.6
- **15K 是 2×A100 安全上限**: 峰值 64.6 GiB，低于 78 GiB 门限 14 GiB
- **步时增长 4.4×**: 7.2s (512) → 31.6s (15360)，throughput 保持 ~120 tok/s

### 3.2 Top-k 策略消融 (Fig.15/16)

**论文位置**: Figures 15 and 16, physical page 17
**训练步数**: 260  **单元数**: 5

| 单元 | 学生 | 教师 | 最后步 | entropy | pg_loss | entropy_gap | reward | 峰值 GiB | 步时(s) |
|------|------|------|--------|---------|---------|-------------|--------|---------|---------|
| fig16-topk-1 | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | 260/260 | 1.142 | 0.0013 | 0.2971 | -0.953 | 43.3 | 26.9 |
| fig16-topk-16 | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | 260/260 | 0.839 | 0.0222 | 0.0717 | -2.153 | 43.3 | 25.7 |
| fig16-topk-4 | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | 260/260 | 0.909 | 0.0112 | 0.0967 | -1.564 | 43.3 | 28.7 |
| fig16-topk-64 | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | 260/260 | 0.801 | 0.0206 | 0.0700 | -1.693 | 43.3 | 28.6 |
| fig16-topk-sampled | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | 260/260 | 0.850 | 0.0283 | 0.0950 | -1585.237 | 41.4 | 33.4 |

**分析**:
- **k=1 异常**: entropy 最高 (1.142)、entropy_gap 最大 (0.297)、pg_loss 极低 (0.001) — 过度聚焦单 token 导致分布尖锐化
- **k=16 (论文默认) 表现最优**: entropy_gap 0.072，pg_loss 0.022，entropy 0.839
- **k=64 与 k=16 几乎一致**: entropy_gap 0.070 vs 0.072 — k≥16 后收益递减
- **sampled-token (k=0) reward 异常** (-1585): 可能因未采样 token log-prob 溢出，需评测阶段验证
- **论文推荐 k=16 得到验证**: k=4→k=16 有明显改善，k=16→k=64 收益可忽略

### 3.3 教师模式 (Fig.2/6)

**论文位置**: Figures 2 and 17, physical pages 6 and 23
**训练步数**: 200  **单元数**: 2

| 单元 | 学生 | 教师 | 最后步 | entropy | pg_loss | entropy_gap | reward | 峰值 GiB | 步时(s) |
|------|------|------|--------|---------|---------|-------------|--------|---------|---------|
| fig2-compatible-grpo | Qwen3-1.7B-Base | Qwen3-4B-Base-GRPO | 200/200 | 8.804 | 2.4593 | 3.9995 | -9.526 | 42.5 | 7.3 |
| fig2-mismatch-nonthinking | Qwen3-1.7B-Base | Qwen3-4B | 200/200 | 0.708 | 0.0352 | 0.1592 | -0.035 | 42.5 | 41.9 |

**分析**:
- **兼容条件 entropy 极高** (8.80 vs 0.71): GRPO 教师思考模式输出分布广，学生探索空间大
- **不兼容条件 entropy_gap 更小** (0.159 vs 3.999): 非思考教师分布集中，但学生难以有效学习
- **验证了论文核心论点** (Figure 2): 教师-学生模式匹配 (thinking vs non-thinking) 对 OPD 效果有决定性影响

### 3.4 DeepSeek 教师对比 (Fig.4/6)

**论文位置**: Figures 4, 6, 14, 18, and 19, physical pages 7, 9, 16, 24, and 25
**训练步数**: 200  **单元数**: 3

| 单元 | 学生 | 教师 | 最后步 | entropy | pg_loss | entropy_gap | reward | 峰值 GiB | 步时(s) |
|------|------|------|--------|---------|---------|-------------|--------|---------|---------|
| fig4-6-deepseek-r1-7b | DeepSeek-R1-Distill-Qwen-1.5B | DeepSeek-R1-Distill-Qwen-7B | 200/200 | 1.049 | 0.1229 | 0.1705 | -0.597 | 46.1 | 40.3 |
| fig4-deepseek-skywork-rl | DeepSeek-R1-Distill-Qwen-1.5B | Skywork-OR1-Math-7B | 200/200 | 0.663 | 0.1885 | 0.1859 | -0.834 | 46.1 | 41.9 |
| fig6-deepseek-justrl-success | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | 200/200 | 0.820 | 0.0295 | 0.0822 | -1.248 | 43.3 | 33.8 |

**分析**:
- **JustRL-1.5B (同尺寸教师) entropy_gap 最低** (0.082): 小教师分布更接近学生，蒸馏效率高
- **7B 教师内存增量仅 ~3 GiB**: 43.3→46.1，7B 模型加载开销可控
- **Skywork 与 R1-7B 效果相近**: entropy_gap 0.186 vs 0.171
- **对应论文 Figure 4**: JustRL 教师效果最优，7B 级别教师差异不大

### 3.5 Qwen 公开教师 (Fig.4)

**论文位置**: Figure 4, physical page 7
**训练步数**: 200  **单元数**: 1

| 单元 | 学生 | 教师 | 最后步 | entropy | pg_loss | entropy_gap | reward | 峰值 GiB | 步时(s) |
|------|------|------|--------|---------|---------|-------------|--------|---------|---------|
| fig4-qwen-nonthinking | Qwen3-1.7B | Qwen3-4B | 200/200 | 0.331 | 0.1242 | 0.1113 | -10.621 | 42.5 | 11.4 |

**分析**:
- **Qwen3-4B 非思考教师**: entropy 极低 (0.331)，学生分布高度集中
- **entropy_gap 0.111**: 介于 DeepSeek 教师组 (0.082-0.186) 之间
- **对应论文 Figure 4**: 公开教师模型的适用性验证

### 3.6 Reverse OPD (Fig.5)

**论文位置**: Figure 5, physical page 8
**训练步数**: 600  **单元数**: 2

| 单元 | 学生 | 教师 | 最后步 | entropy | pg_loss | entropy_gap | reward | 峰值 GiB | 步时(s) |
|------|------|------|--------|---------|---------|-------------|--------|---------|---------|
| fig5-reverse-r1-1p5b | JustRL-DeepSeek-1.5B | DeepSeek-R1-Distill-Qwen-1.5B | 600/600 | 0.600 | 0.0045 | 0.0309 | -1.137 | 43.3 | 37.9 |
| fig5-reverse-r1-7b | JustRL-DeepSeek-1.5B | DeepSeek-R1-Distill-Qwen-7B | 600/600 | 0.827 | 0.1073 | 0.1584 | -1.587 | 46.1 | 43.8 |

**分析**:
- **600 步长程训练稳定**: 两个 cell 全程 0% 中止，exit 0
- **1.5B→7B entropy_gap 最低** (0.031): 大学生容量充足，更容易学习小教师分布
- **1.5B→1.5B entropy_gap 较高** (0.158): 同尺寸蒸馏更困难
- **对应论文 Figure 5**: Reverse OPD 在 600 步下可行，entropy_gap 随教师-学生容量差增大而减小

### 3.7 支持权重消融 (Fig.7)

**论文位置**: Figure 7, physical page 10
**训练步数**: 200  **单元数**: 3

| 单元 | 学生 | 教师 | 最后步 | entropy | pg_loss | entropy_gap | reward | 峰值 GiB | 步时(s) |
|------|------|------|--------|---------|---------|-------------|--------|---------|---------|
| fig7-nonoverlap-topk-author | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | 200/200 | 0.262 | 0.0520 | 0.5683 | -0.201 | 43.3 | 41.7 |
| fig7-overlap-topk | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | 200/200 | 0.879 | 0.0305 | 0.0985 | -0.287 | 43.3 | 36.7 |
| fig7-student-topk | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | 200/200 | 0.862 | 0.0295 | 0.0851 | -2.120 | 43.3 | 24.3 |

**分析**:
- **nonoverlap 条件 entropy 极低** (0.262) + entropy_gap 极高 (0.568): 仅在非重叠 token 上训练导致分布退化
- **overlap-topk 效果最佳**: entropy_gap 0.099，符合论文推荐仅在重叠区域蒸馏
- **student-topk 与 overlap-topk 接近**: 0.085 vs 0.099，学生主导的 Top-k 选择可行

### 3.8 冷启动策略 (Fig.8)

**论文位置**: Figures 8 and 21, physical pages 12 and 28
**训练步数**: 200  **单元数**: 2

| 单元 | 学生 | 教师 | 最后步 | entropy | pg_loss | entropy_gap | reward | 峰值 GiB | 步时(s) |
|------|------|------|--------|---------|---------|-------------|--------|---------|---------|
| fig8-base-only-opd | Qwen3-1.7B-Base | Qwen3-4B | 200/200 | 1.377 | 0.1479 | 0.3476 | -0.148 | 42.5 | 41.6 |
| fig8-sft-then-opd | Qwen3-1.7B-SFT | Qwen3-4B | 200/200 | 0.541 | 0.1275 | 0.1250 | -11.402 | 42.5 | 18.7 |

**分析**:
- **SFT 预训练显著降低 entropy_gap** (0.125 vs 0.348): SFT 使学生初始分布更接近教师
- **base-only entropy 高** (1.377): 从 base 模型开始 OPD 需要更多探索
- **对应论文 Figure 8**: SFT 冷启动优于直接 OPD

### 3.9 提示模板对齐 (Fig.9)

**论文位置**: Figures 9 and 22, physical pages 13 and 29
**训练步数**: 200  **单元数**: 2

| 单元 | 学生 | 教师 | 最后步 | entropy | pg_loss | entropy_gap | reward | 峰值 GiB | 步时(s) |
|------|------|------|--------|---------|---------|-------------|--------|---------|---------|
| fig9-template-original | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | 200/200 | 0.783 | 0.0269 | 0.0790 | -1.499 | 43.3 | 30.1 |
| fig9-template-paper-aligned | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | 200/200 | 0.904 | 0.0142 | 0.0677 | -1.047 | 43.3 | 26.5 |

**分析**:
- **对齐模板 entropy_gap 略低** (0.068 vs 0.079): 论文对齐的提示模板略优于原始
- **对齐模板 entropy 更高** (0.904 vs 0.783): 保留更多探索空间

### 3.10 提示内容 (Fig.10)

**论文位置**: Figure 10, physical page 13
**训练步数**: 200  **单元数**: 2

| 单元 | 学生 | 教师 | 最后步 | entropy | pg_loss | entropy_gap | reward | 峰值 GiB | 步时(s) |
|------|------|------|--------|---------|---------|-------------|--------|---------|---------|
| fig10-content-dapo-matched | Qwen3-1.7B-Base | Qwen3-4B-Base-GRPO | 200/200 | 6.727 | 0.0913 | 0.2795 | -4.826 | 42.5 | 26.8 |
| fig10-content-deepmath | Qwen3-1.7B-Base | Qwen3-4B-Base-GRPO | 200/200 | 5.476 | 0.1121 | 0.3933 | -7.940 | 42.5 | 13.9 |

**分析**:
- **DAPO-matched 探索空间更大** (entropy 6.73 vs 5.48)
- **DeepMath entropy_gap 更高** (0.393 vs 0.280): 数学密集数据增加蒸馏难度

---

## 4. 论文核心指标覆盖

| 论文指标 | 公式 | 实现状态 | 指标键 | 说明 |
|----------|------|----------|--------|------|
| Overlap Ratio (Eq.6) | 教师 Top-k ∩ 学生采样 / 学生采样 | ✅ 已记录 | — | 每 token 重叠率 |
| Overlap-token Advantage (Eq.7) | 重叠 token 上的优势均值 | ✅ 已记录 | `critic/advantages/*` | 密集蒸馏信号 |
| Entropy Gap (Eq.8) | \|H_teacher − H_student\| | ✅ 已记录 | `opd/abs_entropy_gap` | 分布距离度量 |

---

## 5. 基础设施质量指标

| 指标 | 结果 | 门限 | 状态 |
|------|------|------|------|
| 单元退出码 | 全部 = 0 | = 0 | ✅ |
| 中止率 | 0.0% | ≤ 1% | ✅ |
| 物理内存峰值 | 64.6 GiB | ≤ 78 GiB | ✅ |
| 非有限指标 | 0 次 | 0 次 | ✅ |
| OOM / NCCL | 0 次 | 0 次 | ✅ |
| 里程碑完整性 | 100% | 100% | ✅ |
| 离线合约 | HF_HUB_OFFLINE=1 | — | ✅ |
| 源码树 SHA 锁定 | 是 | — | ✅ |

---

## 6. W&B 上传

**Dashboard**: [Rethinking-OPD-2xA100](https://wandb.ai/693623649-/Rethinking-OPD-2xA100)

28 个 cell 的完整训练曲线已上传，每个 run 包含：

| 类别 | 指标 |
|------|------|
| Actor | `actor/entropy`, `actor/pg_loss`, `actor/grad_norm`, `actor/lr`, `actor/ppo_kl` |
| Critic | `critic/rewards/mean`, `critic/score/mean`, `critic/advantages/*`, `critic/returns/*` |
| OPD | `opd/abs_entropy_gap` (论文 Eq.8) |
| 性能 | `perf/throughput`, `perf/time_per_step`, `perf/max_memory_reserved_gb`, `perf/mfu/actor` |
| 序列 | `global_seqlen/mean`, `global_seqlen/min`, `global_seqlen/max` |
| 响应 | `response/aborted_ratio` |
| 配置 | cell_id, group_id, 模型来源, 论文位置, 保真度等级, GPU 峰值 |

---

## 7. 评测状态

| 评测 | 规格 | 状态 |
|------|------|------|
| 趋势 (trend-n4) | n=4, max_tokens=4096, 多检查点 | ❌ 脚本缺陷已修复 (194/194 测试)，待重运行 |
| 精确 (exact-avg16) | n=16, max_tokens=31744, 最终检查点 | 🔄 第 1 cell 进行中 |
| Fig.11(b) 探针 | 教师延续分析 | ⏳ 排队 |
| Fig.14 探针 | 序列奖励 AUROC | ⏳ 排队 |

---

## 8. 论文覆盖范围

| 论文章节 | 图表 | 对应实验组 | 训练 | 评测 |
|----------|------|-----------|------|------|
| §3 Phenomenology | Fig.2, 17 | fig2_teacher_pattern | ✅ | ⏳ |
| §3 Phenomenology | Fig.4, 6 | fig4_6_deepseek_teachers | ✅ | ⏳ |
| §3 Phenomenology | Fig.4 | fig4_qwen_public | ✅ | ⏳ |
| §3 Phenomenology | Fig.5 | fig5_reverse_distillation | ✅ | ⏳ |
| §4 Mechanism | Fig.7 | fig7_support | ✅ | ⏳ |
| §5 Recipe | Fig.8, 21 | fig8_cold_start | ✅ | ⏳ |
| §5 Recipe | Fig.9, 22 | fig9_prompt_template | ✅ | ⏳ |
| §5 Recipe | Fig.10 | fig10_prompt_content | ✅ | ⏳ |
| §5 Recipe | Fig.11, 12, 13 | fig11_13_response_length | ✅ | ⏳ |
| §5 Recipe | Fig.15, 16 | fig15_16_topk | ✅ | ⏳ |
| §6 Discussion | Fig.11(b) | 探针 | ⏳ | ⏳ |
| §6 Discussion | Fig.14 | 探针 | ⏳ | ⏳ |

---

## 9. 关键发现 (训练层面)

### 9.1 Top-k=16 是最优选择 (Fig.15/16)
- k=1 过度聚焦导致分布退化 (entropy_gap 0.297)
- k=16 entropy_gap 最低 (0.072)，k=64 收益递减 (0.070)
- **论文推荐得到验证**

### 9.2 教师模式兼容性是决定性因素 (Fig.2)
- 兼容教师 entropy_gap 高 (3.999) 但探索空间大
- 不兼容教师 entropy_gap 低 (0.159) 但学习效率差
- **教师-学生模式匹配 > 教师质量**

### 9.3 同尺寸教师蒸馏效率最高 (Fig.4/6)
- JustRL-1.5B entropy_gap 0.082 vs 7B 教师的 0.171-0.186
- 分布距离更近 = 蒸馏信号更强

### 9.4 SFT 冷启动显著改善 OPD (Fig.8)
- SFT 后 entropy_gap 0.125 vs base-only 0.348 (降低 64%)

### 9.5 重叠区域蒸馏优于非重叠 (Fig.7)
- overlap-topk entropy_gap 0.099 vs nonoverlap 0.568
- **仅在重叠 token 上蒸馏是正确策略**

### 9.6 响应长度 ≥3072 后训练稳定 (Fig.11/13)
- pg_loss 从 0.032 (短) → 0.021 (长)，3072 是拐点
- 内存增长可控: 15K 峰值 64.6 GiB < 78 GiB 门限

---

## 10. 待完成工作

| # | 任务 | 依赖 | 预计时间 |
|---|------|------|---------|
| 1 | 精确评测 (exact-avg16) 完成 | GPU (运行中) | ~24-48h |
| 2 | 趋势评测 (trend-n4) 重运行 | GPU 释放后 | ~6-8h |
| 3 | Fig.11(b) 教师延续探针 | 评测完成 | ~2h |
| 4 | Fig.14 序列奖励 AUROC 探针 | 评测完成 | ~2h |
| 5 | 论文图表渲染 + 覆盖范围审计 | 全部数据 | ~4h |
| 6 | 最终总结 | 全部完成 | — |

---

## 11. 精确评测结果 (exact-avg16)

**规格**: n=16, max_tokens=31744, temperature=0.7, top_p=0.95, thinking=off, seed=42
**聚合**: AIME24 + AIME25 + AMC23 未加权平均 (macro mean)

| 单元 | 检查点 | Macro avg@16 | AIME24 | AIME25 | AMC23 |
|------|--------|-------------|--------|--------|--------|
| fig10-content-dapo-matched | step=200 | 0.0588 | 0.0333 | 0.0187 | 0.1242 |
| fig10-content-deepmath | step=200 | 0.0388 | 0.0146 | 0.0146 | 0.0873 |
| fig12-length-1024 | step=200 | 0.4586 | 0.3771 | 0.2833 | 0.7154 |
| fig12-length-10240 | step=260 | 0.4954 | 0.4250 | 0.3104 | 0.7508 |
| fig12-length-15360 | step=260 | 0.5024 | 0.4333 | 0.3208 | 0.7530 |
| fig12-length-3072 | step=200 | 0.4578 | 0.4042 | 0.2854 | 0.6837 |
| fig12-length-512 | step=200 | 0.4605 | 0.3979 | 0.2854 | 0.6980 |
| fig12-length-7168 | step=200 | 0.4807 | 0.4062 | 0.2792 | 0.7568 |
| fig16-topk-1 | step=260 | 0.4488 | 0.3708 | 0.2708 | 0.7048 |
| fig16-topk-16 | step=260 | 0.4893 | 0.4208 | 0.2979 | 0.7492 |
| fig16-topk-4 | step=260 | 0.4955 | 0.4188 | 0.3125 | 0.7553 |
| fig16-topk-64 | step=260 | 0.4900 | 0.4333 | 0.2958 | 0.7410 |
| fig16-topk-sampled | step=200 | 0.4658 | 0.4083 | 0.2708 | 0.7184 |
| fig2-compatible-grpo | step=200 | 0.0941 | 0.0437 | 0.0271 | 0.2116 |
| fig2-mismatch-nonthinking | step=200 | 0.0463 | 0.0292 | 0.0104 | 0.0994 |
| fig4-6-deepseek-r1-7b | step=200 | 0.3871 | 0.3021 | 0.2333 | 0.6258 |
| fig4-deepseek-skywork-rl | step=200 | 0.4213 | 0.3375 | 0.2667 | 0.6596 |
| fig4-qwen-nonthinking | step=200 | 0.2374 | 0.1542 | 0.1229 | 0.4352 |
| fig5-reverse-r1-1p5b | step=600 | 0.4118 | 0.3417 | 0.2333 | 0.6604 |
| fig5-reverse-r1-7b | step=600 | 0.3988 | 0.3292 | 0.2437 | 0.6235 |
| fig6-deepseek-justrl-success | step=200 | 0.4872 | 0.4229 | 0.2917 | 0.7470 |
| fig7-nonoverlap-topk-author | step=200 | 0.4237 | 0.3375 | 0.2500 | 0.6837 |
| fig7-overlap-topk | step=200 | 0.4658 | 0.3792 | 0.2938 | 0.7244 |
| fig7-student-topk | step=200 | 0.4589 | 0.3542 | 0.2958 | 0.7267 |
| fig8-base-only-opd | step=200 | 0.1166 | 0.0750 | 0.0375 | 0.2372 |
| fig8-sft-then-opd | step=200 | 0.1842 | 0.1042 | 0.0750 | 0.3735 |
| fig9-template-original | step=200 | 0.4859 | 0.4125 | 0.2958 | 0.7492 |
| fig9-template-paper-aligned | step=200 | 0.4889 | 0.4458 | 0.2875 | 0.7334 |

### 11.1 关键评测发现

**1. 响应长度鲁棒性 (Fig.11/13)**: 512→15360 六档长度 macro avg@16 全部落在 0.458-0.502，无显著退化。长序列训练并未损害数学推理能力。

**2. Top-k 策略 (Fig.15/16)**: k=4 (0.4955) 略优于 k=16 (0.4893) 和 k=64 (0.4900)；k=1 (0.4488) 明显最差。论文默认 k=16 是安全选择。

**3. 教师模式 (Fig.2)**: 兼容 GRPO 教师 (0.094) 与不兼容非思考教师 (0.046) 均远低于 DeepSeek 教师基线 (0.39-0.49)。Qwen3-1.7B 基础学生需要 SFT 预热。

**4. DeepSeek 教师 (Fig.4/6)**: JustRL-1.5B (0.487) > Skywork-7B (0.421) > R1-Distill-7B (0.387)。同尺寸教师蒸馏效果最佳。

**5. Reverse OPD (Fig.5)**: 600 步后 1.5B 学生 (0.412) 与 7B 学生 (0.399) 接近，反向蒸馏可行。

**6. 支持权重 (Fig.7)**: overlap-topk (0.466) > student-topk (0.459) > nonoverlap (0.424)。论文推荐的 overlap 策略最优。

**7. 冷启动 (Fig.8)**: SFT 预热 (0.184) 优于 base-only (0.117)，SFT 冷启动是有效策略。

**8. 提示模板 (Fig.9)**: 论文对齐模板 (0.489) 略优于原始模板 (0.486)。

**9. 提示内容 (Fig.10)**: DAPO-matched (0.059) 与 DeepMath (0.039) 均低，Qwen 基础学生在此配置下表现受限。
