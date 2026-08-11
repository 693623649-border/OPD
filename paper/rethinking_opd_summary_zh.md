# 《Rethinking On-Policy Distillation》精读与复现依据

## 来源与阅读说明

- 原文：[本地 PDF](./rethinking_opd.pdf)，共 30 个物理页；PDF 物理页与论文页脚页码一致。
- 论文：Yaxuan Li 等，*Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe*，arXiv:2604.13016v2。
- 本文区分三类信息：“论文报告”表示作者实验结论；“由报告值计算”表示二次计算；“4B 迁移方案”表示面向本机的工程外推，不冒充论文原实验。

## 核心结论

论文研究的不是“OPD 平均能提高多少”，而是它何时会成功、逐 token 为什么有效、失败后怎样恢复。作者报告两个经验上共同支配 OPD 成败的因素；现有实验没有把它们证明为形式化的必要条件：

1. 学生与教师需要具有兼容的 thinking pattern。论文主要用双方 Top-\(k\) 高概率 token 的重合度来操作化这一概念。
2. 教师必须带来学生尚未获得的新能力。更大或 benchmark 分数更高，不等于在学生访问的状态上存在可迁移的新信号；额外 RL 后训练往往比单纯扩大参数规模更重要。

当 OPD 成功时，学生在自己生成的 prefix 上逐步进入教师的高概率区域：Top-\(k\) overlap 上升、overlap-token advantage 趋近 0、师生 entropy gap 缩小。共享 token 数量很少，却承载双方约 97%–99% 的概率质量。只优化 overlap token 几乎可达到完整 Student Top-\(k\) OPD 的效果。

![原文图 2：不同 thinking pattern 的教师](./images/figure-02-thinking-pattern.png)

*原文图 2（物理页 6）：benchmark 更强并不能保证蒸馏更强；与 Base 学生更兼容的 GRPO 教师从更高的初始 overlap 出发，并取得更好的学生结果。*

## OPD 目标与实现含义

令 prompt 为 \(x\)，学生生成轨迹 \(\hat y\sim\pi_\theta(\cdot\mid x)\)。在第 \(t\) 个学生访问状态上：

\[
p_t(v)=\pi_\theta(v\mid x,\hat y_{<t}),\qquad
q_t(v)=\pi_T(v\mid x,\hat y_{<t}).
\]

序列级 reverse KL 可精确分解为逐 token KL（物理页 3，式 1–2）：

\[
\mathcal L_{\mathrm{OPD}}(\theta)
=
\mathbb E_{x,\hat y\sim\pi_\theta}
\left[
\sum_{t=1}^{T}D_{\mathrm{KL}}(p_t\Vert q_t)
\right].
\]

这里的关键是 rollout 来自当前学生，教师只在学生实际到达的 prefix 上提供分布，因此避免了只在固定教师轨迹上训练所产生的 exposure mismatch。

论文讨论三种监督粒度：

| 方式 | 每个位置使用的 token | 性质与代价 |
|---|---:|---|
| Sampled-token | 从 \(p_t\) 抽取 1 个 | \(\log p_t(\hat y_t)-\log q_t(\hat y_t)\) 是 KL 数值的无偏单样本估计，成本最低 |
| Full-vocabulary | 全词表 | 精确逐 token KL，显存约 \(O(BTM)\) |
| Student Top-\(k\) | 学生最高概率的 \(k\) 个 | 在子集上重新归一化，介于两者之间；论文默认 \(k=16\) |

Student Top-\(k\) 令 \(S_t=\operatorname{TopK}(p_t,k)\)，并在该集合上定义：

\[
\bar p_t(v)=\frac{p_t(v)\mathbf1[v\in S_t]}{\sum_{u\in S_t}p_t(u)},\qquad
\bar q_t(v)=\frac{q_t(v)\mathbf1[v\in S_t]}{\sum_{u\in S_t}q_t(u)}.
\]

训练最小化 \(D_{\mathrm{KL}}(\bar p_t\Vert\bar q_t)\)。它丢弃集合外的概率质量，所以是 full-vocabulary reverse KL 的近似，而不是同一个数值的无损稀疏实现。

## 成功与失败的判据

论文定义三个在线指标（物理页 4–5，式 6–8）：

\[
M_{\mathrm{overlap}}
=\mathbb E_t\left[
\frac{|S_t^{(p)}\cap S_t^{(q)}|}{k}
\right],
\]

\[
A_t(v)=\bar p_t(v)\bigl(\log\bar q_t(v)-\log\bar p_t(v)\bigr),
\]

\[
M_{\mathrm{adv}}
=\mathbb E_t\left[
\frac{1}{|I_t|}\sum_{v\in I_t}A_t(v)
\right],\quad I_t=S_t^{(p)}\cap S_t^{(q)},
\]

\[
\Delta H_t=|H(q_t)-H(p_t)|.
\]

\(M_{\mathrm{adv}}\) 趋近 0 表示学生在共享 token 内的概率校准逐渐接近教师；明显负值表示学生相对教师过于自信。附录还建议同时记录双方的 overlap mass，即 \(\sum_{v\in I_t}p_t(v)\) 与 \(\sum_{v\in I_t}q_t(v)\)，防止“交集内看似对齐、但交集漏掉主要概率质量”的误判。

![原文图 6：成功与失败 OPD 的动态差异](./images/figure-06-success-failure.png)

*原文图 6（物理页 9）：同一 1.5B 学生面对两个总体性能相近的教师，成功运行的 overlap 从约 72% 持续升高，advantage 趋零且 entropy gap 收窄；失败运行从起点便停滞。*

![原文图 7：优化支持集消融](./images/figure-07-overlap-ablation.png)

*原文图 7（物理页 10）：Overlap Top-16 几乎匹配完整 Student Top-16，Non-Overlap Top-16 明显更弱；前两者把 overlap ratio 从约 72% 推高到 91% 以上。*

这个消融支持如下机制：reverse KL 在教师支持的共享高概率 token 上增加质量，竞争性的 non-overlap token 随之被挤出学生 Top-\(k\)，共享区域进一步扩大，形成自增强循环。附录图 18 报告这些 overlap token 在双方分布中持续承载约 97%–99% 的总质量。

## “更强教师”为什么仍会失败

### Thinking-pattern compatibility

论文把 Qwen3-1.7B-Base 固定为学生，比较 Qwen3-4B Non-thinking 与 Qwen3-4B-Base-GRPO。教师的精确 avg@16 为：

| 教师 | AIME 2024 | AIME 2025 | AMC 2023 | 三项均值（由报告值计算） |
|---|---:|---:|---:|---:|
| Qwen3-4B Non-thinking | 0.212 | 0.210 | 0.700 | 0.3740 |
| Qwen3-4B-Base-GRPO | 0.204 | 0.242 | 0.599 | 0.3483 |

尽管 Non-thinking 教师的三项均值更高，Base-GRPO 教师带来的学生增益更大。其初始 overlap 约 0.69，而 Non-thinking 教师约 0.58–0.60。后期两条 overlap 曲线接近，但早期损失没有在准确率上恢复。

### New knowledge, not just scale

论文用 gap recovery rate 衡量学生追回多少师生差距：

\[
\frac{\mathrm{Acc}_{\text{after OPD}}-\mathrm{Acc}_{\text{before OPD}}}
{\mathrm{Acc}_{\text{teacher}}-\mathrm{Acc}_{\text{before OPD}}}.
\]

| 家族 | 教师 | 初始 overlap | Gap recovery |
|---|---|---:|---:|
| DeepSeek | Skywork-OR1-Math-7B（额外 RL） | 71.5% | 16.9% |
| DeepSeek | R1-Distill-7B | 74.7% | 5.3% |
| Qwen | Qwen3-4B-RL-Math（额外 RL） | 70.3% | 58.6% |
| Qwen | Qwen3-4B Non-thinking | 75.7% | 15.6% |

两个额外 RL 的教师初始 overlap 反而略低，却迁移了更多能力。这直接说明 overlap 是重要诊断，但不是充分条件；教师是否获得学生没有的新能力同样关键。

反向蒸馏进一步表明，R1-Distill-1.5B 和同家族 7B 教师都能把经过 RL 的 JustRL-1.5B 拉回接近 RL 前水平。论文据此推断 OPD 会覆盖学生已有的 thinking pattern，且大模型的更高 benchmark 分数不必然对应不同的局部目标分布。不过，论文没有直接报告两个教师之间的 KL，因此“分布不可区分”仍是由相似训练轨迹推断出的结论。

## 失败后的两种恢复 recipe

### 1. Off-policy cold start

先让教师在固定 prompts 上生成回答，对学生做 SFT，再启动 OPD。论文使用 Qwen3-4B Non-thinking 生成 200K 条 OpenThoughts3 数学回答，将 Qwen3-1.7B-Base 全参数 SFT 为 Qwen3-1.7B-SFT，然后在与 SFT prompts 去重的约 30K prompts 上做 OPD。

![原文图 8：SFT cold start](./images/figure-08-cold-start.png)

*原文图 8（物理页 12）：SFT 初始化从一开始便有更高、更平滑的 overlap 和更小的 entropy gap，最终性能上限也高于直接 OPD。*

离线生成使用 temperature 0.7、top-p 0.95、top-k=-1、最大 12,288 tokens。SFT 为 BF16 全参数训练，1 epoch、序列长 14,336、每卡 batch 8、LR \(10^{-5}\)、cosine scheduler、warmup ratio 0.05。

### 2. Teacher-aligned prompts

只替换 prompt 模板、保持题目不变，也会改变学生访问的状态。论文采用的 teacher-aligned 模板为：

```text
{Question} Please reason step by step, and put your final answer within \boxed{}.
```

![原文图 9：模板对齐](./images/figure-09-template-alignment.png)

*原文图 9（物理页 13）：teacher-aligned 模板带来更高准确率与更高 overlap，说明看似细小的呈现方式也是 OPD 状态分布的一部分。*

在 prompt 内容实验中，与教师 RL 数据一致的 DAPO prompts 比去重后的 DeepMath 更强。值得注意的是，其集合 overlap 比例更低，但 overlap mass 更高，同时学生 entropy 显著下降。

![原文图 10：prompt 内容对齐](./images/figure-10-prompt-content.png)

*原文图 10（物理页 13）：教师数据内 prompts 让概率更集中于共享 token，却也几乎压低了学生熵；论文因此建议混入 OOD prompts，但没有报告混合比例消融。*

## 长序列的可靠性上限

论文在最大回复长度 0.5K、1K、3K、7K、10K、15K 上比较 OPD。0.5K/1K 的监督 token 太少，3K/7K 最强，10K/15K 后期出现 overlap 急跌、entropy 与 gradient norm 尖峰。15K 实验中异常先从回复尾部出现，再向前传播；教师 entropy 也有相同现象。

![原文图 11：长度与教师续写能力](./images/figure-11-length-effect.png)

*原文图 11（物理页 14）：教师从学生 prefix 续写的准确率增益随 prefix 深度单调降低，1K/4K/8K/16K 分别为 +0.3659、+0.2709、+0.1522、+0.0237。*

![原文图 12–13：长轨迹后期崩溃](./images/figure-12-13-long-horizon.png)

*原文图 12–13（物理页 15）：10K/15K 运行在训练后期出现 overlap 崩溃，15K 的高熵区域从 suffix 向 prefix 扩散。*

这表明 dense token reward 并非随轨迹长度免费扩展：学生 prefix 离开教师熟悉分布后，教师条件概率会变得更噪。论文还发现失败教师的 sequence-mean reward 仍能区分正确与错误 rollout：成功教师 AUROC 0.7333，失败教师 0.7511。因而失败不一定是全局信号无信息，更可能是局部梯度不可利用。作者提出“逐位置 advantage 方向相互抵消”的 anisotropy 假说，但没有直接验证。

## Top-\(k\) 取值

![原文图 15–16：支持集大小](./images/figure-15-16-topk.png)

*原文图 15–16（物理页 17）：Top-1 最不稳定；Top-4、16、64 与 sampled-token 的最终性能接近，但 Top-16/64 的动态最平滑。*

该消融的学生/教师是 R1-Distill-1.5B→JustRL-1.5B，而不是后文复现入口的 Qwen pair；下表只能作为论文原消融结果，不能当作 Qwen Fig. 2 的目标分数。

| 方法 | AIME 2024 | AIME 2025 | AMC 2023 |
|---|---:|---:|---:|
| Sampled-token | 0.454 | 0.327 | 0.782 |
| Top-1 | 0.446 | 0.310 | 0.772 |
| Top-4 | 0.473 | 0.331 | 0.793 |
| Top-16 | 0.458 | 0.338 | 0.791 |
| Top-64 | 0.463 | 0.338 | 0.785 |

Top-1 的问题不是“只有一个 token”本身，而是始终选择 argmax，形成有偏且 mode-concentrated 的信号；sampled-token 会按学生分布抽样，在训练过程中无偏覆盖高概率区域。考虑到稳定性和论文默认设置，本项目的主复现仍采用 Student Top-16。

## 论文默认超参数

| 项目 | OPD 默认值（Table 2，物理页 24） |
|---|---:|
| Training temperature | 1.0 |
| Global / mini batch size | 64 / 64 |
| Rollout number | 4 |
| LogProb Top-\(k\) | 16 |
| Top-\(k\) strategy | Student Top-\(k\) |
| Top-p | 1.0 |
| Max prompt / response | 1,024 / 7,168 |
| Learning rate | \(10^{-6}\) |
| Epoch | 1 |
| KL coefficient | 0.0 |

评测使用 AIME 2024、AIME 2025、AMC 2023；每题 16 个样本，temperature 0.7、top-p 0.95、最大回复 31,744，报告 avg@16 而不是 pass@16。

## 面向本机的 4B 复现边界

本机为 2×NVIDIA A100 80GB。项目提供两条可运行路径：

| 路径 | 学生 | 冻结教师 | 证据级别 |
|---|---|---|---|
| `MODEL_PAIR=paper` | Qwen3-1.7B-Base | Qwen3-4B-Base-GRPO | 最接近论文 Fig. 2 的成功 pair |
| `MODEL_PAIR=4b` | Qwen3-4B-Base | Qwen3-4B-Base-GRPO | 同家族、同 tokenizer、教师由额外 RL 获得新能力；属于 4B 迁移验证 |

4B 迁移保留 Student Top-16、temperature 1.0、LR \(10^{-6}\)、KL=0 和所有 overlap/entropy/gradient 诊断；`paper`/`pilot`/`smoke` 的 rollout 数分别为 4/2/1。启动器默认使用 Fig. 2 的 original DAPO，另可显式切换为 §5.2 的 teacher-aligned processed DAPO。为适配 2 卡，各 preset 缩小单步 prompt batch，pilot/smoke 还缩短上下文；这改变了同步批量与更新时序，因此最终数值不能直接宣称复现论文曲线。

## 局限与复现缺口

- 所有核心实验仅覆盖数学推理，没有多 seed、误差条或显著性检验。
- 论文未完整披露 OPD optimizer betas、weight decay、gradient clipping、precision、随机种子、分布式策略、checkpoint 选择规则、训练耗时与峰值显存。
- “Thinking pattern”主要由 Top-\(k\) overlap 操作化，不等同于独立测得的语义推理结构。
- “New knowledge”由额外 RL 与结果差异间接支持；RL 同时改变数据暴露、奖励偏好和输出分布。
- Sampled-token 只被证明是 KL 数值的无偏估计；论文没有完整写出 stop-gradient / policy-gradient 实现细节。
- Cold-start 方案额外使用 200K 数据和大量计算，缺少 SFT-only、compute-matched 与非教师 SFT 控制。
- Prompt 内容实验即使去重，DAPO 与 DeepMath 仍可能有难度和分布差异；“混入 OOD 防止熵塌缩”尚未实证。
- 长度实验未按总 token 或计算量对齐，且正文称训练 200 steps、图中却延伸至约 250–260 steps。
- 不同 tokenizer 下逐 token KL 如何定义没有方案；复现必须使用共享 tokenizer/词表的师生模型。

因此，本项目把“训练能跑通”“机制趋势出现”“论文最终分数复现”作为三个不同层级验收，避免仅凭一次曲线相似就作过强结论。
