# From Tokens to Steps: Verification-Aware Speculative Decoding for Efficient Multi-Step Reasoning (SpecGuard)

**Authors:** Authors from arXiv:2604.15244

**Conference:** arXiv 2026

**Year:** 2026

**ArXiv:** [2604.15244](<https://arxiv.org/abs/2604.15244>)

**Topic:** Speculative Decoding

---

## Abstract (English)

Speculative decoding (SD) accelerates large language model inference by allowing a lightweight draft model to propose outputs that a stronger target model verifies. However, its token-centric nature allows erroneous steps to propagate. Prior approaches mitigate this using external reward models, but incur additional latency, computational overhead, and limit generalizability. We propose SpecGuard, a verification-aware speculative decoding framework that performs step-level verification using only model-internal signals. At each step, SpecGuard samples multiple draft candidates and selects the most consistent step, which is then validated using an ensemble of two lightweight model-internal signals: (i) an attention-based grounding score that measures attribution to the input and previously accepted steps, and (ii) a log-probability-based score that captures token-level confidence. These signals jointly determine whether a step is accepted or recomputed using the target, allocating compute selectively. Experiments across a range of reasoning benchmarks show that SpecGuard improves accuracy by 3.6% while reducing latency by ~11%, outperforming both SD and reward-guided SD.

## Abstract (Chinese / 中文摘要)

投机解码(SD)通过允许轻量级草案模型提议由更强的目标模型验证的输出，加速大语言模型推理。然而，其以token为中心的性质允许错误步骤传播。先前的方法使用外部奖励模型来缓解此问题，但会产生额外的延迟、计算开销并限制泛化性。我们提出SpecGuard，一个验证感知的投机解码框架，仅使用模型内部信号执行步骤级验证。在每个步骤中，SpecGuard采样多个草案候选并选择最一致的步骤，然后使用两个轻量级模型内部信号的集合进行验证：(i)基于注意力的基础分数，衡量对输入和先前接受步骤的归因；(ii)基于log概率的分数，捕获token级置信度。实验表明SpecGuard将准确性提高3.6%同时减少约11%的延迟。

---

*Auto-collected from arXiv on 2026-04-17*
