# Acceptance Dynamics Across Cognitive Domains in Speculative Decoding

**Authors:** Saif Mahmoud

**Conference:** arXiv 2026

**Year:** 2026

**ArXiv:** [2604.14682](<https://arxiv.org/abs/2604.14682>)

**Topic:** Speculative Decoding

---

## Abstract (English)

Speculative decoding accelerates large language model (LLM) inference. It uses a small draft model to propose a tree of future tokens. A larger target model then verifies these tokens in a single batched forward pass. Despite the growing body of work on speculative methods, the degree to which the cognitive characteristics of a task affect acceptance probability remains largely unexplored. We present an empirical study of tree-based speculative decoding acceptance dynamics. Our study spans four well-established NLP benchmark domains: code generation, mathematical reasoning, logical reasoning, and open-ended chat. For this, we use TinyLlama-1.1B as the draft model against Llama-2-7B-Chat-GPTQ as the target. Over 99,768 speculative nodes collected from 200 prompts, we derive per-domain acceptance rates, expected accepted lengths, depth-acceptance profiles, and entropy-acceptance correlations. We find that task type is a stronger predictor of acceptance than tree depth. Furthermore, only the chat domain consistently yields an expected accepted length exceeding 1.0 token per step. We also show that the entropy-acceptance correlation is consistently negative but weak across all domains. Counterintuitively, chat produces the highest entropy yet the highest acceptance rate. We attribute this divergence to the lexical predictability of RLHF-aligned register. These findings have direct implications for domain-aware speculation budgets and draft-model selection strategies.

## Abstract (Chinese / 中文摘要)

投机解码加速大语言模型(LLM)推理。它使用一个小型草案模型来提议未来token树，然后更大的目标模型在单次批量前向传播中验证这些token。尽管关于投机方法的研究越来越多，但任务的认知特征对接受概率的影响程度在很大程度上仍未被探索。我们对树基投机解码的接受动态进行了实证研究，跨越四个成熟的NLP基准领域：代码生成、数学推理、逻辑推理和开放式聊天。我们使用TinyLlama-1.1B作为草案模型，Llama-2-7B-Chat-GPTQ作为目标模型。从200个提示收集的99,768个投机节点中，我们推导出每个领域的接受率、预期接受长度、深度-接受率剖面和熵-接受率相关性。我们发现任务类型比树深度是更强的接受率预测因子。此外，只有聊天领域持续产生超过1.0 token/步的预期接受长度。反直觉地，聊天产生最高的熵但最高的接受率。我们将这种分歧归因于RLHF对齐寄存器的词汇可预测性。

---

*Auto-collected from arXiv on 2026-04-17*
