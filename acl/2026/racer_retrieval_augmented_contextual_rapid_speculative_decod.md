# RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding

**Authors:** Zihong Zhang, Zuchao Li, Lefei Zhang, Ping Wang, Hai Zhao

**Conference:** ACL Findings 2026

**Year:** 2026

**ArXiv:** [2604.14885](<https://arxiv.org/abs/2604.14885>)

**GitHub:** [https://github.com/hkr04/RACER](https://github.com/hkr04/RACER)

**Topic:** Speculative Decoding

---

## Abstract (English)

Autoregressive decoding in Large Language Models (LLMs) generates one token per step, causing high inference latency. Speculative decoding (SD) mitigates this through a guess-and-verify strategy, but existing training-free variants face trade-offs: retrieval-based drafts break when no exact match exists, while logits-based drafts lack structural guidance. We propose RACER (Retrieval-Augmented Contextual Rapid Speculative Decoding), a lightweight and training-free method that integrates retrieved exact patterns with logit-driven future cues. This unification supplies both reliable anchors and flexible extrapolation, yielding richer speculative drafts. Experiments on Spec-Bench, HumanEval, and MGSM-ZH demonstrate that RACER consistently accelerates inference, achieving more than 2x speedup over autoregressive decoding, and outperforms prior training-free methods, offering a scalable, plug-and-play solution for efficient LLM decoding.

## Abstract (Chinese / 中文摘要)

大语言模型(LLM)中的自回归解码每步生成一个token，导致高推理延迟。投机解码(SD)通过猜测-验证策略缓解这一问题，但现有的免训练变体面临权衡：基于检索的草案在没有精确匹配时会失效，而基于logits的草案缺乏结构指导。我们提出RACER(检索增强上下文快速投机解码)，一种轻量级免训练方法，将检索到的精确模式与logit驱动的未来线索整合。这种统一既提供了可靠的锚点又提供了灵活的推断，产生更丰富的投机草案。在Spec-Bench、HumanEval和MGSM-ZH上的实验证明，RACER持续加速推理，实现超过2倍的加速比，优于之前的免训练方法，提供可扩展的即插即用解决方案。

---

*Auto-collected from arXiv on 2026-04-17*
