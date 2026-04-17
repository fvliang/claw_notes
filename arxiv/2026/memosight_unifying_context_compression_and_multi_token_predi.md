# MemoSight: Unifying Context Compression and Multi Token Prediction for Reasoning Acceleration

**Authors:** Xinyu Liu, Xin Liu, Bo Jin, Runsong Zhao, Pengcheng Huang, Junhao Ruan, Bei Li, Chunyang Xiao, Tong Xiao, Jingbo Zhu

**Conference:** arXiv 2026

**Year:** 2026

**ArXiv:** [2604.14889](<https://arxiv.org/abs/2604.14889>)

**Topic:** KV Cache

---

## Abstract (English)

While Chain-of-thought (CoT) reasoning enables LLMs to solve challenging reasoning problems, as KV cache grows linearly with the number of generated tokens, CoT reasoning faces scaling issues in terms of speed and memory usage. In this work, we propose MemoSight (Memory-Foresight-based reasoning), a unified framework that integrates both context compression and multi-token prediction to mitigate the efficiency issues while maintaining CoT reasoning performance. Our framework adopts the same minimalist design for both context compression and multi-token prediction via special tokens and their corresponding position layout tailored to each token type. Comprehensive experiments on four reasoning benchmarks demonstrate that MemoSight reduces the KV cache footprint by up to 66% and accelerates inference by 1.56x, while outperforming existing CoT compression methods.

## Abstract (Chinese / 中文摘要)

虽然思维链(CoT)推理使LLM能够解决具有挑战性的推理问题，但随着KV缓存随生成的token数量线性增长，CoT推理在速度和内存使用方面面临扩展性问题。在这项工作中，我们提出MemoSight(基于记忆-预见推理)，一个统一框架，集成了上下文压缩和多token预测以缓解效率问题同时保持CoT推理性能。我们的框架对上下文压缩和多token预测采用了相同的最小化设计，通过特殊token及其相应的位置布局为每种token类型量身定制。在四个推理基准上的综合实验表明，MemoSight将KV缓存占用减少高达66%并将推理加速1.56倍。

---

*Auto-collected from arXiv on 2026-04-17*
