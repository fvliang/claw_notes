# SMART: When is it Actually Worth Expanding a Speculative Tree?

**Source:** arxiv | **Category:** Speculative Decoding | **Date:** 2026-04-09
**ArXiv ID:** 2604.09731
**Authors:** Lifu Wang, Pan Zhou
**Tags:** speculative-decoding, tree-construction, marginal-analysis, hardware-aware, smart

## Links

- 📄 [Paper (PDF)](https://arxiv.org/pdf/2604.09731)
- 🌐 [ArXiv Page](https://arxiv.org/abs/2604.09731)

## Abstract (English)

Tree-based speculative decoding accelerates autoregressive generation by verifying a branching tree of draft tokens in a single target-model forward pass. However, existing methods prioritize maximizing token-level likelihood or accepted tokens while ignoring a critical efficiency paradox: computational overhead of drafting and verifying big trees can grow super-linearly, leading to negative wall-clock speedup when batch sizes increase or hardware saturation limits are reached. SMART is a system-aware marginal analysis framework for runtime tree construction. It reformulates tree expansion as a hardware-aware optimization problem directly maximizing end-to-end speedup. By applying a principled marginal benefit-cost rule at inference time, SMART expands a node only when its marginal benefit-cost ratio exceeds the tree-level speedup. SMART is training-free and plug-and-play for existing frameworks like MSD and EAGLE. It delivers average additional speedup of 20.0% for MLLMs and 15.4% for LLMs across compute-bound batching regimes.

## Abstract (Chinese)

基于树的投机解码通过在单次目标模型前向传播中验证分支草稿token树来加速自回归生成。但现有方法优先最大化token级似然或接受token数，忽略了关键的效率悖论：草拟和验证大树的计算开销可能超线性增长，导致批次增大或硬件饱和时出现负墙钟加速。SMART是运行时树构建的系统感知边际分析框架。将树扩展重新表述为直接最大化端到端加速的硬件感知优化问题。在推理时应用边际收益-成本规则，仅当边际收益-成本比超过树级加速时才扩展节点。SMART无需训练，可作为MSD和EAGLE等框架的即插即用控制器。在计算受限批次场景下，MLLM平均额外加速20.0%，LLM平均额外加速15.4%。

## Key Contributions

1. **SMART** — Tree-based speculative decoding accelerates autoregressive generation by verifying a branching tree ...
2. Addresses core challenges in Speculative Decoding systems
3. Demonstrates significant improvements over existing baselines

## Notes

- Added on 2026-04-16
- Paper published on 2026-04-09
