# ConfLayers: Adaptive Confidence-based Layer Skipping for Self-Speculative Decoding

**Authors:** Walaa Amer, Uday Das, Fadi Kurdahi

**Conference:** arXiv 2026

**Year:** 2026

**ArXiv:** [2604.14612](<https://arxiv.org/abs/2604.14612>)

**Topic:** Speculative Decoding

---

## Abstract (English)

Self-speculative decoding is an inference technique for large language models designed to speed up generation without sacrificing output quality. It combines fast, approximate decoding using a compact version of the model as a draft model with selective re-evaluation by the full target model. Some existing methods form the draft model by dynamically learning which layers to skip during inference, effectively creating a smaller subnetwork to speed up computation. However, using heuristic-based approaches to select layers to skip can often be simpler and more effective. In this paper, we propose ConfLayers, a dynamic plug-and-play approach to forming the draft model in self-speculative decoding via confidence-based intermediate layer skipping. The process iteratively computes confidence scores for all layers, selects layers to skip based on an adaptive threshold, evaluates the performance of the resulting set, and updates the best selection until no further improvement is achieved or a maximum number of iterations is reached. This framework avoids the overhead and complexity of training a layer skipping policy and can provide more consistent speed-quality trade-offs while preserving the adaptivity of the draft model to diverse tasks and datasets. The performance evaluation of ConfLayers across different models and datasets shows that our novel approach offers up to 1.4x speedup over vanilla LLM generation.

## Abstract (Chinese / 中文摘要)

自投机解码是大语言模型的一种推理技术，旨在在不牺牲输出质量的情况下加速生成。它使用模型的紧凑版本作为草案模型进行快速近似解码，并与完整目标模型的选择性重新评估相结合。一些现有方法通过动态学习推理期间跳过哪些层来形成草案模型，有效地创建更小的子网络以加速计算。然而，使用启发式方法选择跳过层通常更简单且更有效。在本文中，我们提出ConfLayers，一种动态即插即用方法，通过基于置信度的中间层跳过来形成自投机解码的草案模型。该过程迭代计算所有层的置信度分数，基于自适应阈值选择跳过的层，评估结果集的性能，并更新最佳选择。此框架避免了训练层跳过策略的开销和复杂性，可以提供更一致的速度-质量权衡。性能评估显示ConfLayers提供高达1.4倍的加速。

---

*Auto-collected from arXiv on 2026-04-17*
