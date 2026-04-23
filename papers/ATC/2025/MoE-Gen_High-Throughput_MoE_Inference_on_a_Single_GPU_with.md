# MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching

**ArXiv ID:** 2503.09716
**Published:** 2025-03-12
**Authors:** Tairan Xu, Leyang Xue, Zhan Lu, Adrian Jackson, Luo Mai
**Conference/Venue:** arXiv.org
**URL:** https://arxiv.org/abs/2503.09716
**PDF:** https://arxiv.org/pdf/2503.09716
**GitHub:** EfficientMoE/MoE-Gen
**Categories:** Computer Science

## Abstract (English)

This paper presents MoE-Gen, a high-throughput MoE inference system optimized for single-GPU execution. Existing inference systems rely on model-based or continuous batching strategies, originally designed for interactive inference, which result in excessively small batches for MoE's key modules-attention and expert modules-leading to poor throughput. To address this, we introduce module-based batching, which accumulates tokens in host memory and dynamically launches large batches on GPUs to maximize utilization. Additionally, we optimize the choice of batch sizes for each module in an MoE to fully overlap GPU computation and communication, maximizing throughput. Evaluation demonstrates that MoE-Gen achieves 8-31x higher throughput compared to state-of-the-art systems employing model-based batching (FlexGen, MoE-Lightning, DeepSpeed), and offers even greater throughput improvements over continuous batching systems (e.g., vLLM and Ollama) on popular MoE models (DeepSeek and Mixtral) across offline inference tasks. MoE-Gen's source code is publicly available at https://github.com/EfficientMoE/MoE-Gen

## 摘要 (中文)

*(待翻译)*

## Introduction (English)

*(需要阅读原文PDF补充)*

## 引言 (中文)

*(需要阅读原文PDF补充)*

## 博客内容

*(待补充)*

## GitHub 介绍

https://github.com/EfficientMoE/MoE-Gen

---
*注: 此文件由晚间自动化论文搜集系统生成于 2026-04-23，部分内容待完善。*
