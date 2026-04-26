# Attention Distribution-Aware Softmax for NPU-Accelerated On-Device Inference of LLMs: An Edge-Oriented Approximation Design

## Metadata
- **Authors:** Sanoop Sadheerthan, Min-Jie Hsu, Chihan Huang, Yin-Tien Wang
- **Conference:** arXiv 2026
- **Topic:** Edge Inference
- **arXiv ID:** 
- **Published:** 2026-03-20
- **GitHub:** 

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

Low-power NPUs enable on-device LLM inference through efficient integer and fixed-point algebra, yet their lack of native exponential support makes Transformer softmax a critical performance bottleneck. Existing NPU kernels approximate using uniform piecewise polynomials to enable O(1) SIMD indexing, but this wastes computation by applying high-degree arithmetic indiscriminately in every segment. Conversely, fully adaptive approaches maximize statistical fidelity but introduce pipeline stalls due to comparator-based boundary search. To bridge this gap, we propose an attention distribution-aware softmax that uses Particle Swarm Optimization (PSO) to define non-uniform segments and variable polynomial degrees, prioritizing finer granularity and lower arithmetic complexity in attention-dense regions. To ensure efficiency, we snap boundaries into a 128-bin LUT, enabling O(1) retrieval of segment parameters without branching. Inference measurements show that this favors low-degree execution, minimizing exp-kernel overhead. Using TinyLlama-1.1B-Chat as a testbed, the proposed weighted design reduces cycles per call exp kernel (CPC) by 18.5% versus an equidistant uniform Degree-4 baseline and 13.1% versus uniform Degree-3, while preserving ranking fidelity. These results show that grid-snapped, variable-degree approximation can improve softmax efficiency while largely preserving attention ranking fidelity, enabling accurate edge LLM inference.

## 摘要 (中文)

[中文翻译待补充] Low-power NPUs enable on-device LLM inference through efficient integer and fixed-point algebra, yet their lack of native exponential support makes Transformer softmax a critical performance bottlenec...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

[GitHub仓库待搜索补充]

---
*Auto-collected on 2026-04-26 evening*
