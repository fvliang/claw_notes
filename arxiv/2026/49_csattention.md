---
title: CSAttention: Centroid-Scoring Attention for Accelerating LLM Inference
authors: Chuxu Song, Zhencan Peng, Jiuqi Wei, Chuanhui Yang
arxiv_id: 
conference: arxiv
full_conference: ARXIV 2026
year: "2026"
topic: Attention
url: 
pdf_url: 
added_date: 2026-04-15
---

# CSAttention: Centroid-Scoring Attention for Accelerating LLM Inference

## 论文信息

- **arXiv**: 
- **会议**: ARXIV 2026
- **作者**: Chuxu Song, Zhencan Peng, Jiuqi Wei, Chuanhui Yang
- **主题**: Attention

## 摘要 (Abstract)

Long-context LLMs increasingly rely on extended, reusable prefill prompts for agents and domain Q&A, pushing attention and KV-cache to become the dominant decode-time bottlenecks. While sparse attention reduces computation and transfer costs, it often struggles to maintain accuracy at high sparsity levels due to distribution shift between query and key patterns. We propose CSAttention, a centroid-scoring approach that clusters KV-cache entries and scores cluster centroids to identify relevant blocks, enabling high sparsity with maintained accuracy.

## 摘要中文

长上下文LLM越来越多地依赖扩展的、可重用的预填充提示用于代理和领域问答，使注意力和KV缓存成为主要的解码时间瓶颈。虽然稀疏注意力减少了计算和传输成本，但由于查询和键模式之间的分布偏移，在高稀疏度下往往难以保持准确性。我们提出了CSAttention，一种质心评分方法，聚类KV缓存条目并评分聚类质心以识别相关块，实现在保持准确性的同时高稀疏度。

## 引言 (Introduction)

As LLMs handle increasingly long contexts, the KV cache grows proportionally, making attention computation a dominant bottleneck during decoding. Sparse attention methods attempt to reduce this cost but face accuracy challenges.

## 引言中文

随着LLM处理越来越长的上下文，KV缓存相应增长，使注意力计算成为解码期间的主要瓶颈。稀疏注意力方法试图减少这种成本但面临准确性挑战。

## 主要贡献

1. (待补充)

## 原文链接

- arXiv: (待确认)
- GitHub: (待补充)

## 补充材料

- 博客: (待补充)
- 相关GitHub: (待补充)

## 备注
