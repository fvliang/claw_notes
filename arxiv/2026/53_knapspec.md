---
title: KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem
authors: Multiple Authors
arxiv_id: 
conference: arxiv
full_conference: ARXIV 2026
year: "2026"
topic: Speculative Decoding
url: 
pdf_url: 
added_date: 2026-04-15
---

# KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem

## 论文信息

- **arXiv**: 
- **会议**: ARXIV 2026
- **作者**: Multiple Authors
- **主题**: Speculative Decoding

## 摘要 (Abstract)

Self-speculative decoding skips intermediate model layers to generate draft tokens, but selecting which layers to skip is challenging. We formulate layer selection as a knapsack optimization problem, where each layer's contribution to draft quality is weighed against its computational cost. KnapSpec dynamically selects the optimal layer skipping pattern for each input, maximizing drafting accuracy while minimizing computation overhead.

## 摘要中文

自投机解码跳过中间模型层来生成起草token，但选择跳过哪些层具有挑战性。我们将层选择形式化为背包优化问题，其中每层对起草质量的贡献与其计算成本权衡。KnapSpec为每个输入动态选择最优层跳过模式，最大化起草准确性同时最小化计算开销。

## 引言 (Introduction)

Self-speculative decoding offers a way to accelerate LLM inference without requiring a separate draft model. However, deciding which layers to skip for drafting is non-trivial—skipping too many reduces draft quality while skipping too few provides little acceleration.

## 引言中文

自投机解码提供了无需单独起草模型即可加速LLM推理的方法。然而，决定跳过哪些层进行起草并非易事——跳过太多降低起草质量，跳过太少提供很少加速。

## 主要贡献

1. (待补充)

## 原文链接

- arXiv: (待确认)
- GitHub: (待补充)

## 补充材料

- 博客: (待补充)
- 相关GitHub: (待补充)

## 备注
