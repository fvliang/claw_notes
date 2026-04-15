---
title: Dual-Pool Token-Budget Routing for Cost-Efficient and Reliable LLM Serving
authors: Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen
arxiv_id: 
conference: arxiv
full_conference: ARXIV 2026
year: "2026"
topic: Scheduling
url: 
pdf_url: 
added_date: 2026-04-15
---

# Dual-Pool Token-Budget Routing for Cost-Efficient and Reliable LLM Serving

## 论文信息

- **arXiv**: 
- **会议**: ARXIV 2026
- **作者**: Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen
- **主题**: Scheduling

## 摘要 (Abstract)

Existing LLM serving systems typically configure each instance for worst-case context length, leading to substantial KV-cache over-allocation and under-utilized concurrency. In practice, 80-95% of requests are short, yet are served under configurations optimized for long contexts, wasting 4-8x throughput capacity and triggering reliability issues. We propose Dual-Pool, a token-budget routing system that maintains separate serving pools for short and long contexts, enabling cost-efficient and reliable LLM serving.

## 摘要中文

现有LLM服务系统通常为最坏情况的上下文长度配置每个实例，导致大量KV缓存过度分配和并发利用率不足。实际上，80-95%的请求是短请求，却在为长上下文优化的配置下服务，浪费4-8倍吞吐量容量并引发可靠性问题。我们提出了Dual-Pool，一个token预算路由系统，为短和长上下文维护分离的服务池，实现成本高效和可靠的LLM服务。

## 引言 (Introduction)

The mismatch between typical request lengths and worst-case-oriented configurations creates enormous inefficiency in LLM serving. Short requests suffer from unnecessary resource reservation while the system struggles to handle occasional long requests.

## 引言中文

典型请求长度与面向最坏情况配置之间的不匹配在LLM服务中造成了巨大效率损失。短请求遭受不必要的资源预留，而系统难以处理偶尔的长请求。

## 主要贡献

1. (待补充)

## 原文链接

- arXiv: (待确认)
- GitHub: (待补充)

## 补充材料

- 博客: (待补充)
- 相关GitHub: (待补充)

## 备注
