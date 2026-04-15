---
title: Token-Budget-Aware Pool Routing for Cost-Efficient LLM Inference
authors: Huamin Chen, Xunzhuo Liu, Junchen Jiang, Bowei He, Xue Liu
arxiv_id: 
conference: arxiv
full_conference: ARXIV 2026
year: "2026"
topic: Scheduling
url: 
pdf_url: 
added_date: 2026-04-15
---

# Token-Budget-Aware Pool Routing for Cost-Efficient LLM Inference

## 论文信息

- **arXiv**: 
- **会议**: ARXIV 2026
- **作者**: Huamin Chen, Xunzhuo Liu, Junchen Jiang, Bowei He, Xue Liu
- **主题**: Scheduling

## 摘要 (Abstract)

We present token-budget routing, a simple yet effective approach that reduces GPU costs for LLM inference by routing requests to appropriately-sized serving pools. Our theoretical analysis shows that GPU savings follow the formula alpha * (1 - 1/rho), predicting fleet-level GPU savings from two observable quantities: the short-traffic fraction alpha and the throughput gain ratio rho. On traces from the Azure LLM Inference Dataset and LMSYS-Chat-1M serving Llama-3-70B on A100 GPUs, token-budget routing reduces GPU costs significantly.

## 摘要中文

我们提出了token预算路由，一种简单而有效的方法，通过将请求路由到适当大小的服务池来降低LLM推理的GPU成本。我们的理论分析表明，GPU节省遵循公式alpha * (1 - 1/rho)，从两个可观测量预测舰队级GPU节省：短流量比例alpha和吞吐量增益比率rho。在Azure LLM推理数据集和LMSYS-Chat-1M的追踪数据上，使用A100 GPU服务Llama-3-70B，token预算路由显著降低了GPU成本。

## 引言 (Introduction)

LLM inference costs are dominated by GPU resources, and the current practice of provisioning for worst-case scenarios leads to massive waste. Most requests are short but are served on instances sized for the longest possible context.

## 引言中文

LLM推理成本主要由GPU资源主导，当前为最坏情况配置的做法导致大量浪费。大多数请求是短请求，却在为最长可能上下文大小的实例上服务。

## 主要贡献

1. (待补充)

## 原文链接

- arXiv: (待确认)
- GitHub: (待补充)

## 补充材料

- 博客: (待补充)
- 相关GitHub: (待补充)

## 备注
