---
title: Foundry: Template-Based CUDA Graph Context Materialization for Fast LLM Serving Cold Start
authors: Xueshen Liu, Yongji Wu, Yuncheng Yao, Danyang Zhuo, Ion Stoica, Z. Morley Mao
arxiv_id: 
conference: arxiv
full_conference: ARXIV 2026
year: "2026"
topic: LLM Serving
url: 
pdf_url: 
added_date: 2026-04-15
---

# Foundry: Template-Based CUDA Graph Context Materialization for Fast LLM Serving Cold Start

## 论文信息

- **arXiv**: 
- **会议**: ARXIV 2026
- **作者**: Xueshen Liu, Yongji Wu, Yuncheng Yao, Danyang Zhuo, Ion Stoica, Z. Morley Mao
- **主题**: LLM Serving

## 摘要 (Abstract)

Modern LLM service providers increasingly rely on autoscaling and parallelism reconfiguration to respond to rapidly changing workloads, but cold-start latency remains a critical bottleneck. CUDA graph capture, which enables fast kernel dispatch, requires significant setup time that delays new instance readiness. We present Foundry, a template-based approach that pre-materializes CUDA graph contexts for common configurations, enabling near-instant cold starts when deploying new LLM serving instances.

## 摘要中文

现代LLM服务提供商越来越依赖自动扩展和并行重配置来响应快速变化的工作负载，但冷启动延迟仍然是关键瓶颈。CUDA图捕获（实现快速内核调度）需要大量设置时间，延迟了新实例的启动。我们提出了Foundry，一种基于模板的方法，预先为常见配置物化CUDA图上下文，在部署新LLM服务实例时实现近乎即时的冷启动。

## 引言 (Introduction)

The shift to autoscaling in LLM serving means new instances are frequently spun up and torn down. Each cold start involves model loading, memory allocation, and CUDA graph capture—operations that can take minutes. Foundry addresses this by pre-building reusable CUDA graph templates.

## 引言中文

LLM服务向自动扩展的转变意味着新实例频繁启动和关闭。每次冷启动涉及模型加载、内存分配和CUDA图捕获——这些操作可能需要数分钟。Foundry通过预先构建可重用的CUDA图模板来解决这个问题。

## 主要贡献

1. (待补充)

## 原文链接

- arXiv: (待确认)
- GitHub: (待补充)

## 补充材料

- 博客: (待补充)
- 相关GitHub: (待补充)

## 备注
