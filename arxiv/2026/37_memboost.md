# MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference

## 论文信息

- **标题**: MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference
- **作者**: Joris Köster, Zixuan Liu, Siavash Khajavi, Zizhan Zheng
- **arXiv**: 2603.26557
- **会议/来源**: arXiv
- **年份**: 2026
- **主题**: LLM Serving
- **提交日期**: 2026年3月27日

## 原文链接

- arXiv: https://arxiv.org/abs/2603.26557
- PDF: https://arxiv.org/pdf/2603.26557

## 摘要 (英文)

Large Language Models (LLMs) deliver strong performance but incur high inference cost in real-world services, especially under workloads with repeated or near-duplicate queries across users and sessions. In this work, we propose MemBoost, a memory-boosted LLM serving framework that enables a lightweight model to reuse previously generated answers and retrieve relevant supporting information for cheap inference, while selectively escalating difficult or uncertain queries to a stronger model. Unlike standard retrieval-augmented generation, which primarily grounds a single response, MemBoost is designed for interactive settings by supporting answer reuse, continual memory growth, and cost-aware routing. Experiments across multiple models under simulated workloads show that MemBoost substantially reduces expensive large-model invocations and overall inference cost, while maintaining high answer quality comparable to the strong model baseline.

## 摘要 (中文)

大型语言模型(LLM)在现实场景中表现出强大的性能,但推理成本较高,特别是在跨用户和会话的重复或近似重复查询工作负载下。本工作提出MemBoost,一个内存增强的LLM服务框架,使轻量级模型能够重用先前生成的答案并检索相关支持信息以实现低成本推理,同时有选择地将困难或不确定的查询升级到更强的模型。与标准检索增强生成(主要为单个响应提供基础)不同,MemBoost专为交互式场景设计,支持答案重用、持续内存增长和成本感知路由。在模拟工作负载下针对多个模型的实验表明,MemBoost显著减少了昂贵的大型模型调用和整体推理成本,同时保持与强模型基线相当的高答案质量。

## 引言 (英文)

The deployment of Large Language Models (LLMs) in production services has grown exponentially, powering applications from chatbots to code assistants. However, serving LLMs at scale remains computationally expensive, with inference costs dominating the operational expenses of AI services. A key observation in real-world workloads is that many queries are repeated or semantically similar across users and sessions, yet current serving systems treat each request independently, leading to redundant computation.

In this work, we propose MemBoost, a novel memory-boosted LLM serving framework that addresses this inefficiency through intelligent answer reuse and cost-aware routing. MemBoost maintains a memory of previously generated answers and uses a lightweight model to either directly reuse relevant answers or retrieve supporting information for context-enhanced inference. Only when queries are sufficiently novel or challenging does MemBoost escalate to a more powerful (and expensive) model.

## 引言 (中文)

大型语言模型(LLM)在生产服务中的部署呈指数级增长,从聊天机器人到代码助手,为各种应用提供支持。然而,在规模上服务LLM仍然计算成本高昂,推理成本占AI服务运营费用的很大一部分。实际工作负载中的一个关键观察是,许多查询在用户和会话之间是重复的或语义相似的,但当前的服务系统独立处理每个请求,导致冗余计算。

本工作提出MemBoost,一种新型的内存增强LLM服务框架,通过智能答案重用和成本感知路由来解决这一低效率问题。MemBoost维护先前生成答案的内存,并使用轻量级模型直接重用相关答案或检索支持信息以进行上下文增强推理。只有当查询足够新颖或具有挑战性时,MemBoost才会升级到更强大(且更昂贵)的模型。

## 核心贡献

1. **内存增强框架**: 提出MemBoost框架,支持答案重用、持续内存增长和成本感知路由
2. **成本感知路由**: 根据查询难度动态选择使用轻量级还是强大的模型
3. **显著成本降低**: 实验表明可大幅减少大型模型调用和整体推理成本