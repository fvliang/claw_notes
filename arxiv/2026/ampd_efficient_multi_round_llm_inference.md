# AMPD: Efficient Multi-round LLM Inference over Disaggregated Serving

**arXiv**: 2602.14516
**链接**: https://arxiv.org/abs/2602.14516
**作者**: Wenhao He, Youhe Jiang, Penghao Zhao, Quanqing Xu, Eiko Yoneki, Bin Cui, Fangcheng Fu
**会议**: arXiv 2026
**主题**: llm_serving / Disaggregated Serving / Multi-round Inference

## 摘要 (Abstract)

With the rapid evolution of Large Language Models (LLMs), multi-round workflows, such as autonomous agents and iterative retrieval, have become increasingly prevalent. However, this raises hurdles for serving LLMs under prefill-decode (PD) disaggregation, a widely adopted paradigm that separates the compute-bound prefill phase and memory-bound decode phase onto individual resources. Specifically, existing systems overlook the interleaved prefill-decode workload pattern in multi-round inference, leading to sub-optimal handling of the incremental prefill workloads and model deployment for the two phases. In this work, we present AMPD, a brand new disaggregated serving framework for multi-round LLM inference. The core of AMPD is to coordinate the prefill workloads based on real-time workloads by adaptively determining where to carry out these workloads and how they are scheduled, in order to maximize service level objective (SLO) attainment. In addition, we tailor a planning algorithm for our scenario, facilitating the deduction of optimal resource allocation and parallel strategies for the two phases. Empirical results demonstrate that AMPD substantially improves SLO attainment compared to state-of-the-art baselines.

## 摘要 (中文)

随着 LLM 的快速发展，多轮工作流（如自主 agent 和迭代检索）日益普遍。然而，这对 prefill-decode (PD) 解耦服务范式带来了挑战——该范式将计算密集的 prefill 阶段和内存密集的 decode 阶段分离到独立资源上。现有系统忽略了多轮推理中交错 prefill-decode 工作负载模式，导致增量 prefill 工作负载和两阶段模型部署的次优处理。本文提出 AMPD，一种全新的多轮 LLM 推理解耦服务框架。AMPD 的核心是基于实时工作负载协调 prefill 工作负载，自适应确定执行位置和调度方式，以最大化 SLO 达成率。此外，我们为该场景定制了规划算法，推导两阶段的最优资源分配和并行策略。实验表明 AMPD 相比最先进基线显著提升了 SLO 达成率。

## 关键贡献

1. 馢应式协调多轮推理中的 prefill 工作负载
2. 自适应确定 prefill 执行位置与调度方式
3. 专用规划算法推导最优资源分配和并行策略
4. 显著提升 SLO 达成率