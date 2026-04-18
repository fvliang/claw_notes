# The Workload-Router-Pool Architecture for LLM Inference Optimization

**arXiv**: 2603.21354
**链接**: https://arxiv.org/abs/2603.21354
**作者**: Huamin Chen, Xunzhuo Liu, Bowei He, Fuyuan Lyu, Yankai Chen, Xue Liu, Yuhan Liu, Junchen Jiang
**会议**: arXiv 2026 (Vision Paper)
**主题**: llm_serving / Inference Routing / Architecture

## 摘要 (Abstract)

Over the past year, the vLLM Semantic Router project has released a series of work spanning: (1) core routing mechanisms -- signal-driven routing, context-length pool routing, router performance engineering, policy conflict detection, low-latency embedding models, category-aware semantic caching, user-feedback-driven routing adaptation, hallucination detection, and hierarchical content-safety classification for privacy and jailbreak protection; (2) fleet optimization -- fleet provisioning and energy-efficiency analysis; (3) agentic and multimodal routing -- multimodal agent routing, tool selection, CUA security, and multi-turn context memory and safety; (4) governance and standards -- inference routing protocols and multi-provider API extensions. Each paper tackled a specific problem in LLM inference, but the problems are not independent; for example, fleet provisioning depends on the routing policy, which depends on the workload mix, shifting as organizations adopt agentic and multimodal workloads. This paper distills those results into the Workload-Router-Pool (WRP) architecture, a three-dimensional framework for LLM inference optimization. Workload characterizes what the fleet serves (chat vs. agent, single-turn vs. multi-turn, warm vs. cold, prefill-heavy vs. decode-heavy). Router determines how each request is dispatched (static semantic rules, online bandit adaptation, RL-based model selection, quality-aware cascading). Pool defines where inference runs (homogeneous vs. heterogeneous GPU, disaggregated prefill/decode, KV-cache topology). We map our prior work onto a 3x3 WRP interaction matrix, identify which cells we have covered and which remain open, and propose twenty-one concrete research directions at the intersections, each grounded in our prior measurements, tiered by maturity from engineering-ready to open research.

## 摘要 (中文)

过去一年中，vLLM Semantic Router 项目发布了一系列工作：核心路由机制（信号驱动路由、上下文长度池路由、路由器性能工程、策略冲突检测等）、舰队优化（舰队供给和能效分析）、agent 和多模态路由、治理与标准等。每篇论文解决了 LLM 推理中的特定问题，但这些问题并不独立。本文将这些结果提炼为 Workload-Router-Pool (WRP) 架构——LLM 推理优化的三维框架。Workload 表征舰队服务的对象（chat vs agent，单轮 vs 多轮，prefill-heavy vs decode-heavy）；Router 决定请求的调度方式；Pool 定义推理运行的位置。将先前工作映射到 3x3 WRP 交互矩阵，识别已覆盖和开放的研究方向，并提出 21 个具体研究方向。

## 关键贡献

1. WRP 三维框架：Workload × Router × Pool
2. 3x3 交互矩阵系统化现有研究
3. 21 个具体研究方向，从工程成熟到开放研究分层