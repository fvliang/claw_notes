# Characterizing Performance-Energy Trade-offs of Large Language Models in Multi-Request Workflows

**arXiv**: 2604.09611
**链接**: https://arxiv.org/abs/2604.09611
**作者**: Md. Monzurul Amin Ifath, Israat Haque
**会议**: arXiv 2026
**主题**: llm_serving / Energy Efficiency / Multi-Request Workflows

## 摘要 (Abstract)

Large language models (LLMs) are increasingly used in applications forming multi-request workflows like document summarization, search-based copilots, and multi-agent programming. While these workflows unlock richer functionality, they also amplify latency and energy demand during inference. Existing measurement and benchmarking efforts either focus on assessing LLM inference systems or consider single-request evaluations, overlooking workflow dependencies and cross-request interactions unique to multi-request workflows. Moreover, the energy usage of such interdependent LLM calls remains underexplored. To address these gaps, this paper presents the first systematic characterization of performance-energy trade-offs in multi-request LLM inference. We develop four representative workloads capturing sequential, interactive, agentic, and composite patterns common in modern deployments. Using an NVIDIA A100 testbed with state-of-the-art serving systems (vLLM and Parrot), we analyze how key energy knobs affect latency, throughput, and component-level energy use. Our findings reveal batch size as the most impactful lever, though benefits are workload dependent. While optimal batching benefits workloads with large shared prompts, it is ineffective for sequential summarization and only partially effective for multi-agent coding. GPU power capping provides modest but predictable savings, while output length induces linear energy scaling with limited efficiency gains. We further show that engine-level optimizations in vLLM maintain higher GPU utilization and efficiency, especially for decode-heavy workloads, while Parrot's workflow-aware scheduling achieves lower energy consumption under strict power constraints.

## 摘要 (中文)

LLMs 日益用于形成多请求工作流的应用中，如文档摘要、搜索辅助和多 agent 编程。这些工作流解锁了更丰富的功能，但也放大了推理过程中的延迟和能耗。现有评估工作要么关注 LLM 推理系统要么考虑单请求评估，忽略了多请求工作流特有的依赖关系和跨请求交互。本文首次系统性地刻画了多请求 LLM 推理中的性能-能效权衡。开发了四种代表性工作负载（顺序、交互、agent 和复合模式），使用 NVIDIA A100 测试平台和 vLLM/Parrot 服务系统分析关键能效参数如何影响延迟、吞吐量和组件级能耗。发现 batch size 是最有影响力的调节杠杆，但收益取决于工作负载类型。GPU 功率上限提供适度但可预测的节省，输出长度导致线性能耗增长。vLLM 的引擎级优化在 decode-heavy 工作负载中维持更高的 GPU 利用率和效率。

## 关键贡献

1. 首次系统化刻画多请求 LLM 推理的性能-能效权衡
2. 四种代表性工作负载（顺序/交互/agent/复合）
3. batch size 是最关键的能效杠杆，但效果依赖工作负载类型
4. vLLM vs Parrot 的对比分析