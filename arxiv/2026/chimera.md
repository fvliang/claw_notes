# Chimera: Latency- and Performance-Aware Multi-agent Serving for Heterogeneous LLMs

**Source:** arxiv | **Category:** LLM Serving | **Date:** 2026-03-23
**ArXiv ID:** 2603.22206
**Authors:** Kangqi Ni, Wenyue Hua, Xiaoxiang Shi, Jiang Guo, Shiyu Chang, Tianlong Chen
**Tags:** multi-agent-serving, heterogeneous-llm, semantic-routing, load-balancing, chimera

## Links

- 📄 [Paper (PDF)](https://arxiv.org/pdf/2603.22206)
- 🌐 [ArXiv Page](https://arxiv.org/abs/2603.22206)

## Abstract (English)

Multi-agent applications execute complex tasks as multi-stage workflows where each stage is an LLM call. Existing LLM serving systems largely assume homogeneous clusters with identical model replicas, overlooking the potential of heterogeneous deployments where models of different sizes enable finer latency-performance trade-offs. Chimera is a predictive scheduling system for multi-agent workflow serving on heterogeneous LLM clusters. It applies semantic routing to estimate per-model confidence scores, predicts total remaining output length, and estimates per-model congestion using in-flight predicted token volumes for load balancing. Evaluated on code generation and math reasoning workflows, Chimera traces the best latency-performance frontier, reducing end-to-end latency by 1.2-2.4x and improving task performance by 8.0-9.5 percentage points over competitive baselines including vLLM.

## Abstract (Chinese)

多智能体应用将复杂任务作为多阶段工作流执行，每个阶段是一个LLM调用。现有LLM服务系统大多假设同构集群（相同模型副本），忽略了异构部署的潜力——不同大小和能力模型可以实现更精细的延迟-性能权衡。Chimera是异构LLM集群上多智能体工作流服务的预测调度系统。应用语义路由估计每个模型的置信度分数，预测工作流总剩余输出长度，并使用在途预测token量估计每模型拥塞以进行负载均衡。在代码生成和数学推理工作流上评估，Chimera追踪最佳延迟-性能前沿，端到端延迟降低1.2-2.4倍，任务性能比vLLM等基线改善8.0-9.5个百分点。

## Key Contributions

1. **Chimera** — Multi-agent applications execute complex tasks as multi-stage workflows where each stage is an LLM c...
2. Addresses core challenges in LLM Serving systems
3. Demonstrates significant improvements over existing baselines

## Notes

- Added on 2026-04-16
- Paper published on 2026-03-23
