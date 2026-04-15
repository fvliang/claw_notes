# RouterWise: Joint Resource Allocation and Routing for Latency-Aware Multi-Model LLM Serving

**Source:** arxiv | **Category:** LLM Serving | **Date:** 2026-04-12
**ArXiv ID:** 2604.10907
**Authors:** Hossein Hosseini Kasnavieh, Christopher Leckie, Adel N. Toosi
**Tags:** multi-model-routing, resource-allocation, latency-aware, joint-optimization, routerwise

## Links

- 📄 [Paper (PDF)](https://arxiv.org/pdf/2604.10907)
- 🌐 [ArXiv Page](https://arxiv.org/abs/2604.10907)

## Abstract (English)

Multi-model LLM routing has emerged as an effective approach for reducing serving cost and latency while maintaining output quality. However, prior routing methods typically assume each model has fixed latency. In real deployments, multiple models often share limited GPU resources, and a model's latency depends strongly on both its allocated resources and the request load induced by the routing policy. Consequently, routing and resource allocation are tightly coupled. RouterWise formalizes this as a constrained joint optimization over deployment setup and routing fractions, combining a dual-price formulation for score-maximizing routing with setup-specific latency models derived from system profiling. Results show that achievable output-quality score can vary by up to 87% across retained setups on the same GPU cluster.

## Abstract (Chinese)

多模型LLM路由已成为降低服务成本和延迟同时保持输出质量的有效方法。但先前路由方法假设每个模型有固定延迟，这在实际部署中不准确：多个模型往往共享有限的GPU资源，模型的延迟很大程度上取决于其分配资源和路由策略引起的请求负载。路由和资源分配紧密耦合。RouterWise将此问题形式化为部署设置和路由比例的约束联合优化，结合双价格公式进行分数最大化路由和基于系统分析的设置特定延迟模型。结果显示，在同一GPU集群上，可实现的输出质量分数在不同设置间变化高达87%。

## Key Contributions

1. **RouterWise** — Multi-model LLM routing has emerged as an effective approach for reducing serving cost and latency w...
2. Addresses core challenges in LLM Serving systems
3. Demonstrates significant improvements over existing baselines

## Notes

- Added on 2026-04-16
- Paper published on 2026-04-12
