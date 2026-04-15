# StepCache: Step-Level Reuse with Lightweight Verification and Selective Patching for LLM Serving

**Source:** arxiv | **Category:** LLM Serving | **Date:** 2026-03-24
**ArXiv ID:** 2603.28795
**Authors:** Azam Nouri
**Tags:** step-level-caching, selective-patching, structured-output, reuse, stepcache

## Links

- 📄 [Paper (PDF)](https://arxiv.org/pdf/2603.28795)
- 🌐 [ArXiv Page](https://arxiv.org/abs/2603.28795)

## Abstract (English)

StepCache addresses LLM serving workloads where repeated requests share common solution structure but differ in localized constraints. Prior caching approaches reuse either full responses (semantic caching) or model-internal KV/prefix states, which are respectively brittle under partial changes or tightly coupled to specific backends. StepCache is a backend-agnostic step-level reuse layer that segments outputs into ordered steps, retrieves the best-matching cached request, verifies steps using lightweight task-aware checks, and regenerates only failing regions via selective patching. It supports strict structured-output enforcement for JSON, including single-step extraction and required-key constraints. In perturbation-heavy micro-benchmarks, StepCache reduces mean latency from 2.13s to 0.67s, median latency from 2.42s to 0.01s, and improves end-to-end correctness from 72.5% to 100%. 79.7% of requests take the reuse-only fast path.

## Abstract (Chinese)

StepCache处理LLM服务中重复请求共享解决方案结构但局部约束不同的场景。先前缓存方法要么重用完整响应（语义缓存），要么重用模型内部KV/前缀状态，前者在部分变更下脆弱，后者与特定后端紧密耦合。StepCache是后端无关的步骤级重用层，将输出分段为有序步骤，检索最佳匹配的缓存请求，使用轻量任务感知检查验证步骤，并通过选择性补丁仅重新生成失败区域。支持JSON的严格结构化输出强制执行。在扰动密集的微基准测试中，StepCache将平均延迟从2.13秒降至0.67秒，中位延迟从2.42秒降至0.01秒，端到端正确性从72.5%提升至100%。79.7%的请求走重用快速路径。

## Key Contributions

1. **StepCache** — StepCache addresses LLM serving workloads where repeated requests share common solution structure bu...
2. Addresses core challenges in LLM Serving systems
3. Demonstrates significant improvements over existing baselines

## Notes

- Added on 2026-04-16
- Paper published on 2026-03-24
