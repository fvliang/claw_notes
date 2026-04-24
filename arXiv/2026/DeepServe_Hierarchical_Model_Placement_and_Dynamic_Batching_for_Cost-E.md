# DeepServe: Hierarchical Model Placement and Dynamic Batching for Cost-Efficient Multi-Tenant LLM Inference at Scale

## Metadata
- **Authors:** Tejas Pravinbhai Patel, P. Agarwal
- **Conference:** arXiv 2026
- **Topic:** Inference Scheduling
- **arXiv ID:** 
- **Published:** 2026-02-20
- **GitHub:** sgl-project/SpecForge

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

Large Language Model (LLM) inference serving at scale presents critical challenges in multi-tenant cloud environments, where organizations must balance conflicting objectives of cost efficiency, response latency, and resource utilization across heterogeneous GPU infrastructure. Current serving systems employ static model placement strategies and homogeneous batching policies, resulting in $40 - 60 \%$ GPU underutilization and $3-5 \times$ cost inefficiency for variable workloads. We present DeepServe, a hierarchical model placement framework with dynamic batching that optimizes cost per served token across five GPU tiers (A100-80GB, A100-40GB, V100, L4, T4) and four model scales (7B-175B parameters). Our system integrates three key innovations: (1) Memoryaware hierarchical placement algorithm achieving 82% GPU utilization through optimal model-to-GPU assignment based on memory footprint and throughput profiles, (2) Adaptive continuous batching with SLA-aware scheduling attaining $3.8 \times$ throughput improvement over sequential processing while maintaining P95 latency under 2.5 seconds, and (3) Multi-objective cost optimizer reducing inference cost by 47% compared to static provisioning through heterogeneous GPU selection. Evaluation on production-representative workloads with 50,000 requests demonstrates: P50 latency of 890 ms, P95 latency of $2.31 ~\mathrm{s}, 79 \%$ average GPU utilization, and $0.0012 cost per 1000 tokens-matching vLLM throughput benchmarks while achieving 47% cost reduction versus naive A100-only deployment.

## 摘要 (中文)

[中文翻译待补充] Large Language Model (LLM) inference serving at scale presents critical challenges in multi-tenant cloud environments, where organizations must balance conflicting objectives of cost efficiency, respo...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

sgl-project/SpecForge

---
*Auto-collected on 2026-04-25*
