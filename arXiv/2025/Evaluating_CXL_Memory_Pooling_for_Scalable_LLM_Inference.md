# Evaluating CXL Memory Pooling for Scalable LLM Inference

## Metadata
- **Authors:** Sai Krishna Vemuri, Venkata Ravi Shankar Jonnalagadda, Ajay Joshi, Rohit Sindhu, Vijay Kumar Motagi
- **Conference:** arXiv 2025
- **Topic:** KV Cache
- **arXiv ID:** 
- **Published:** 2025-12-17
- **GitHub:** vllm-project/vllm

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

Large-context LLM inference increasingly faces bottlenecks in Key-Value (KV) cache capacity and bandwidth rather than raw compute. While on-package HBM delivers exceptional bandwidth, its capacity is limited; when long prompts or multi-tenant workloads exceed the HBM KV cache budget, operators turn to re-computation or offloading to system memory or storage—both with significant performance costs. Compute Express Link (CXL) introduces a flexible, low-latency, load/store memory tier that can be pooled and shared across nodes using PCIe 6.0 PHY, multi-level switching, and fabric features in CXL 3.x. This paper examines when and why CXL memory pooling helps LLM serving: we explain the inference pipeline and KV cache growth dynamics, discuss architectural advantages and deployment considerations for CXL pooling, and present a compact analytical model of per-accelerator throughput vs. context length that compares HBM-only with re-computation, HBM + distributed DDR, and HBM + CXL pooled memory. With realistic parameters from contemporary GPUs, 800 GbE networks, and CXL fabrics, our model shows that CXL pooling sustains substantially higher throughput beyond the HBM-fit regime (e.g., ≥ 100 k tokens) by reducing per-segment residual stalls compared to distributed DDR, while avoiding the severe collapse of re-computation. We close with sensitivity analyses, limitations, and guidance for deploying CXL fabrics in long-context serving stacks.

## 摘要 (中文)

[中文翻译待补充] Large-context LLM inference increasingly faces bottlenecks in Key-Value (KV) cache capacity and bandwidth rather than raw compute. While on-package HBM delivers exceptional bandwidth, its capacity is ...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

vllm-project/vllm

---
*Auto-collected on 2026-04-25*
