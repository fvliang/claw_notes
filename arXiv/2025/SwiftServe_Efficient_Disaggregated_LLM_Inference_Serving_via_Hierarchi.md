# SwiftServe: Efficient Disaggregated LLM Inference Serving via Hierarchical Max-Flow in Heterogeneous GPUs and Network

## Metadata
- **Authors:** Tao Zhang, Yan Hu, Shuangwu Chen, Zian Wang, Huihuang Qin
- **Conference:** arXiv 2025
- **Topic:** Prefill/Disaggregation
- **arXiv ID:** 
- **Published:** 2025-12-12
- **GitHub:** vllm-project/vllm

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

Large language models (LLMs) have achieved remarkable performance across a variety of tasks. Disaggregated LLM inference serving (DLIS), which separates the compute-intensive prefill phase and the memory-intensive decode phase, enables more flexible and efficient resource utilization in heterogeneous GPUs. However, deploying DLIS in real-world environments presents two significant challenges. Firstly, the heterogeneity of phase-specific resource requirements complicates the alignment of GPU capabilities with workload demands, often resulting in suboptimal performance. Secondly, transferring key-value (KV) caches between the prefill and decode phases over heterogeneous links introduces substantial communication overhead, creating performance bottlenecks. To address these challenges, we propose SwiftServe, an efficient disaggregated LLM inference serving system for heterogeneous GPUs. SwiftServe models DLIS deployment as a hierarchical max-flow deploying problem formulated as a constrained mixed-integer nonlinear program accounting for hardware and phase heterogeneity. We design a hierarchical alternating max-flow optimization algorithm for effective resource deployment. Experiments show SwiftServe achieves up to 1.68× higher throughput and 2.25× lower latency than existing methods.

## 摘要 (中文)

[中文翻译待补充] Large language models (LLMs) have achieved remarkable performance across a variety of tasks. Disaggregated LLM inference serving (DLIS), which separates the compute-intensive prefill phase and the mem...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

vllm-project/vllm

---
*Auto-collected on 2026-04-27 morning*
