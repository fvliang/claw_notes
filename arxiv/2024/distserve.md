# DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving

## 论文信息
- **作者**: Y. Sheng, L. Zheng, Y. Zhu, et al. (PKU)
- **会议**: OSDI 2024
- **arXiv**: https://arxiv.org/abs/2401.09670
- **GitHub**: https://github.com/LLMServe/DistServe
- **日期**: 2024.01

## 摘要 (Abstract)
Large Language Models (LLMs) are increasingly deployed in latency-sensitive applications. However, existing systems couple the processing of prefilling (computng prompt embeddings) and decoding (autoregressively generating tokens) phases into a single GPU, leading to severe interference: a long prompt forces all requests to wait, while a long decoding request slows down the processing of other requests.

We propose DistServe, a system that disaggregates prefilling and decoding to different GPUs. The key insight is that isolating the two phases eliminates interference and enables tailoring the hardware configuration for each. We develop an efficient method to determine the optimal placement of prefilling and decoding workloads on heterogeneous GPU clusters. We implement DistServe and evaluate it on a cluster with 16 A100 GPUs. Compared to existing systems like vLLM and Orca, DistServe improves throughput by up to 2.6x while ensuring the latency SLOs are met.

## 摘要中文
大型语言模型（LLM）越来越多地部署在延迟敏感的应用中。然而，现有系统将预填充（计算提示嵌入）和解码（自回归生成token）阶段耦合到单个GPU中，导致严重的干扰：长prompt强制所有请求等待，而长解码请求会减慢其他请求的处理。

我们提出了DistServe，这是一个将预填充和解码分配到不同GPU的系统。关键在于隔离这两个阶段可以消除干扰，并为每个阶段定制硬件配置。我们开发了一种高效的方法来确定异构GPU集群上预填充和解码工作负载的最佳放置位置。我们实现了DistServe并在具有16个A100 GPU的集群上对其进行了评估。与vLLM和Orca等现有系统相比，DistServe在确保延迟SLO的同时，将吞吐量提高了2.6倍。

## 引言 (Introduction)
LLM serving differs from traditional ML inference in that each request consists of two phases: prefilling (processing the prompt and computing intermediate KV cache) and decoding (autoregressively generating tokens). Prior systems interleave these two phases in a single batch, which leads to significant challenges:

1. ** interference between requests with different characteristics
2. **Inability to optimize each phase independently
3. **Poor resource utilization due to different compute/memory requirements

DistServe addresses these by disaggregating prefilling and decoding onto separate GPU groups.

## GitHub 介绍
DistServe is a system for disaggregated LLM serving that separates prefill and decode phases onto different GPUs. It eliminates interference between requests and allows optimal hardware configuration for each phase.