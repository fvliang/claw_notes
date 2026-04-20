# Ragged Paged Attention: A High-Performance and Flexible LLM Inference Kernel for TPU

- **Arxiv ID**: 2604.15464
- **Conference**: arxiv 2026
- **Link**: https://arxiv.org/abs/2604.15464
- **GitHub**: 
- **Tags**: llm-inference, tpu, attention-kernel, kv-cache

## Abstract (English)

We present Ragged Paged Attention (RPA), a high-performance and flexible LLM inference kernel designed for TPU. Unlike GPUs that rely on the Paged Attention technique with block tables, TPUs employ a distinct execution model characterized by larger-scale matrix multiplication units and a software-managed scratchpad memory called VMEM. RPA addresses the challenges of TPU-based inference by leveraging a ragged tensor representation and VMEM-based paged KV cache management. Our approach efficiently handles variable sequence lengths and dynamic KV cache allocations without the overhead of block table lookups. Experimental results demonstrate that RPA achieves significant performance improvements over baseline implementations, enabling efficient LLM inference on TPU architectures.

## Abstract (Chinese)

我们提出了Ragged Paged Attention (RPA)，一种面向TPU的高性能灵活LLM推理内核。不同于GPU依赖Paged Attention技术与块表，TPU采用以大规模矩阵乘法单元和软件管理的VMEM暂存内存为特征的执行模型。RPA利用ragged张量表示和基于VMEM的分页KV缓存管理，解决了TPU推理的挑战，高效处理可变序列长度和动态KV缓存分配，无需块表查找开销。

## Introduction (English)

Large language model (LLM) inference has become a critical workload in modern computing infrastructure. While GPU-based systems have been extensively optimized for LLM serving through techniques such as Paged Attention and vLLM, TPU-based inference remains relatively underexplored. TPUs offer distinct advantages for LLM inference, including larger matrix multiplication units and deterministic performance characteristics, but require fundamentally different memory management strategies. The key challenge lies in efficiently managing the KV cache—the memory allocated for storing key and value tensors during autoregressive generation—on TPU's software-managed scratchpad memory (VMEM).

## Introduction (Chinese)

LLM推理已成为现代计算基础设施中的关键工作负载。虽然GPU系统通过Paged Attention和vLLM等技术已被大量优化，但TPU推理仍相对未被充分探索。TPU为LLM推理提供了独特优势，包括更大的矩阵乘法单元和确定性性能特征，但需要根本不同的内存管理策略。核心挑战在于如何在TPU的软件管理暂存内存(VMEM)上高效管理KV缓存。

## GitHub Introduction

N/A - No GitHub repository found for this paper.

## Blog Content

N/A - No blog post found for this paper.

---
*Auto-collected on 2026-04-21*
