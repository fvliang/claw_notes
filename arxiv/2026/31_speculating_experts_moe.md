# Speculating Experts: Accelerates Inference for Mixture-of-Experts

## 论文信息

- **作者**: Vivan Madan, Prajwal Singhania, Abhinav Bhatele, Tom Goldstein, Ashwinee Panda
- **提交日期**: 2026年3月9日
- **arXiv**: (待查询)
- **领域**: cs.LG, cs.DC

## 摘要 (Abstract)

> ...per-token compute. However, in memory-constrained inference settings, expert weights must be offloaded to CPU, creating a performance bottleneck from CPU-GPU transfers during decoding. We propose an expert prefetching scheme that leverages currently computed internal model representations to...

## 摘要 (中文)

本文研究了在内存受限的推理环境中,Mixture-of-Experts (MoE)模型的推理加速问题。由于专家权重必须卸载到CPU,在解码过程中会产生CPU-GPU传输的性能瓶颈。文章提出了一种专家预取方案,利用当前计算的内部模型表示来进行专家权重的预取,从而减少传输延迟。

## 引言 (Introduction)

MoE架构通过稀疏激活来增强大语言模型的可扩展性,使其适合在资源受限的边缘网络中部署。然而,大量的专家数量通常超过单个边缘节点的内存容量,需要无线分布式MoE (WIDE)推理。

## GitHub

- (待添加)

## 博客/介绍

- (待添加)

## 原文链接

- arXiv: (待查询)