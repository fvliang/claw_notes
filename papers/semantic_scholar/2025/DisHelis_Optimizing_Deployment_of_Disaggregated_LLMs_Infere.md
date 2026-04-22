# DisHelis: Optimizing Deployment of Disaggregated LLMs Inference Serving Over Heterogeneous Environments via Hierarchical Max-Flow

**ArXiv ID:** N/A
**Published:** N/A
**Authors:** Tao Zhang, Huihuang Qin, Dong Jin, Shuangwu Chen, Huasen He, Xiaobin Tan, Shiyin Zhu, Jian Yang
**URL:** https://www.semanticscholar.org/paper/d984f5572faa069196c15da8c0aba1c74b55412f
**PDF:** N/A
**GitHub:** 暂无
**Fields:** Computer Science

## Abstract (English)

Disaggregated LLM inference service (DLIS), which decouples the compute-intensive prefill phase and the memory-intensive decode phase, enables more flexible and efficient resource usage. Existing solutions for deploying DLIS are typically designed for homogeneous environments. However, real-world production environments are becoming increasingly heterogeneous due to GPU shortages and rapid hardware evolution. Deploying DLIS in heterogeneous environments introduces three key challenges: 1) complicated GPU resource allocation caused by significant performance differences between GPUs, 2) communication bottlenecks caused by additional key-value (KV) cache transfers, and 3) dynamic inference loads caused by time-varying arrival rates and diverse task types. Existing methods adopted uniform model partitioning on heterogeneous GPUs and single-instance partitioning, suffering from heavy straggler effect and cross-node communication overheads. Additionally, they rely on reloading model parameters for dynamic tasks, which leads to significant service interruptions. To address these issues, we propose DisHelis, a high-throughput and low-latency DLIS system for heterogeneous environments. We formulate the DLIS deployment over heterogeneous environments as a hierarchical max-flow problem. This formulation jointly incorporates non-uniform model partitioning and hybrid instance partitioning to maximize DLIS throughput. Furthermore, we design a light-weight instance-switching approach to handle dynamic tasks without service interruptions. We solve it via a hierarchical alternating optimization algorithm that iteratively converges to a high-quality deployment plan. Experimental results show that DisHelis improves throughput by up to $1.63\times $ and reduces latency by up to $2.4\times $ over existing approaches.

## 摘要 (中文)

*(待翻译)*

## Introduction (English)

*(需要阅读原文PDF补充)*

## 引言 (中文)

*(需要阅读原文PDF补充)*

## 博客内容

*(待补充)*

## GitHub 介绍

暂无 GitHub 仓库

---
*注: 此文件由自动化论文搜集系统生成于 2026-04-23，部分内容待完善。*
