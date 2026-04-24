# EdgeShard: Efficient LLM Inference via Collaborative Edge Computing

## Metadata
- **Authors:** Mingjin Zhang, Xiaoming Shen, Jiannong Cao, Zeyang Cui, Shan Jiang
- **Conference:** arXiv 2025
- **Topic:** Quantization
- **arXiv ID:** 
- **Published:** 2025-05-15
- **GitHub:** skyzh/tiny-llm

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

Large language models (LLMs) have shown great success in content generation and intelligent intelligent decision making for IoT systems. Traditionally, LLMs are deployed on the cloud, incurring prolonged latency, high bandwidth costs, and privacy concerns. More recently, edge computing has been considered promising in addressing such concerns because the edge devices are closer to data sources. However, edge devices are cursed by their limited resources and can hardly afford LLMs. Existing studies address such a limitation by offloading heavy workloads from edge to cloud or compressing LLMs via model quantization. These methods either still rely heavily on the remote cloud or suffer substantial accuracy loss. This work is the first to deploy LLMs on a collaborative edge computing environment, in which edge devices and cloud servers share resources and collaborate to infer LLMs with high efficiency and no accuracy loss. We design EdgeShard, a novel approach to partition a computation-intensive LLM into affordable shards and deploy them on distributed devices. The partition and distribution are nontrivial, considering device heterogeneity, bandwidth limitations, and model complexity. To this end, we formulate an adaptive joint device selection and model partition problem and design an efficient dynamic programming algorithm to optimize the inference latency and throughput. Extensive experiments of the popular Llama2 serial models on a real-world testbed reveal that EdgeShard achieves up to 50% latency reduction and $2 \times $ throughput improvement over the state-of-the-art.

## 摘要 (中文)

[中文翻译待补充] Large language models (LLMs) have shown great success in content generation and intelligent intelligent decision making for IoT systems. Traditionally, LLMs are deployed on the cloud, incurring prolon...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

skyzh/tiny-llm

---
*Auto-collected on 2026-04-24 evening*
