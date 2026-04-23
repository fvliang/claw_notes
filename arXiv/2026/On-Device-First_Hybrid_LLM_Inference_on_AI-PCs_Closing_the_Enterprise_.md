# On-Device-First Hybrid LLM Inference on AI-PCs: Closing the Enterprise GenAI Divide

## Metadata
- **Authors:** ['S. Begum', 'Kris Fleming', 'T. Lewellen']
- **Conference:** arXiv 2026
- **Topic:** KV Cache
- **arXiv ID:** 
- **Published:** 2026-02-03
- **GitHub:** kvcache-ai/Mooncake

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

AI-PCs and modern PCs equipped with capable CPUs, GPUs, and NPUs now make on-device inference for small language models (SLMs) practical across many enterprise workloads. This enables assistants that are low-latency, privacy-preserving, and resilient to network conditions, while frontier cloud models still offer higher accuracy, richer tool use, and stronger multimodality than is practical to run on every endpoint. This paper surveys and organizes the design space for an on-device-first hybrid architecture in which an SLM on the AI-PC handles most requests by default, and escalation to cloud models is reserved for difficult, tool-heavy, or long-context tasks. We synthesize advances in compact local on-device optimized model design including architectures that mix self-attention with state-space models (SSMs), parameter-efficient adaptation, multimodal and structure-aware retrieval-augmented generation (RAG), and long-horizon memory, and connect these to hardware and serving optimizations such as quantization, mixed precision, and KV-cache management. Building on this, we propose routing and reporting patterns that make latency, energy, cost, and privacy explicit design parameters, and outline a total-cost-of-ownership view that clarifies when AI-PC–centric deployments outperform all-cloud strategies. We argue that such edge-first hybrid designs offer a practical path to narrowing the "GenAI Divide," in which impressive pilots fail to translate into durable, production-grade business impact.

## 摘要 (中文)

[中文翻译待补充] AI-PCs and modern PCs equipped with capable CPUs, GPUs, and NPUs now make on-device inference for small language models (SLMs) practical across many enterprise workloads. This enables assistants that ...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

kvcache-ai/Mooncake

---
*Auto-collected on 2026-04-24*
