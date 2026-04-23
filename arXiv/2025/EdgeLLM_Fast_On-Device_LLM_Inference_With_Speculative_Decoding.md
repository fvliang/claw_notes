# EdgeLLM: Fast On-Device LLM Inference With Speculative Decoding

## Metadata
- **Authors:** ['Daliang Xu', 'Wangsong Yin', 'Hao Zhang', 'Xin Jin', 'Ying Zhang', 'Shiyun Wei', 'Mengwei Xu', 'Xuanzhe Liu']
- **Conference:** arXiv 2025
- **Topic:** Speculative Decoding
- **arXiv ID:** 
- **Published:** 2025-04-01
- **GitHub:** asprenger/ray_vllm_inference

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

Generative tasks, such as text generation and question answering, are essential for mobile applications. Given their inherent privacy sensitivity, executing them on devices is demanded. Nowadays, the execution of these generative tasks heavily relies on the Large Language Models (LLMs). However, the scarce device memory severely hinders the scalability of these models. We present EdgeLLM, an efficient on-device LLM inference system for models whose sizes exceed the device's memory capacity. EdgeLLM is built atop speculative decoding, which delegates most tokens to a smaller, memory-resident (draft) LLM. EdgeLLM integrates three novel techniques: (1) Instead of generating a fixed width and depth token tree, EdgeLLM proposes compute-efficient branch navigation and verification to pace the progress of different branches according to their accepted probability to prevent the wasteful allocation of computing resources to the wrong branch and to verify them all at once efficiently. (2) It uses a self-adaptive fallback strategy that promptly initiates the verification process when the smaller LLM generates an incorrect token. (3) To not block the generation, EdgeLLM proposes speculatively generating tokens during large LLM verification with the compute-IO pipeline. Through extensive experiments, EdgeLLM exhibits impressive token generation speed which is up to 9.3× faster than existing engines.

## 摘要 (中文)

[中文翻译待补充] Generative tasks, such as text generation and question answering, are essential for mobile applications. Given their inherent privacy sensitivity, executing them on devices is demanded. Nowadays, the ...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

asprenger/ray_vllm_inference

---
*Auto-collected on 2026-04-24*
