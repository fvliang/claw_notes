# Towards Efficient LLM Inference via Collective and Adaptive Speculative Decoding

## Metadata
- **Authors:** ['Siqi Wang', 'Hailong Yang', 'Xuezhu Wang', 'Tongxuan Liu', 'Pengbo Wang', 'Yufan Xu', 'Xuning Liang', 'Kejie Ma', 'Tianyu Feng', 'Xin You', 'Ruihao Gong', 'Rui Wang', 'Zhongzhi Luan', 'Yi Liu', 'Depei Qian']
- **Conference:** arXiv 2025
- **Topic:** Speculative Decoding
- **arXiv ID:** 
- **Published:** 2025-11-15
- **GitHub:** kvcache-ai/Mooncake

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

Large language models (LLMs) have gained considerable attention for their remarkable performance across a wide range of tasks. However, efficient LLM inference remains challenging because of the autoregressive decoding process, which generates only one token at a time. Speculative decoding has been introduced to address the limitation by using small speculative models (SSMs) to speed up LLM inference. However, the low acceptance rate of SSMs and the high verification cost of LLM prohibit further performance improvement. In this paper, we present Smurfs, an LLM inference system designed to accelerate LLM inference through collective and adaptive speculative decoding. Smurfs adopts a majority-voted mechanism that harnesses multiple SSMs to collaboratively predict LLM outputs in multi-task scenarios, while avoiding high verification cost. It also decouples SSM speculation from LLM verification and uses a pipelined execution to hide the latency of SSM speculation. Additionally, Smurfs proposes a mechanism to dynamically determine the optimal speculation length of SSM at runtime, balancing the performance impact of accepted tokens and verification cost. The experimental results demonstrate the superiority of Smurfs in terms of inference throughput and latency compared to the state-of-the-art LLM inference systems.

## 摘要 (中文)

[中文翻译待补充] Large language models (LLMs) have gained considerable attention for their remarkable performance across a wide range of tasks. However, efficient LLM inference remains challenging because of the autor...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

kvcache-ai/Mooncake

---
*Auto-collected on 2026-04-24*
