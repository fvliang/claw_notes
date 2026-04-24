# Efficient LLM Inference via Chunked Prefills

## Metadata
- **Authors:** Arney Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra
- **Conference:** arXiv 2025
- **Topic:** Prefill/Disaggregation
- **arXiv ID:** 
- **Published:** 2025-08-04
- **GitHub:** SharpAI/SwiftLM

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

Large Language Model (LLM) inference serving faces a fundamental challenge due to the distinct characteristics of its two phases: compute-intensive pre fill and memory-intensive decode. Existing scheduling strategies often prioritize one phase over the other, leading to a difficult tradeoff between system throughput and request latency. Prefill-prioritizing schedulers improve throughput but introduce significant latency jitter (generation stalls) by interfering with ongoing decodes. Conversely, decode-prioritizing schedulers maintain low latency but underutilize GPU resources, resulting in low throughput. This paper revisits the technique of chunked prefills, demonstrating its efficacy in mitigating this tradeoff. By splitting large prefill computations into smaller, manageable chunks and interleaving them with decode operations using stall-free batching, we can leverage the compute slack inherent in the decode phase. This approach significantly improves serving capacity under strict latency constraints, minimizes generation stalls, and reduces pipeline bubbles in distributed deployments, enabling efficient and responsive inference.

## 摘要 (中文)

[中文翻译待补充] Large Language Model (LLM) inference serving faces a fundamental challenge due to the distinct characteristics of its two phases: compute-intensive pre fill and memory-intensive decode. Existing sched...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

SharpAI/SwiftLM

---
*Auto-collected on 2026-04-25*
