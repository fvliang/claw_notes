# vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention

## 原文链接
- arXiv: https://arxiv.org/abs/2309.06180
- GitHub: https://github.com/vllm-project/vllm
- 论文PDF: https://arxiv.org/pdf/2309.06180

## 会议
SOSP 2023

## 作者
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica

## 摘要 (Abstract)
High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time. However, existing systems struggle because the key-value cache (KV cache) memory for each request is huge and grows and shrinks dynamically. When managed inefficiently, this memory can be significantly wasted by fragmentation and redundant duplication, limiting the batch size. To address this problem, we propose PagedAttention, an attention algorithm inspired by the classical virtual memory and paging techniques in operating systems. On top of it, we build vLLM, an LLM serving system that achieves (1) near-zero waste in KV cache memory and (2) flexible sharing of KV cache within and across requests to further reduce memory usage. Our evaluations show that vLLM improves the throughput of popular LLMs by 2-4× with the same level of latency compared to the state-of-the-art systems, such as FasterTransformer and Orca. The improvement is more pronounced with longer sequences, larger models, and more complex decoding algorithms.

## 摘要 (中文)
大型语言模型（LLM）的高吞吐量服务需要同时批处理足够多的请求。然而，现有系统面临挑战，因为每个请求的键值缓存（KV缓存）内存很大，且会动态增长和收缩。当管理不高效时，这些内存会因碎片化和冗余复制而严重浪费，限制了批处理大小。为解决这一问题，我们提出了PagedAttention，这是一种受操作系统中经典虚拟内存和分页技术启发的注意力算法。在此基础上，我们构建了vLLM，一个LLM服务系统，实现了（1）KV缓存内存近零浪费，（2）灵活共享KV缓存在请求内部和跨请求以进一步减少内存使用。我们的评估表明，vLLM与FasterTransformer和Orca等最新系统相比，在相同延迟水平下将流行LLM的吞吐量提高了2-4倍。这种改进在更长序列、更大模型和更复杂解码算法的情况下更为明显。

## 引言 (Introduction)
Large language models (LLMs) have become the backbone of many modern AI applications. Serving LLMs at scale requires high throughput to serve many concurrent users. A common technique to improve throughput is to batch multiple requests together and process them simultaneously. However, the memory consumption of LLM serving is dominated by the key-value (KV) cache, which stores intermediate activation for the attention mechanism. The KV cache is large and grows and shrinks dynamically as the model generates new tokens. Efficient management of the KV cache is critical for maximizing batch size and thus throughput.

Existing LLM serving systems manage the KV cache using static allocation. They pre-allocate a contiguous chunk of memory for each request at the beginning of inference and retain the memory until the request completes. This approach leads to two problems: (1) internal fragmentation - the pre-allocated memory may be larger than what the request actually needs; (2) reserved memory cannot be shared across requests even when they have common prefixes.

To address these problems, we propose PagedAttention, a new attention algorithm that allows the KV cache to be stored in non-contiguous memory blocks. Inspired by virtual memory in operating systems, PagedAttention stores each attention query's keys and values in a block that can be located anywhere in memory. This allows dynamic memory allocation without reservation and enables flexible sharing of the KV cache.

基于此，我们构建了vLLM，这是一个支持PagedAttention的LLM服务系统。vLLM采用类似操作系统的分页系统来管理KV缓存，实现了近零浪费的内存管理。我们实现了一个新的内存分配器，专门设计用于LLM推理的解码模式，它按需分配分页，并有效地整合已完成的请求释放的页面。

## GitHub 介绍
vLLM 是一个快速且易用的 LLM 推理和服务库。

**主要特性：**
- 先进的服务吞吐量
- 使用 PagedAttention 高效管理注意力键值内存
- 连续批处理 incoming 请求
- 使用 CUDA/HIP graph 快速模型执行
- 量化支持：GPTQ, AWQ, AutoRound, INT4, INT8, FP8
- 优化的 CUDA 内核，包括与 FlashAttention 和 FlashInfer 的集成
- Speculative decoding
- Chunked prefill
- 与流行 Hugging Face 模型的无缝集成
- 高吞吐量服务，支持各种解码算法
- Tensor、pipeline、data 和 expert parallelism 支持分布式推理
- 流式输出
- OpenAI 兼容的 API 服务器
- 支持 NVIDIA GPU、AMD CPU 和 GPU、Intel CPU 和 GPU、PowerPC CPU、Arm CPU 和 TPU
- Prefix caching 支持
- Multi-LoRA 支持

**官方文档：** https://docs.vllm.ai
**官方博客：** https://blog.vllm.ai/

---

*本文档由自动化任务生成于 2026-03-24*