# vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention

## 基本信息

- **标题**: Efficient Memory Management for Large Language Model Serving with PagedAttention
- **作者**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
- **会议**: SOSP 2023
- **arXiv**: [2309.06180](https://arxiv.org/abs/2309.06180)
- **GitHub**: [vllm-project/vllm](https://github.com/vllm-project/vllm)
- **发布时间**: 2023年9月

## 摘要 (Abstract)

High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time. However, existing systems struggle because the key-value cache (KV cache) memory for each request is huge and grows and shrinks dynamically. When managed inefficiently, this memory can be significantly wasted by fragmentation and redundant duplication, limiting the batch size. To address this problem, we propose PagedAttention, an attention algorithm inspired by the classical virtual memory and paging techniques in operating systems. On top of it, we build vLLM, an LLM serving system that achieves (1) near-zero waste in KV cache memory and (2) flexible sharing of KV cache within and across requests to further reduce memory usage. Our evaluations show that vLLM improves the throughput of popular LLMs by 2-4× with the same level of latency compared to the state-of-the-art systems, such as FasterTransformer and Orca. The improvement is more pronounced with longer sequences, larger models, and more complex decoding algorithms.

## 摘要 (中文)

大型语言模型(LLM)的高吞吐量服务需要同时批处理足够多的请求。然而，现有的系统表现不佳，因为每个请求的键值缓存(KV缓存)内存很大，并且会动态增长和收缩。当管理效率低下时，这些内存会被碎片化和冗余复制所严重浪费，从而限制批处理大小。为了解决这个问题，我们提出了PagedAttention，这是一种受操作系统中经典虚拟内存和分页技术启发的注意力算法。在此基础上，我们构建了vLLM，一个LLM服务系统，实现了(1)KV缓存内存近乎零浪费，(2)在请求内部和跨请求之间灵活共享KV缓存以进一步减少内存使用。我们的评估表明，vLLM在相同延迟水平下，将流行LLM的吞吐量提高了2-4倍 compared to the state-of-the-art systems, such as FasterTransformer and Orca。这种改进在更长的序列、更大的模型和更复杂的解码算法下更为明显。

## 引言 (Introduction)

Large language models (LLMs) have revolutionized natural language processing tasks, enabling a wide range of applications such as chatbots, code completion, and content generation. However, deploying LLMs for online serving remains challenging due to their large memory footprint and computational requirements.

The autoregressive decoding process of LLMs generates tokens one by one, where each token depends on all previous tokens stored in the key-value (KV) cache. The KV cache can become very large, especially for long sequences. Existing LLM serving systems manage the KV cache using contiguous memory allocation, similar to how traditional deep learning frameworks manage tensors. This approach leads to significant memory waste due to fragmentation and pre-reservation.

We address this problem by introducing PagedAttention, which borrows the concept of paging from operating systems. Instead of allocating contiguous memory blocks for the KV cache, PagedAttention stores the KV cache in non-contiguous physical pages that can be dynamically allocated and freed. This approach enables near-zero waste in KV cache memory and flexible sharing of KV cache within and across requests.

## 引言 (中文)

大型语言模型(LLM)已经革新了自然语言处理任务，使聊天机器人、代码补全和内容生成等广泛的应用成为可能。然而，由于LLM占用内存大、计算要求高，部署LLM进行在线服务仍然具有挑战性。

LLM的自回归解码过程逐个生成令牌，每个令牌依赖于存储在键值(KV)缓存中的所有先前令牌。KV缓存可能变得非常大，特别是对于长序列。现有的LLM服务系统使用连续内存分配来管理KV缓存，类似于传统深度学习框架管理张量的方式。这种方法会导致显著的内存浪费，因为碎片化和预分配。

我们通过引入PagedAttention来解决这个问题，该算法借鉴了操作系统的分页概念。PagedAttention不是为KV缓存分配连续的内存块，而是将KV缓存存储在可以动态分配和释放的非连续物理页面中。这种方法实现了KV缓存内存近乎零浪费，并在请求内部和跨请求之间灵活共享KV缓存。

## GitHub 仓库介绍

vLLM 是一个快速且易于使用的LLM推理服务库。

### 主要特性
- **PagedAttention**: 受操作系统分页启发的注意力算法
- **连续批处理**: 高效的请求批处理
- **CUDA内核优化**: 高性能的CUDA实现
- **OpenAI兼容API**: 与OpenAI API兼容

### 性能提升
- 吞吐量提升2-4倍
- 延迟更低
- 内存使用更高效