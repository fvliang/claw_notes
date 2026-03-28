# CacheSolidarity: Preventing Prefix Caching Side Channels in Multi-tenant LLM Serving Systems

## 论文信息

- **作者**: Panagiotis Georgios Pennas, Konstantinos Papaioannou, Marco Guarnieri, Thaleia Dimitra Doudali
- **arXiv**: [2603.10726](https://arxiv.org/abs/2603.10726)
- **提交日期**: 2026年3月11日
- **领域**: Cryptography and Security (cs.CR); Distributed, Parallel, and Cluster Computing (cs.DC); Machine Learning (cs.LG)

## 摘要 (Abstract)

Large Language Models (LLMs) rely on optimizations like Automatic Prefix Caching (APC) to accelerate inference. APC works by reusing previously computed states for the beginning part of a request (prefix), when another request starts with the same text. While APC improves throughput, it introduces timing side channels: cache hits are faster than misses, creating observable latency differences. In multi-tenant systems, attackers can exploit these differences to infer sensitive information, e.g., by incrementally reconstructing another user's request by observing hit/miss patterns. Current defenses take a sledgehammer approach: they disable APC and cache sharing, isolating users, and sacrificing efficiency for regular users. This paper presents CacheSolidarity, a system that secures multi-tenant LLM serving systems against APC side channels without sacrificing performance and efficiency. CacheSolidarity monitors cache reuse across users, flags suspicious sharing, and selectively isolates prefixes, restricting their reuse only when necessary. Evaluation shows that CacheSolidarity enables up to 70% higher cache reuse and 30% lower inference latency compared to existing defenses that isolate users. CacheSolidarity's lightweight design demonstrates how security in LLM serving does not have to come at the cost of unnecessarily reduced performance or unbearable overheads.

## 摘要 (中文)

大语言模型(LLM)依赖于自动前缀缓存(APC)等优化来加速推理。APC的工作原理是：当另一个请求以相同文本开头时，重用先前为请求开头部分（前缀）计算的状态。虽然APC提高了吞吐量，但它引入了时序侧信道：缓存命中比未命中更快，产生可观察的延迟差异。在多租户系统中，攻击者可以利用这些差异推断敏感信息，例如通过观察命中/未命中模式逐步重建另一个用户的请求。当前的防御措施采取了一种简单粗暴的方法：禁用APC和缓存共享，隔离用户，并为普通用户牺牲效率。本文提出了CacheSolidarity，一种在不牺牲性能的情况下保护多租户LLM服务系统免受APC侧信道攻击的系统。CacheSolidarity监控跨用户的缓存复用，标记可疑共享，并有选择性地隔离前缀，仅在必要时限制其复用。评估表明，与隔离用户的现有防御相比，CacheSolidarity可实现高达70%的更高缓存复用和30%的更低推理延迟。CacheSolidarity的轻量级设计表明，LLM服务中的安全性不必以不必要的性能降低或不可接受的 overhead 为代价。

## 引言 (Introduction)

自动前缀缓存(APC)是LLM推理优化的关键技术，但它引入了时序侧信道攻击风险。CacheSolidarity通过监控和有选择性地隔离来防御此类攻击，同时保持高性能。

## GitHub

暂无官方GitHub仓库。

## 博客

暂无公开博客。