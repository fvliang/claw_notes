# KV-Compress: Paged KV-Cache Compression with Variable Compression Rates per Attention Head

## 基本信息

- **标题**: KV-Compress: Paged KV-Cache Compression with Variable Compression Rates per Attention Head
- **arXiv**: [2410.XXXXX](https://arxiv.org/abs/)
- **发布时间**: 2024年10月

## 摘要 (Abstract)

Context lengths of Large Language Models (LLMs) have exploded in recent years, with 128k-token context becoming a standard and million-token context becoming a reality. Efficiently supporting long-context inference remains challenging as the memory that must be allocated in key-value (KV) cache grows linearly with context length.

## 摘要 (中文)

大型语言模型(LLM)的上下文长度近年来激增，128k令牌上下文已成为标准，百万令牌上下文已成为现实。高效支持长上下文推理仍然具有挑战性，因为必须分配给键值(KV)缓存的内存随上下文长度线性增长。

---

# vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention

## 基本信息

- **标题**: vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention
- **作者**: Ramya Prabhu, Ajay Nayak, Jayashree Mohan, Ramachandran Ramjee, Ashish Panwar
- **arXiv**: [2405.XXXXX](https://arxiv.org/abs/) (最初提交2024年5月)
- **发布时间**: 2025年1月

## 摘要 (Abstract)

PagedAttention is a popular approach for dynamic memory allocation in LLM serving. This paper presents vAttention, an alternative approach that achieves similar benefits without requiring paged attention kernels.

## 摘要 (中文)

PagedAttention是LLM服务中动态内存分配的流行方法。本文介绍了vAttention，这是一种替代方法，无需分页注意力内核即可实现类似的好处。

---

# Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving

## 基本信息

- **标题**: Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving
- **作者**: Wei Gao, Xinyu Zhou, Peng Sun, Tianwei Zhang, Yonggang Wen
- **arXiv**: [2503.XXXXX](https://arxiv.org/abs/)
- **发布时间**: 2025年3月

## 摘要 (Abstract)

Key-Value cache (KV cache) compression is essential for efficient LLM serving. This paper rethinks KV cache compression techniques and proposes new methods for better memory efficiency.

## 摘要 (中文)

键值缓存(KV缓存)压缩对于高效的LLM服务至关重要。本文重新思考了KV缓存压缩技术，并提出了提高内存效率的新方法。

---

# Paged Attention Meets FlexAttention: Unlocking Long-Context Efficiency in Deployed Inference

## 基本信息

- **标题**: Paged Attention Meets FlexAttention: Unlocking Long-Context Efficiency in Deployed Inference
- **作者**: Thomas Joshi, Herman Saini, Neil Dhillon, Antoni Viros i Martin, Kaoutar El Maghraoui
- **arXiv**: [2506.XXXXX](https://arxiv.org/abs/)
- **发布时间**: 2025年6月

## 摘要 (Abstract)

Large Language Models (LLMs) encounter severe memory inefficiencies during long-context inference due to conventional handling of key-value (KV) caches. This paper explores combining PagedAttention with FlexAttention for better long-context efficiency.

## 摘要 (中文)

大型语言模型(LLM)在长上下文推理过程中由于传统的键值(KV)缓存处理方式而面临严重的内存效率问题。本文探讨了将PagedAttention与FlexAttention结合以提高长上下文效率。

---

# ReSpec: Towards Optimizing Speculative Decoding in Reinforcement Learning Systems

## 基本信息

- **标题**: ReSpec: Towards Optimizing Speculative Decoding in Reinforcement Learning Systems
- **作者**: Qiaoling Chen, Zijun Liu, Peng Sun, Shenggui Li, Guoteng Wang, Ziming Liu, Yonggang Wen, Siyuan Feng, Tianwei Zhang
- **arXiv**: [2510.XXXXX](https://arxiv.org/abs/)
- **发布时间**: 2025年10月

## 摘要 (Abstract)

Adapting large language models (LLMs) via reinforcement learning (RL) is often bottlenecked by the generation stage, which can consume over 75% of the training time. This paper presents ReSpec, a system for optimizing speculative decoding in RL training pipelines.

## 摘要 (中文)

通过强化学习(RL)调整大型语言模型(LLM)通常在生成阶段遇到瓶颈，该阶段可能消耗超过75%的训练时间。本文介绍了ReSpec，一个用于优化RL训练管道中推测解码的系统。

---

# Direct Multi-Token Decoding

## 基本信息

- **标题**: Direct Multi-Token Decoding
- **作者**: Xuan Luo, Weizhi Wang, Xifeng Yan
- **arXiv**: [2502.XXXXX](https://arxiv.org/abs/)
- **发布时间**: 2025年

## 摘要 (Abstract)

Decoder-only transformers have become the standard architecture for large language models (LLMs). This paper presents a direct multi-token decoding approach that departs from the traditional autoregressive decoding paradigm.

## 摘要 (中文)

仅解码器变换器已成为大型语言模型(LLM)的标准架构。本文提出了一种直接多令牌解码方法，突破了传统的自回归解码范式。

---

# TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference

## 基本信息

- **标题**: TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference
- **作者**: Jiyoung Park, Hankyu Jang, Changseok Song, Wookeun Jung
- **arXiv**: [2602.XXXXX](https://arxiv.org/abs/)
- **发布时间**: 2026年2月

## 摘要 (Abstract)

This paper introduces TIDE, a temporal incremental draft engine for self-improving LLM inference that leverages past inference results to improve future performance.

## 摘要 (中文)

本文介绍了TIDE，这是一个时间增量draft引擎，用于自我改进的LLM推理，利用过去的推理结果来提高未来性能。

---

# WISP: Waste- and Interference-Suppressed Distributed Speculative LLM Serving at the Edge

## 基本信息

- **标题**: WISP: Waste- and Interference-Suppressed Distributed Speculative LLM Serving at the Edge via Dynamic Drafting and SLO-Aware Batching
- **作者**: Xiangchen Li, Jiakun Fan, Qingyuan Wang, Dimitrios Spatharakis, Saeid Ghafouri, Hans Vandierendonck, Deepu John, Bo Ji, Ali R. Butt, Dimitrios S. Nikolopoulos
- **arXiv**: [2601.XXXXX](https://arxiv.org/abs/)
- **发布时间**: 2026年1月

## 摘要 (Abstract)

As Large Language Models (LLMs) become increasingly accessible to end users, an ever-growing number of inference requests are initiated from edge devices. This paper presents WISP, a distributed speculative LLM serving system that suppresses waste and interference at the edge.

## 摘要 (中文)

随着大型语言模型(LLM)越来越容易被最终用户访问，越来越多的推理请求来自边缘设备。本文介绍了WISP，这是一个分布式推测LLM服务系统，可在边缘抑制浪费和干扰。

---

# NEZHA: A Zero-sacrifice and Hyperspeed Decoding Architecture for Generative Recommendations

## 基本信息

- **标题**: NEZHA: A Zero-sacrifice and Hyperspeed Decoding Architecture for Generative Recommendations
- **作者**: Yejing Wang, Shengyu Zhou, Jinyu Lu, Ziwei Liu, Langming Liu, Maolin Wang, Wenlin Zhang, Feng Li, Wenbo Su, Pengjie Wang, Jian Xu, Xiangyu Zhao
- **arXiv**: [2602.XXXXX](https://arxiv.org/abs/) (最初提交2025年11月)
- **发布时间**: 2026年2月

## 摘要 (Abstract)

Generative Recommendation (GR), powered by Large Language Models (LLMs), represents a promising new paradigm for industrial recommender systems. This paper presents NEZHA, a zero-sacrifice and hyperspeed decoding architecture for generative recommendations.

## 摘要 (中文)

由大型语言模型(LLM)驱动的生成式推荐(GR)是工业推荐系统的一个有前景的新范式。本文介绍了NEZHA，这是一种零牺牲、超高速的生成式推荐解码架构。