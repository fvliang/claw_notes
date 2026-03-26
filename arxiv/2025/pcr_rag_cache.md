# PCR: A Prefetch-Enhanced Cache Reuse System for Low-Latency RAG Serving

## 论文信息

- **arXiv**: https://arxiv.org/abs/2503.XXXXX
- **作者**: Wenfeng Wang, Xiaofeng Hou, Peng Tang, Hengyi Zhou, Jing Wang, Xinkai Wang, Chao Li, Minyi Guo
- **提交时间**: 2025年3月24日

## 摘要

检索增强生成 (Retrieval-Augmented Generation, RAG) 系统通过整合检索到的外部文档来增强大语言模型 (LLMs) 的性能，从而实现更准确和上下文感知的响应。

然而，集成这些外部文档通常会导致**非常长的输入序列**，这显著增加了预填充 (prefill) 阶段的计算成本。

## 核心问题

- RAG系统中长输入序列带来的高计算成本
- Prefill阶段成为延迟瓶颈
- KV缓存效率低下

## 核心贡献

**PCR (Prefetch-Enhanced Cache Reuse)** 是一个**低延迟RAG服务系统的优化方案**：

1. **预取增强**: 智能预取相关文档
2. **缓存复用**: 重复利用已计算的KV缓存
3. **延迟优化**: 显著降低RAG系统的响应延迟

## 技术特点

- 文档级别的缓存管理
- 智能预取策略
- 自适应缓存淘汰

## 相关工作

与以下技术相关：
- RAG系统优化
- KV缓存管理
- 长上下文LLM推理