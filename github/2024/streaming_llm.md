# Streaming LLM: Efficient Streaming Language Models with Attention Sinks

## 基本信息

- **仓库**: [mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm)
- **描述**: Efficient Streaming Language Models with Attention Sinks
- **语言**: Python
- **Stars**: 7207
- **更新时间**: 2026

## 主要特性

- **流式推理**: 支持无限长度的流式语言模型推理
- **注意力汇(Attention Sink)**: 通过虚拟token保持注意力机制稳定
- **无限上下文**: 无需额外训练即可处理无限长度的序列
- **开源实现**: 提供完整的PyTorch实现

## 原文链接

- GitHub: https://github.com/mit-han-lab/streaming-llm

## 介绍

Streaming LLM是MIT Han Lab推出的高效流式语言模型推理框架。该方法通过引入"注意力汇"(Attention Sink)的概念,解决了流式推理中长期依赖保持的问题。研究表明,只需添加少量可学习的注意力汇token,就可以让LLM在不影响性能的情况下处理无限长度的序列。

## 相关论文

- 原始论文描述了注意力汇机制的理论基础
- 该技术已被集成到多个主流推理框架中

---