# DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads

## 基本信息

- **标题**: DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads
- **作者**: (待补充)
- **arXiv**: (待补充)
- **会议**: ICLR 2025
- **GitHub**: [mit-han-lab/duo-attention](https://github.com/mit-han-lab/duo-attention)
- **Stars**: 534

## 摘要 (Abstract)

DuoAttention introduces a dual-head attention mechanism that separates retrieval and streaming workloads, enabling efficient long-context LLM inference with reduced memory and compute requirements.

## 摘要 (中文)

DuoAttention引入了一种双头注意力机制,将检索和流式工作负载分离,能够在减少内存和计算需求的情况下实现高效的长上下文LLM推理。

## 引言 (Introduction)

长上下文LLM需要同时处理不同类型的注意力模式:检索任务需要全局注意力,而流式生成主要需要局部注意力。DuoAttention通过引入两个专门的注意力头(检索头和流式头)来分别处理这两种模式,从而优化整体效率。

## 原文链接

- arXiv: (待补充)
- GitHub: https://github.com/mit-han-lab/duo-attention

---