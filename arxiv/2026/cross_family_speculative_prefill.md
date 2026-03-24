# Cross-Family Speculative Prefill: Training-Free Long-Context Compression with Small Draft Models

## 论文信息

- **作者**: Shubhangi Upasani, Ravi Shanker Raju, Bo Li, Mengmeng Ji, John Long, Chen Wu, Urmish Thakker, Guangtao Wang
- **提交日期**: 2026年3月12日
- **arXiv链接**: https://arxiv.org/abs/xxx (待补充)
- **关键词**: Speculative Decoding, Prefill Optimization, Long Context, Cross-Family

## 摘要 (Abstract)

Prompt length is a major bottleneck in agentic large language model (LLM) workloads, where repeated inference steps and multi-call loops incur substantial prefill cost. Recent work on speculative decoding has primarily focused on the decoding phase, neglecting the prefill stage which dominates latency in long-context scenarios.

在智能体LLM工作负载中，提示长度是一个主要瓶颈，重复推理步骤和多轮调用会产生大量prefill成本。最近的投机解码工作主要集中于解码阶段，忽略了在长上下文场景中占据主导地位的prefill阶段。

## 引言 (Introduction)

### 背景
- Agentic工作负载需要长上下文处理
- Prefill阶段计算成本高
- 现有的speculative decoding主要优化decode阶段

### 本文贡献
1. 提出Cross-Family Speculative Prefill方法
2. 利用小型draft模型进行prefill加速
3. 训练-free的即插即用方案

## 方法

### 核心思想
使用小型draft模型生成候选token，然后由主模型验证，减少prefill阶段的计算量。

### 技术细节
- 跨模型族兼容性
- 动态draft长度调整
- 验证与接受策略

---

*更新时间: 2026-03-24*