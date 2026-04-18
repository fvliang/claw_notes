# ForkKV: Scaling Multi-LoRA Agent Serving via Copy-on-Write Disaggregated KV Cache

**arXiv**: 2604.06370
**链接**: https://arxiv.org/abs/2604.06370
**作者**: Shao Wang, Rui Ren, Lin Gui
**会议**: arXiv 2026
**主题**: llm_serving / Multi-LoRA Serving / KV Cache

## 摘要 (Abstract)

The serving paradigm of large language models (LLMs) is rapidly shifting towards complex multi-agent workflows where specialized agents collaborate over massive shared contexts. While Low-Rank Adaptation (LoRA) enables the efficient co-hosting of these specialized agents on a single base model, it introduces a critical memory footprint bottleneck during serving. Specifically, unique LoRA activations cause Key-Value (KV) cache divergence across agents, rendering traditional prefix caching ineffective for shared contexts. This forces redundant KV cache maintenance, rapidly saturating GPU capacity and degrading throughput. To address this challenge, we introduce ForkKV, a serving system for multi-LoRA agent workflows centered around a novel memory management paradigm in OS: fork with copy-on-write (CoW). By exploiting the structural properties of LoRA, ForkKV physically decouples the KV cache into a massive shared component (analogous to the parent process's memory pages) and lightweight agent-specific components (the child process's pages). To support this mechanism, we propose a DualRadixTree architecture that allows newly forked agents to inherit the massive shared cache and apply CoW semantics for their lightweight unique cache. Furthermore, to guarantee efficient execution, we design ResidualAttention, a specialized kernel that reconstructs the disaggregated KV cache directly within on-chip SRAM. Comprehensive evaluations across diverse language models and practical datasets demonstrate that ForkKV achieves up to 3.0x the throughput of state-of-the-art multi-LoRA serving systems with a negligible impact on generation quality.

## 摘要 (中文)

LLM 的服务范式正快速转向复杂的多 agent 工作流，其中专业化 agent 在大规模共享上下文上协作。虽然 LoRA 使专业化 agent 能高效共存于单一基础模型上，但在服务时引入了严重的内存瓶颈。具体而言，不同的 LoRA 激活导致各 agent 的 KV cache 分化，使传统前缀缓存对共享上下文失效，迫使冗余 KV cache 维护，快速耗尽 GPU 容量并降低吞吐量。本文提出 ForkKV，以操作系统中 fork + copy-on-write (CoW) 的内存管理范式为核心的多 LoRA agent 工作流服务系统。利用 LoRA 的结构特性，ForkKV 将 KV cache 物理解耦为大规模共享组件（类似父进程内存页）和轻量级 agent 特定组件（子进程页）。为此提出 DualRadixTree 架构，允许新 fork 的 agent 继承大规模共享缓存并对轻量级唯一缓存应用 CoW语义。还设计了 ResidualAttention 专用内核，直接在片上 SRAM 中重建解耦的 KV cache。综合评估表明，ForkKV 相比最先进的多 LoRA 服务系统吞吐量提升达 3.0x，生成质量影响可忽略。

## 关键贡献

1. 借鉴 OS fork + CoW 的内存管理范式应用于 KV cache
2. DualRadixTree 架构支持共享缓存继承与 CoW 语义
3. ResidualAttention 内核在 SRAM 中重建解耦 KV cache
4. 吞吐量提升 3.0x，质量影响可忽略