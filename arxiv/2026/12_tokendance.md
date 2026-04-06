# TokenDance: Scaling Multi-Agent LLM Serving via Collective KV Cache Sharing

- **arXiv**: [2604.03143](https://arxiv.org/abs/2604.03143)
- **Authors**: Zhuohang Bian, Feiyang Wu, Chengrui Zhang, Hangcheng Dong, Yun Liang, Youwei Zhuo
- **Year**: 2026
- **Conference**: arXiv
- **Topic**: LLM Serving

## Abstract (English)

Multi-agent LLM applications organize execution in synchronized rounds where a central scheduler gathers outputs from all agents and redistributes the combined context. This All-Gather communication pattern creates massive KV Cache redundancy, because every agent's prompt contains the same shared output blocks, yet existing reuse methods fail to exploit it efficiently. We present TokenDance, a system that scales the number of concurrent agents by exploiting the All-Gather pattern for collective KV Cache sharing. TokenDance's KV Collector performs KV Cache reuse over the full round in one collective step, so the cost of reusing a shared block is paid once regardless of agent count. Its Diff-Aware Storage encodes sibling caches as block-sparse diffs against a single master copy, achieving 11-17x compression on representative workloads. Evaluation on GenerativeAgents and AgentSociety shows that TokenDance supports up to 2.7x more concurrent agents than vLLM with prefix caching under SLO requirement, reduces per-agent KV Cache storage by up to 17.5x, and achieves up to 1.9x prefill speedup over per-request position-independent caching.

## Abstract (中文)

多智能体LLM应用以同步轮次组织执行，中心调度器收集所有智能体的输出并重新分配组合上下文。这种All-Gather通信模式产生大量KV Cache冗余，因为每个智能体的提示都包含相同的共享输出块，但现有重用方法无法有效利用这一特性。我们提出TokenDance，一个通过利用All-Gather模式进行集体KV Cache共享来扩展并发智能体数量的系统。TokenDance的KV Collector在一轮中执行一次集体KV Cache共享，因此无论智能体数量多少，重用共享块的成本只支付一次。其Diff-Aware Storage将兄弟缓存编码为相对于单个主副本的块稀疏差分，在代表性工作负载上实现11-17倍压缩。在GenerativeAgents和AgentSociety上的评估表明，TokenDance在SLO要求下支持比vLLM前缀缓存多2.7倍的并发智能体，减少每个智能体KV Cache存储最多17.5倍，并实现比每请求位置无关缓存最高1.9倍的预填充加速。

## Introduction (English)

Multi-agent LLM systems have emerged as a powerful paradigm for complex task solving, where multiple LLM agents collaborate to handle sophisticated workflows. These systems typically organize execution in synchronized rounds, where agents process their assigned tasks and share results through a central scheduler. However, this architectural pattern introduces significant KV Cache redundancy, as all agents' prompts contain common system prompts, shared context, and previously generated content.

## Introduction (中文)

多智能体LLM系统已成为复杂任务解决的强大范式，多个LLM智能体协作处理复杂工作流。这些系统通常以同步轮次组织执行，智能体处理分配的任务并通过中心调度器共享结果。然而，这种架构模式引入了显著的KV Cache冗余，因为所有智能体的提示都包含常见的系统提示、共享上下文和先前生成的内容。

## GitHub

(None found)

## Blog

(None found)