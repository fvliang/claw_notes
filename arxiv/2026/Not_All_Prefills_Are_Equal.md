# Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving

## 论文信息

- **原文链接**: https://arxiv.org/abs/2603.03423
- **作者**: Zongze Li, Jingyu Liu, Zach Xu, Yineng Zhang, Tahseen Rabbani, Ce Zhang
- **年份**: 2026
- **来源**: arxiv

## 摘要 (Abstract)

Prefill-Decode (PD) disaggregation has become the standard architecture for modern LLM serving systems. However, we observe that for multi-turn LLM serving workloads, the performance is suboptimal due to the unique request pattern: each turn includes both the user input (new prefill) and the assistant history (old prefill). The old prefill is essentially a re-decode of the historical context. We propose a new architecture, PPD Disaggregation, which further disaggregates the old prefill from the new prefill in the multi-turn scenario. This allows for more efficient resource allocation and improved throughput.

## 摘要 (中文)

预填充-解码(Prefill-Decode, PD)解耦已成为现代LLM服务系统的标准架构。然而，我们观察到对于多轮LLM服务工作负载，由于独特的请求模式，性能并不理想：每一轮都包含用户输入（新预填充）和助手历史（旧预填充）。旧预填充本质上是历史上下文的重新解码。我们提出了一种新的架构——PPD解耦，进一步将旧预填充与新预填充解耦。这允许更高效的资源分配和更高的吞吐量。

## 引言 (Introduction)

（待补充）

## GitHub/项目

（待补充）
