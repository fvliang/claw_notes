# Nightjar: Dynamic Adaptive Speculative Decoding for Large Language Models Serving

## 论文信息

- **标题**: Nightjar: Dynamic Adaptive Speculative Decoding for Large Language Models Serving
- **作者**: Rui Li, Zhaoning Zhang, Libo Zhang, Huaimin Wang, Xiang Fu, Zhiquan Lai
- **arXiv**: [arXiv:2512.22420](https://arxiv.org/abs/2512.22420)
- **提交日期**: 2025年12月 (v1), 2026年3月 (v4)
- **领域**: Distributed, Parallel, and Cluster Computing (cs.DC)
- **会议**: arXiv预印本

## 摘要 (Abstract)

Speculative decoding (SD) accelerates LLM inference by verifying draft tokens in parallel. However, this method presents a critical trade-off: it improves throughput in low-load, memory-bound systems but degrades performance in high-load, compute-bound environments due to verification overhead. Existing speculative decoding methods use fixed lengths and cannot adapt to workload changes or decide when to stop speculation. The cost of restarting speculative inference also remains unquantified. Under high load, the benefit of speculation diminishes, while retaining the draft model reduces KV-cache capacity, limiting batch size and degrading throughput. To overcome this, we propose Nightjar, a resource-aware adaptive speculative framework. It first adjusts to the request load by dynamically selecting the optimal speculative length for different batch sizes. Crucially, Nightjar proactively disables speculative decoding when the MAB planner determines that speculation is no longer beneficial, and during the disabled phase, offloads the draft model to the CPU only under GPU memory pressure. This reclaims memory for the KV cache, thereby facilitating larger batch sizes and maximizing overall system throughput. Experiments show that Nightjar achieves average 27.29% higher throughput and up to 20.18% lower latency compared to standard speculative decoding under dynamic request arrival rates in real-time LLM serving scenarios.

## 摘要 (中文)

投机解码(SD)通过并行验证草稿标记来加速LLM推理。然而，这种方法存在一个关键的权衡：它在低负载、内存受限的系统中提高吞吐量，但在高负载、计算受限的环境中由于验证开销而降低性能。现有的投机解码方法使用固定长度，无法适应工作负载变化或决定何时停止投机。重启投机推理的成本也未被量化。在高负载下，投机的好处减少，同时保留草稿模型会减少KV缓存容量，限制批处理大小并降低吞吐量。为了克服这个问题，我们提出了Nightjar，一个资源感知的自适应投机框架。它首先通过为不同的批处理大小动态选择最佳投机长度来适应请求负载。关键的是，当MAB规划器确定投机不再有益时，Nightjar会主动禁用投机解码，并在禁用阶段仅在GPU内存压力下将草稿模型卸载到CPU。这回收了KV缓存的内存，从而实现更大的批处理大小并最大化系统吞吐量。实验表明，在实时LLM服务的动态请求到达场景下，Nightjar相比标准投机解码平均提高了27.29%的吞吐量，最多降低了20.18%的延迟。

## 引言 (Introduction)

大型语言模型(LLM)的自回归解码过程是推理延迟的主要瓶颈。投机解码通过使用一个较小的草稿模型并行生成多个候选标记，然后由主模型验证，从而加速这一过程。然而，现有的投机解码方法存在以下问题：

1. **固定投机长度**：无法根据动态工作负载调整
2. **高负载下性能下降**：验证开销在计算密集型场景下成为瓶颈
3. **内存占用**：保留草稿模型占用KV缓存空间，限制批处理大小
4. **重启成本**：投机推理重启的成本未被考虑

Nightjar通过以下创新解决这些问题：
- **自适应投机长度**：根据请求负载动态调整
- **智能禁用机制**：当投机不再有益时主动禁用
- **草稿模型卸载**：释放GPU内存用于KV缓存

## GitHub

暂无公开GitHub仓库