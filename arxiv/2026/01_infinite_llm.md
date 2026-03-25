# Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache

## 论文信息

- **原文链接**: https://arxiv.org/abs/2401.02669
- **作者**: Bin Lin, Chen Zhang, Tao Peng, et al.
- **年份**: 2024
- **来源**: arXiv

## 摘要 (Abstract)

Large Language Models (LLMs) demonstrate substantial potential across a diverse array of domains via request serving. However, as trends continue to push for expanding context sizes, the autoregressive nature of LLMs results in highly dynamic behavior of the attention layers, showcasing significant differences in computational characteristics and memory requirements from the non-attention layers. This presents substantial challenges for resource management and performance optimization in service systems. Existing static model parallelism and resource allocation strategies fall short when dealing with this dynamicity. To address the issue, we propose Infinite-LLM, a novel LLM serving system designed to effectively handle dynamic context lengths. Infinite-LLM disaggregates attention layers from an LLM's inference process, facilitating flexible and independent resource scheduling that optimizes computational performance and enhances memory utilization jointly. By leveraging a pooled GPU memory strategy across a cluster, Infinite-LLM not only significantly boosts system throughput but also supports extensive context lengths. Evaluated on a dataset with context lengths ranging from a few to 2000K tokens across a cluster with 32 A100 GPUs, Infinite-LLM demonstrates throughput improvement of 1.35-3.4x compared to state-of-the-art methods, enabling efficient and elastic LLM deployment.

## 摘要 (中文)

大型语言模型（LLM）在各个领域通过请求服务展示了巨大的潜力。然而，随着上下文长度不断扩展的趋势，LLM的自回归特性导致注意力层表现出高度动态的行为，与非注意力层在计算特性和内存需求方面存在显著差异。这给服务系统的资源管理和性能优化带来了巨大挑战。现有静态模型并行和资源分配策略在处理这种动态性方面存在不足。为解决这个问题，我们提出了Infinite-LLM，一种专门为有效处理动态上下文长度而设计的新型LLM服务系统。Infinite-LLM将注意力层从LLM的推理过程中解耦，实现灵活独立的资源调度，共同优化计算性能和内存利用率。通过利用集群中的GPU内存池化策略，Infinite-LLM不仅显著提升了系统吞吐量，还支持更长的上下文长度。在包含32个A100 GPU的集群上，针对从几千到2000K tokens不同上下文长度的数据集进行评估，Infinite-LLM相比最先进的方法实现了1.35-3.4倍的吞吐量提升，实现了高效且弹性的LLM部署。

## 引言 (Introduction)

LLM的推理服务面临的主要挑战包括：
1. 注意力层的动态行为与静态并行策略不匹配
2. 长上下文场景下KV-Cache内存开销巨大
3. 现有系统难以处理动态变化的上下文长度

Infinite-LLM提出解耦注意力层的分布式KV-Cache管理，通过GPU内存池化来提升吞吐量和支持更长上下文。

## GitHub/项目

（待补充）