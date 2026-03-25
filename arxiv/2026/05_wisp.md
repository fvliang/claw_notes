# WISP: Waste- and Interference-Suppressed Distributed Speculative LLM Serving at the Edge

## 论文信息

- **原文链接**: https://arxiv.org/abs/2601.01839
- **作者**: Xiangchen Li, Jiakun Fan, Qingyuan Wang, et al.
- **年份**: 2026
- **来源**: arXiv

## 摘要 (Abstract)

As Large Language Models (LLMs) become increasingly accessible to end users, an ever-growing number of inference requests are initiated from edge devices and computed on centralized GPU clusters. However, the resulting exponential growth in computation workload is placing significant strain on data centers, while edge devices remain largely underutilized, leaving substantial computational resources idle. We present WISP, a waste- and interference-suppressed distributed speculative LLM serving system that coordinates edge and cloud resources for efficient inference. WISP introduces a dynamic drafting strategy that adapts to varying network conditions and request characteristics, while suppressing both computation waste and interference between concurrent requests. Our system achieves 2.3x higher throughput compared to state-of-the-art edge-cloud LLM serving systems.

## 摘要 (中文)

随着大型语言模型（LLM）越来越容易为终端用户使用，越来越多的推理请求从边缘设备发起，在集中式GPU集群上计算。然而，计算工作负载的指数级增长给数据中心带来了巨大压力，而边缘设备却基本未被充分利用，留下大量闲置的计算资源。我们提出了WISP，一个 waste- 和 interference-suppressed（浪费和干扰抑制）分布式投机LLM服务系统，协调边缘和云资源进行高效推理。WISP引入了一种动态draft策略，可以适应不同的网络条件和请求特征，同时抑制计算浪费和并发请求之间的干扰。我们的系统与最先进的边缘云LLM服务系统相比，实现了2.3倍更高的吞吐量。

## 引言 (Introduction)

边缘云LLM服务面临的挑战：
1. 数据中心计算压力巨大
2. 边缘设备资源未充分利用
3. 网络延迟和带宽限制
4. 并发请求之间的干扰

WISP的核心创新：
- 动态draft策略适应网络条件
- 浪费和干扰抑制机制
- 边缘云资源协调
- 2.3倍吞吐量提升

## GitHub/项目

（待补充）