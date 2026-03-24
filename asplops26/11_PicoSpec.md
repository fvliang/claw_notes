# A Pipelined Collaborative Speculative Decoding Framework for Efficient Edge-Cloud LLM Inference

**论文链接**: [arXiv:2603.19133](https://arxiv.org/abs/2603.19133)

**作者**: Yida Zhang, Zhiyong Gao, Shuaibing Yue, Jie Li, Rui Wang

**会议**: ASPLOS 2026 (可能)

**提交日期**: 2026年3月19日

---

## Abstract (摘要)

Recent advancements and widespread adoption of Large Language Models (LLMs) in both industry and academia have catalyzed significant demand for LLM serving. However, traditional cloud services incur high costs, while on-device inference alone faces challenges due to limited resources. Edge-cloud collaboration emerges as a key research direction to combine the strengths of both paradigms, yet efficiently utilizing limited network bandwidth while fully leveraging and balancing the computational capabilities of edge devices and the cloud remains an open problem. To address these challenges, we propose Pipelined Collaborative Speculative Decoding Framework (PicoSpec), a novel, general-purpose, and training-free speculative decoding framework for LLM edge-cloud collaborative inference. We design an asynchronous pipeline that resolves the mutual waiting problem inherent in vanilla speculative decoding within edge collaboration scenarios, which concurrently executes a Small Language Model (SLM) on the edge device and a LLM in the cloud. Meanwhile, to mitigate the significant communication latency caused by transmitting vocabulary distributions, we introduce separate rejection sampling with sparse compression, which completes the rejection sampling with only a one-time cost of transmitting the compressed vocabulary. Experimental results demonstrate that our solution outperforms baseline and existing methods, achieving up to 2.9 speedup.

---

大型语言模型（LLM）在工业界和学术界的最新进展和广泛采用引发了对其服务的巨大需求。然而，传统的云服务成本高昂，而纯设备端推理由于资源有限而面临挑战。边缘云协作成为一个关键研究方向，结合两种范式的优势，但在有效利用有限网络带宽的同时充分利用和平衡边缘设备和云的计算能力仍然是一个开放问题。为了应对这些挑战，我们提出了流水线协作投机解码框架（PicoSpec），这是一种新颖的、通用的、无需训练的投机解码框架，用于LLM边缘云协作推理。我们设计了一个异步流水线，解决了边缘协作场景中 vanilla 投机解码固有的相互等待问题，该流水线在边缘设备上并行执行小型语言模型（SLM），在云上执行LLM。同时，为了减轻传输词汇分布引起的大量通信延迟，我们引入了稀疏压缩的独立拒绝采样，只需一次性传输压缩词汇即可完成拒绝采样。实验结果表明，我们的解决方案优于基线和现有方法，实现了高达2.9倍的加速。

---

## 1. Introduction (引言)

*(摘要内容有限，建议下载PDF获取完整引言)*

Large Language Models (LLMs) have become increasingly important in both industry and academia. However, traditional cloud-based LLM services face challenges of high costs and latency, while on-device inference is limited by hardware resources. Edge-cloud collaboration has emerged as a promising direction to address these challenges.

The key contributions of this paper include:
1. **PicoSpec Framework**: A novel pipelined collaborative speculative decoding framework
2. **Asynchronous Pipeline**: Resolves the mutual waiting problem in edge-cloud scenarios
3. **Sparse Compression**: Reduces communication latency with compressed vocabulary

---

大型语言模型（LLM）在工业界和学术界变得越来越重要。然而，传统的基于云的LLM服务面临高成本和延迟的挑战，而设备端推理受到硬件资源的限制。边缘云协作已成为应对这些挑战的有前途的方向。

本文的主要贡献包括：
1. **PicoSpec框架**：一种新颖的流水线协作投机解码框架
2. **异步流水线**：解决边缘云场景中的相互等待问题
3. **稀疏压缩**：通过压缩词汇减少通信延迟

---

## 实验结果

- **加速比**: 最高 2.9×

---

## 关键词

Edge-Cloud Collaboration, Speculative Decoding, LLM Inference, Distributed Computing