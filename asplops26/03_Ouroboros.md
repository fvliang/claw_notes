# Ouroboros: Wafer-Scale SRAM CIM with Token-Grained Pipelining for Large Language Model Inference

**论文链接**: [arXiv:2603.02737](https://arxiv.org/abs/2603.02737)

**作者**: Yiqi Liu, Cheng Liu, Zhen Gu, Tianchen Ding, Zongyue Zhao, Ziyu Yang, Yufei Ding, Yibo Lin, Mingjie Lin, Xiaowei Li, Zidong Du, Chen Liu, Yunji Chen

**会议**: ASPLOS 2026

---

## Abstract (摘要)

Conventional LLM inference architectures suffer from high energy and latency due to frequent data movement across memory hierarchies. We propose Ouroboros, a wafer-scale SRAM-based Computing-in-Memory (CIM) architecture that executes all operations in situ, eliminating off-chip migration. To maximize its limited first-level capacity, we introduce three innovations:

1. **Token-Grained Pipelining**: Replaces sequence-level pipelining to mitigate length variations, boosting utilization and reducing activation storage.

2. **Distributed Dynamic KV Cache Management**: Decouples memory from compute to leverage fragmented SRAM for efficient KV storage.

3. **Communication-Aware Mapping**: Optimizes core allocation for locality and fault tolerance across the wafer.

Experimental results show Ouroboros achieves average gains of 4.1× in throughput and 4.2× in energy efficiency, peaking at 9.1× and 17× for the 13B model.

---

传统的LLM推理架构由于跨内存层次结构的频繁数据移动而遭受高能耗和高延迟。我们提出了Ouroboros，这是一种基于晶圆级SRAM的存算一体（CIM）架构，在本地执行所有操作，消除了芯片外迁移。为了最大化其有限的一级容量，我们引入了三个创新：

1. **Token级流水线**：用序列级流水线替换以缓解长度变化，提高利用率并减少激活存储。

2. **分布式动态KV缓存管理**：将内存与计算解耦，以利用碎片化SRAM进行高效的KV存储。

3. **通信感知映射**：优化晶圆上核心分配的局部性和容错性。

实验结果表明，Ouroboros在吞吐量上平均实现4.1倍的提升，在能效上实现4.2倍的提升，在13B模型上分别达到9.1倍和17倍的峰值。

---

## 1. Introduction (引言)

*(注：本文获取introduction失败，仅获取到摘要内容。如需完整introduction，建议下载PDF：https://arxiv.org/pdf/2603.02737)*

The development of Large Language Models (LLMs) has revolutionized artificial intelligence, but their deployment faces significant challenges due to the massive computational and memory requirements. Conventional LLM inference architectures suffer from high energy consumption and latency because of frequent data movement across memory hierarchies.

To address these challenges, we propose Ouroboros, a wafer-scale SRAM-based Computing-in-Memory (CIM) architecture designed specifically for efficient LLM inference. The key innovation of Ouroboros is that it executes all operations in situ, completely eliminating the need for off-chip data migration.

---

大型语言模型（LLM）的发展已经革新了人工智能，但其部署由于大量的计算和内存需求而面临重大挑战。传统的LLM推理架构由于跨内存层次的频繁数据移动而遭受高能耗和高延迟。

为了应对这些挑战，我们提出了Ouroboros，这是一种专门为高效LLM推理设计的晶圆级SRAM存算一体（CIM）架构。Ouroboros的关键创新在于它在本地执行所有操作，完全消除了芯片外数据迁移的需要。

---

## 主要创新

1. **Token-Grained Pipelining（Token级流水线）**
   - 替换传统的序列级流水线
   - 缓解长度变化问题
   - 提高硬件利用率
   - 减少激活存储需求

2. **Distributed Dynamic KV Cache Management（分布式动态KV缓存管理）**
   - 将内存与计算解耦
   - 利用碎片化SRAM空间
   - 实现高效的KV缓存存储

3. **Communication-Aware Mapping（通信感知映射）**
   - 优化核心分配策略
   - 提高局部性
   - 增强容错能力

---

## 实验结果

| 指标 | 平均提升 | 13B模型峰值 |
|------|----------|-------------|
| 吞吐量 | 4.1× | 9.1× |
| 能效 | 4.2× | 17× |