# EAGLE-Pangu: Accelerator-Safe Tree Speculative Decoding on Ascend NPUs

## 论文信息

- **标题**: EAGLE-Pangu: Accelerator-Safe Tree Speculative Decoding on Ascend NPUs
- **作者**: Chang Han, Yijie Hu, Jingling Liu
- **来源**: arXiv
- **日期**: 2026年3月9日
- **主题**: Speculative Decoding, NPU Acceleration

## 摘要 (Abstract)

### English
Autoregressive decoding remains a primary bottleneck in large language model (LLM) serving, motivating the adoption of speculative decoding techniques. However, most existing speculative decoding methods are designed for GPUs and don't fully utilize the unique characteristics of NPUs (Neural Processing Units). We present EAGLE-Pangu, a tree speculative decoding framework specifically optimized for Huawei Ascend NPUs. EAGLE-Pangu leverages NPU-specific features like efficient tree traversal hardware and dedicated attention kernels to achieve optimal performance. Our implementation achieves 2.3x speedup on Ascend NPUs while maintaining output quality.

### 中文
自回归解码仍然是大型语言模型（LLM）服务的主要瓶颈，这促使人们采用投机解码技术。然而，大多数现有的投机解码方法是针对GPU设计的，没有充分利用NPU（神经处理单元）的独特特性。我们提出了EAGLE-Pangu，一个专门针对华为Ascend NPUs优化的树状投机解码框架。EAGLE-Pangu利用NPU特定功能，如高效树遍历硬件和专用注意力内核，以实现最佳性能。我们的实现在Ascend NPU上实现了2.3倍的加速，同时保持输出质量。

## 原文链接

- arXiv: (待确认)
- GitHub: (待补充)

## 备注

- 状态: 需要验证arXiv ID
- 专门针对华为Ascend NPU优化