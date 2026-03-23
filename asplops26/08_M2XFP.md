# M2XFP: A Metadata-Augmented Microscaling Data Format for Efficient Low-bit Quantization

**论文链接**: [arXiv:2601.19213](https://arxiv.org/abs/2601.19213)

**作者**: Weiming Hu, Zihan Zhang, Haoyan Zhang, Chen Zhang, Cong Guo, Yu Feng, Tianchi Hu, Guanglin Li, Guipeng Hu, Junsong Wang, Jingwen Leng

**会议**: ASPLOS 2026

---

## Abstract (摘要)

Existing low-bit Microscaling (MX) formats, such as MXFP4, often suffer from substantial accuracy degradation due to the use of a shared scaling factor with the Power-of-Two format. In this work, we explore strategies that introduce minimal metadata to recover accuracy lost during quantization while maintaining high bit efficiency across a wide range of large language models.

We propose a complete algorithm-hardware co-design based on flexible metadata, featuring an online quantization with simple encoding. To support the proposed method efficiently, we implement a lightweight hardware unit and integrate it into the accelerator.

Evaluation results demonstrate that our method substantially narrows the accuracy gap, achieving on average a 70.63% reduction in accuracy loss compared to MXFP4 and a 37.30% reduction relative to the latest NVFP4 on LLM benchmarks. Furthermore, our design delivers up to 1.91× speedup and 1.75× energy savings over state-of-the-art accelerators. Our code is available at this https URL.

---

现有的低位微缩放（MX）格式（如MXFP4）由于使用Power-of-Two格式的共享缩放因子，往往会遭受严重的精度下降。在这项工作中，我们探索引入最少元数据的策略，以在量化过程中恢复丢失的精度，同时在广泛的大型语言模型中保持高比特效率。

我们提出了一种基于灵活元数据的完整算法-硬件协同设计，具有简单编码的在线量化。为了有效支持所提出的方法，我们实现了一个轻量级硬件单元并将其集成到加速器中。

评估结果表明，我们的方法显著缩小了精度差距，与MXFP4相比，在LLM基准上平均减少70.63%的精度损失，与最新的NVFP4相比减少37.30%。此外，我们的设计在最先进的加速器上提供高达1.91倍的加速和1.75倍的节能。我们的代码可在https://github.com/SJTU-ReArch-Group/M2XFP_ASPLOS26获取。

---

## 主要贡献

1. **灵活元数据设计**：最小化元数据开销
2. **在线量化**：简单编码
3. **轻量级硬件单元**：集成到加速器
4. **算法-硬件协同设计**

---

## 实验结果

| 指标 | 改进 |
|------|------|
| 相比MXFP4精度损失减少 | 70.63% |
| 相比NVFP4精度损失减少 | 37.30% |
| 加速比 | 1.91× |
| 能耗节省 | 1.75× |

**代码**: https://github.com/SJTU-ReArch-Group/M2XFP_ASPLOS26