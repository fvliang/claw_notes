# RedFuser: An Automatic Operator Fusion Framework for Cascaded Reductions on AI Accelerators

**论文链接**: [arXiv:2603.10026](https://arxiv.org/abs/2603.10026)

**作者**: Xinsheng Tang, Yuhui Zhao, Jintao Li, Jiaming Xu, Shuo Li, Jiansong Chen, Chen Zhang, Yong Li, Xiaoyong Liu, Ji Liu, Jin Wang, Wei Lin

**会议**: ASPLOS 2026

---

## Abstract (摘要)

Operator fusion, as a key performance optimization technique in the deployment of AI models, significantly improves execution efficiency and has been widely adopted in modern AI compilers. However, for cascaded reduction operations involving multiple loops with inter-loop data dependencies, such as the safe softmax followed by GEMM within attention mechanisms, existing compilers lack effective automated fusion and kernel generation capabilities. Although some works have addressed specific instances through hand-crafted fusion strategies, their solutions are limited in generality and difficult to extend to other similar structures. Given the prevalence of such computational patterns in deep learning models, there remains significant untapped potential in achieving general and automated fusion optimization.

In this paper, we present a formal theoretical methodology for analyzing cascaded reductions which can fuse them into a single loop and introduce an incremental computation form. Based on this methodology, we design Reduction Fuser (RedFuser), a framework that automatically identifies supported cascaded reduction patterns and generates optimized fused kernels. Experiments show that RedFuser successfully fuses diverse workloads, achieving up to 2× to 5× speedup over state-of-the-art AI compilers and matching the performance of highly optimized hand-written kernels. The code is available at this https URL.

---

算子融合作为AI模型部署中的关键性能优化技术，显著提高了执行效率，并已被现代AI编译器广泛采用。然而，对于涉及循环间数据依赖的级联归约操作，例如注意力机制中的安全softmax后接GEMM，现有编译器缺乏有效的自动融合和内核生成能力。虽然一些工作通过手工融合策略解决了特定实例，但其解决方案通用性有限，难以扩展到其他类似结构。鉴于此类计算模式在深度学习模型中的普遍性，在实现通用和自动融合优化方面仍有显著的未开发潜力。

在本文中，我们提出了一种形式化的理论方法来分析级联归约，可以将其融合到单个循环中并引入增量计算形式。基于这种方法，我们设计了Reduction Fuser（RedFuser），这是一个自动识别支持的级联归约模式并生成优化融合内核的框架。实验表明，RedFuser成功融合了多样化的工作负载，在最先进的AI编译器上实现了高达2倍到5倍的加速，并与高度优化的手工内核性能相匹配。代码可在https://github.com/alibaba/redfuser获取。

---

## 主要贡献

1. **形式化理论方法**：分析级联归约，可融合为单个循环
2. **增量计算形式**：优化计算过程
3. **RedFuser框架**：自动识别和融合级联归约模式

---

## 实验结果

- 在最先进的AI编译器上实现 **2-5倍** 加速
- 性能与高度优化的手工内核相当

**代码**: https://github.com/alibaba/redfuser