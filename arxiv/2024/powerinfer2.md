# PowerInfer-2: High-Speed LLM Inference for Smartphones

## 论文信息

- **arXiv**: https://arxiv.org/abs/2406.06282
- **作者**: 来自上海交通大学IPADS实验室
- **提交时间**: 2024年6月

## 摘要

PowerInfer-2是一个专为智能手机设计的高度优化推理框架。

使用TurboSparse-Mixtral-47B，PowerInfer-2在智能手机上实现了**11.68 tokens/s**的推理速度，比其他SOTA框架快达**22倍**。

## 核心创新

1. **稀疏激活利用**: 利用LLM的稀疏激活特性
2. **混合计算**: CPU-GPU协同计算
3. **移动端优化**: 针对移动设备的专门优化

## TurboSparse

与PowerInfer-2一起发布的还有TurboSparse模型系列：

- **TurboSparse-Mixtral**: 激活仅4B参数的Mixtral级模型
- 稀疏化率约90%
- 仅需$0.1M微调成本
- 保持原有模型性能

## 论文

- [PowerInfer-2论文](https://arxiv.org/abs/2406.06282)
- [TurboSparse论文](https://arxiv.org/abs/2406.05955)

## GitHub

https://github.com/Tiiny-AI/PowerInfer