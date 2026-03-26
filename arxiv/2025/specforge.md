# SpecForge: A Flexible and Efficient Open-Source Training Framework for Speculative Decoding

## 论文信息

- **arXiv**: https://arxiv.org/abs/2503.14866
- **作者**: Shenggui Li, Chao Wang, Yikai Zhu, Yubo Wang, Fan Yin, Shuai Shi, Yefei Chen, Xiaomin Dong, Qiaoling Chen, Jin Pan, Ji Li, Laixin Xie, Yineng Zhang, Lei Yu, Yonggang Wen, Ivor Tsang, Tianwei Zhang
- **提交时间**: 2025年3月

## 摘要

大型语言模型由于顺序自回归解码而产生高推理延迟。Speculative Decoding (投机解码) 通过使用draft模型预测多个token，然后使用target模型并行验证，是加速自回归LLM推理的一种有前景的方法。

然而，现有的投机解码框架主要关注推理阶段，缺乏对draft模型训练的支持，导致部署效率受限。

**SpecForge** 是一个灵活高效的**开源训练框架**，专门用于投机解码。

## 核心贡献

1. **完整训练流程**: 提供draft模型训练的完整支持
2. **灵活架构**: 支持多种draft-target模型组合
3. **高效实现**: 优化训练过程中的计算效率
4. **开源**: 便于研究社区使用和改进

## 技术特点

- 支持多种draft模型架构
- 灵活的训练策略配置
- 高效的梯度计算
- 与主流推理框架集成

## GitHub

https://github.com/SpecForge