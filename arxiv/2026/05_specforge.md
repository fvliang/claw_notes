# SpecForge: A Flexible and Efficient Open-Source Training Framework for Speculative Decoding

## 论文信息

- **标题**: SpecForge: A Flexible and Efficient Open-Source Training Framework for Speculative Decoding
- **作者**: Shenggui Li, Chao Wang, Yikai Zhu, Yubo Wang, Fan Yin, Shuai Shi, Yefei Chen, Xiaomin Dong, Qiaoling Chen, Jin Pan, Laixin Xie, Yineng Zhang, Lei Yu, Yonggang Wen, Ivor Tsang, Tianwei Zhang
- **来源**: arXiv
- **日期**: 2026年3月19日
- **主题**: Speculative Decoding, LLM Training

## 摘要 (Abstract)

### English
Large language models incur high inference latency due to sequential autoregressive decoding. Speculative decoding addresses this by using a draft model to propose candidate tokens, which are then verified in parallel by the target model. However, training effective draft models remains challenging and requires careful design of training objectives and data pipelines. We present SpecForge, a flexible and efficient open-source training framework for speculative decoding. SpecForge provides modular components for draft model training, including:
- Multiple training objectives (policy matching, reward-based, etc.)
- Automated dataset preparation pipelines
- Seamless integration with serving frameworks like SGLang

Our framework reduces the barrier to entry for speculative decoding research and enables rapid experimentation.

### 中文
大型语言模型由于顺序自回归解码而产生高推理延迟。投机解码通过使用起草模型提出候选token来解决这个问题，然后由目标模型并行验证。然而，训练有效的起草模型仍然具有挑战性，需要仔细设计训练目标和数据管道。我们提出了SpecForge，一个灵活高效的开源投机解码训练框架。SpecForge为起草模型训练提供模块化组件，包括：
- 多种训练目标（策略匹配、基于奖励等）
- 自动化数据集准备管道
- 与SGLang等服务框架的无缝集成

我们的框架降低了投机解码研究的门槛，实现了快速实验。

## 引言 (Introduction)

### English
Speculative decoding consists of two phases:
1. **Drafting**: A smaller model generates candidate tokens
2. **Verification**: The target model validates candidates in parallel

The effectiveness of speculative decoding depends heavily on the quality of the draft model. Training good draft models requires:
- Appropriate training objectives
- Quality data pipelines
- Integration with serving systems

SpecForge addresses these needs by providing a comprehensive training framework.

### 中文
投机解码包括两个阶段：
1. **起草**：较小的模型生成候选token
2. **验证**：目标模型并行验证候选

投机解码的有效性很大程度上取决于起草模型的质量。训练好的起草模型需要：
- 适当的训练目标
- 高质量的数据管道
- 与服务系统的集成

SpecForge通过提供全面的训练框架来满足这些需求。

## 原文链接

- arXiv: (待确认)
- GitHub: https://github.com/sgl-project/SpecForge

## 补充材料

- 博客: (待补充)
- SGLang集成: https://github.com/sgl-project/sglang

## 备注

- 状态: 需要验证arXiv ID
- GitHub项目已存在，需要补充更详细的内容