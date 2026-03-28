# SpecForge: Training Speculative Decoding Framework

## 基本信息

- **标题**: SpecForge: A Flexible and Efficient Open-Source Training Framework for Speculative Decoding
- **作者**: Shenggui Li, Chao Wang, Yikai Zhu, Yubo Wang, Fan Yin, Shuai Shi, Yefei Chen, Xiaomin Dong, Qiaoling Chen, Jin Pan, Ji Li, Laixin Xie, Yineng Zhang, Lei Yu, Yonggang Wen, Ivor Tsang, Tianwei Zhang
- **arXiv**: [2603.XXXXX](https://arxiv.org/abs/)
- **GitHub**: [sgl-project/SpecForge](https://github.com/sgl-project/SpecForge)
- **发布时间**: 2026年3月

## 摘要 (Abstract)

Large language models incur high inference latency due to sequential autoregressive decoding. Speculative decoding has emerged as a promising solution to accelerate inference by leveraging a smaller draft model to propose candidate tokens, which are then verified in parallel by a larger target model. However, training effective draft models remains challenging due to the lack of flexible and efficient training frameworks. We introduce SpecForge, an open-source training framework for speculative decoding that addresses these challenges.

## 摘要 (中文)

大型语言模型由于顺序自回归解码而产生高推理延迟。推测解码已成为一种有前景的解决方案，通过利用较小的draft模型提出候选令牌，然后由较大的目标模型并行验证来加速推理。然而，由于缺乏灵活有效的训练框架，训练有效的draft模型仍然具有挑战性。我们推出了SpecForge，这是一个用于推测解码的开源训练框架，解决了这些挑战。

## GitHub 仓库介绍

SpecForge 是一个用于轻松训练推测解码模型并将其顺利移植到SGLang服务的框架。

### 主要特性
- **灵活的训练框架**: 支持多种推测解码训练方法
- **SGLang集成**: 无缝集成到SGLang服务
- **开源**: 完全开源

### Stars
- 749 stars

### 使用示例

```python
# Train speculative decoding models effortlessly
from specforge import Trainer

trainer = Trainer(model=target_model, draft_model=draft_model)
trainer.train()
```