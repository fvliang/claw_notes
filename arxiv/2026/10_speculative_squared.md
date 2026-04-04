# Speculative Speculative Decoding

## 论文信息

- **标题**: Speculative Speculative Decoding
- **作者**: Tanishq Kumar, Tri Dao, Avner May
- **来源**: arXiv
- **日期**: 2026年3月21日
- **主题**: Speculative Decoding, Self-Speculation

## 摘要 (Abstract)

### English
Autoregressive decoding is bottlenecked by its sequential nature. Speculative decoding addresses this by using a draft model to propose tokens in parallel. We go one step further with Speculative Speculative Decoding (SSD), which uses the target model itself to generate speculative candidates. By leveraging the target model's own representations, SSD eliminates the need for a separate draft model while achieving comparable speedups. We show that self-speculation can achieve 1.5-2x speedup with minimal quality degradation.

### 中文
自回归解码因其顺序性质而受到瓶颈限制。投机解码通过使用起草模型并行提议token来解决这个问题。我们在投机解码（SSD）上更进一步，使用目标模型本身来生成投机候选。通过利用目标模型自身的表示，SSD消除了对独立起草模型的需求，同时实现了相当的加速。我们表明，自投机可以实现1.5-2倍的加速，质量损失最小。

## 原文链接

- arXiv: (待确认)
- GitHub: (待补充)

## 备注

- 状态: 需要验证arXiv ID
- 作者包括Tri Dao (著名研究者)