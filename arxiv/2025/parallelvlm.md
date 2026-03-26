# ParallelVLM: Lossless Video-LLM Acceleration with Visual Alignment Aware Parallel Speculative Decoding

## 论文信息

- **arXiv**: https://arxiv.org/abs/2503.XXXXX
- **作者**: Quan Kong, Yuhao Shen, Yicheng Ji, Huan Li, Cong Wang
- **提交时间**: 2025年3月23日

## 摘要

尽管当前的视频语言模型 (Video-LLMs) 在视频理解任务上取得了令人印象深刻的性能，但它们的自回归解码效率仍然受到大量视频token的限制。

视觉token剪枝可以部分缓解这一瓶颈，但现有方法仍存在信息丢失问题，且加速效果有限。

**ParallelVLM** 提出了一种**视觉对齐感知的并行投机解码**方法，实现无损加速。

## 核心问题

- Video-LLM中大量视觉token导致的推理瓶颈
- 现有剪枝方法造成信息丢失
- 加速效果不理想

## 核心贡献

1. **视觉对齐感知**: 考虑视觉token之间的对齐关系
2. **并行投机解码**: 支持多个token并行预测
3. **无损加速**: 在不损失模型性能的前提下加速

## 技术特点

- 视觉token关系建模
- 并行draft预测
- 自适应验证策略