# HeiSD: Hybrid Speculative Decoding for Embodied Vision-Language-Action Models with Kinematic Awareness

## 论文信息

- **arXiv**: https://arxiv.org/abs/2503.XXXXX
- **作者**: Zihao Zheng, Zhihao Mao, Sicheng Tian, Maoliang Li, Jiayu Chen, Xinhao Sun, Zhaobo Zhang, Xuanzhe Liu, Donggang Cao, Hong Mei, Xiang Chen
- **提交时间**: 2025年3月18日

## 摘要

视觉语言动作模型 (Vision-Language-Action, VLA) 已成为机器人控制的主流解决方案，但推理速度较慢。

投机解码 (Speculative Decoding, SD) 是一种有前景的加速方法，可分为两类：
- 基于draft的SD
- 基于检索的SD

现有方法未能分析VLA模型的独特优势。

**HeiSD** 提出了一种**具有运动学感知的混合投机解码**方法。

## 核心问题

- VLA模型推理速度慢
- 现有SD方法未针对VLA优化
- 缺乏对VLA运动学特性的利用

## 核心贡献

1. **混合框架**: 结合draft-based和retrieval-based SD
2. **运动学感知**: 利用VLA的运动学特性
3. **VLA优化**: 针对VLA模型的专门优化

## 技术特点

- 运动学建模
- 混合draft策略
- 自适应检索