# SpecSteer: Synergizing Local Context and Global Reasoning for Efficient Personalized Generation

## 论文信息

- **arXiv**: https://arxiv.org/abs/2503.XXXXX
- **作者**: Hang Lv, Sheng Liang, Hao Wang, Yongyue Zhang, Hongchao Gu, Wei Guo, Defu Lian, Yong Liu, Enhong Chen
- **提交时间**: 2025年3月17日

## 摘要

个性化生成需要结合用户的本地上下文和云端的大规模推理能力。本地设备上的模型可以快速响应，但能力有限；云端模型能力强，但延迟高。

**SpecSteer** 是一个将私有设备端上下文与云端规模推理协同的框架。

## 核心创新

**将协作定义为贝叶斯知识融合**，并**重新利用投机解码作为分布式对齐协议**：

- **Draft-Verify-Recover管道**:
  - 设备端模型生成个性化序列 (Draft)
  - 云端模型验证 (Verify)
  - 错误恢复机制 (Recover)

## 技术特点

- 贝叶斯知识融合
- 分布式对齐协议
- 设备-云协同
- 自适应验证策略

## 解决的问题

- 设备端模型能力不足
- 云端推理延迟高
- 个性化与性能的平衡