# MineDraft: A Framework for Batch Parallel Speculative Decoding

## 论文信息

- **arXiv**: https://arxiv.org/abs/2502.XXXXX
- **作者**: Zhenwei Tang, Arun Verma, Zijian Zhou, Zhaoxuan Wu, Alok Prakash, Daniela Rus, Bryan Kian Hsiang Low
- **提交时间**: 2026年2月

## 摘要

投机解码 (Speculative Decoding) 是一种通过使用draft模型预测token序列，然后由target模型验证来加速LLM推理的技术。

然而，现有的投机解码框架主要针对单序列推理进行优化，无法有效处理批处理场景。

**MineDraft** 是一个**批量并行投机解码框架**，专门针对批处理推理场景进行优化。

## 核心贡献

1. **批量并行**: 支持多个序列同时进行投机解码
2. **高效调度**: 优化批处理中的token验证调度
3. **内存优化**: 减少批处理场景下的内存占用

## 技术特点

- 批量token树构建
- 并行验证策略
- 动态批处理大小调整

## 解决的问题

- 批处理场景下的投机解码效率
- 多序列并行处理
- 资源利用率优化