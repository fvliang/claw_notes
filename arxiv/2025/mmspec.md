# MMSpec: Benchmarking Speculative Decoding for Vision-Language Models

## 论文信息

- **arXiv**: https://arxiv.org/abs/2503.XXXXX
- **作者**: Hui Shen, Xin Wang, Ping Zhang, Yunta Hsieh, Qi Han, Zhongwei Wan, Ziheng Zhang, Jingxuan Zhang, Jing Xiong, Ziyuan Liu, Yifan Zhang, Hangrui Cao, Chenyang Zhao, Mi Zhang
- **提交时间**: 2025年3月16日

## 摘要

视觉语言模型 (Vision-Language Models, VLMs) 在多模态任务上表现出色，但由于模型规模大、上下文长，推理延迟很高。

投机解码 (Speculative Decoding, SD) 是一种有前景的加速方法，但现有工作主要集中在纯语言模型上，VLM上的投机解码缺乏系统研究。

**MMSpec** 是第一个专门针对**视觉语言模型投机解码**的基准测试框架。

## 核心问题

- VLM推理延迟高
- 现有SD方法未针对VLM优化
- 缺乏VLM上SD的评估基准

## 核心贡献

1. **首个VLM投机解码基准**: 系统评估VLM上的SD方法
2. **评估框架**: 统一的性能评估标准
3. **详细分析**: 不同SD策略在VLM上的表现

## 技术特点

- 多VLM模型测试
- 多维度性能评估
- 视觉token处理优化