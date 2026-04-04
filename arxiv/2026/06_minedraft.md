# MineDraft: A Framework for Batch Parallel Speculative Decoding

## 论文信息

- **标题**: MineDraft: A Framework for Batch Parallel Speculative Decoding
- **作者**: Zhenwei Tang, Arun Verma, Zijian Zhou, Zhaoxuan Wu, Alok Prakash, Daniela Rus, Bryan Kian Hsiang Low
- **来源**: arXiv
- **日期**: 2026年2月24日
- **主题**: Speculative Decoding, Batch Processing

## 摘要 (Abstract)

### English
Speculative decoding accelerates LLM inference by parallelizing token generation. However, existing approaches focus on single-request optimization and don't fully exploit batch-level parallelism. We present MineDraft, a framework for batch parallel speculative decoding that maximizes throughput by processing multiple speculation trees simultaneously. MineDraft introduces:
- Batch-level tree construction algorithms
- Efficient batch verification kernels
- Adaptive scheduling for heterogeneous batch sizes

Our evaluation shows that MineDraft achieves up to 3.2x throughput improvement over single-request speculative decoding on various workloads.

### 中文
投机解码通过并行化token生成来加速LLM推理。然而，现有方法专注于单请求优化，无法充分利用批级并行性。我们提出了MineDraft，一个批处理并行投机解码框架，通过同时处理多个投机树来最大化吞吐量。MineDraft引入了：
- 批级树构建算法
- 高效的批验证内核
- 针对异构批大小的自适应调度

我们的评估表明，MineDraft在各种工作负载上实现了比单请求投机解码高达3.2倍的吞吐量提升。

## 引言 (Introduction)

### English
Traditional speculative decoding optimizes a single request at a time. In production serving scenarios, however, multiple requests arrive concurrently and need to be served together. Batch parallel speculative decoding can significantly improve GPU utilization and throughput.

MineDraft addresses this by:
1. Constructing speculation trees for multiple requests together
2. Verifying all candidates in optimized batch kernels
3. Scheduling based on batch characteristics

### 中文
传统投机解码一次优化一个请求。然而，在生产服务场景中，多个请求同时到达，需要一起服务。批处理并行投机解码可以显著提高GPU利用率和吞吐量。

MineDraft通过以下方式解决这个问题：
1. 一起为多个请求构建投机树
2. 在优化的批处理内核中验证所有候选
3. 根据批处理特性进行调度

## 原文链接

- arXiv: (待确认)
- GitHub: (待补充)

## 补充材料

- 博客: (待补充)
- 相关GitHub: (待补充)

## 备注

- 状态: 需要验证arXiv ID
- 需要补充完整的GitHub链接和博客内容