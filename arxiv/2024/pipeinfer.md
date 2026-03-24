# PipeInfer: Asynchronous Pipelined Speculation

## 论文信息
- **作者**: AutonomicPerfectionist
- **会议**: arXiv 2024
- **GitHub**: https://github.com/AutonomicPerfectionist/PipeInfer
- **日期**: 2024

## 摘要 (Abstract)
PipeInfer introduces asynchronous pipelined speculation for LLM inference acceleration. The key innovation is combining:

1. **Pipelining**: Overlap multiple requests' computation
2. **Speculation**: Use draft tokens for acceleration
3. **Asynchronous execution**: Maximize GPU utilization

This approach achieves better throughput by keeping the GPU constantly busy with minimal idle time.

## 摘要中文
PipeInfer为LLM推理加速引入了异步流水线投机。关键创新结合了：

1. **流水线**: 重叠多个请求的计算
2. **投机**: 使用draft tokens加速
3. **异步执行**: 最大化GPU利用率

这种方法通过保持GPU持续繁忙并最大限度地减少空闲时间来获得更好的吞吐量。

## 引言 (Introduction)
Traditional speculative decoding has limitations:
- Synchronous execution creates bubbles
- Poor GPU utilization between speculation and verification
- Limited parallelism across requests

PipeInfer addresses these with:
- **Continuous speculation pipeline**: Always have drafts in flight
- **Decoupled verification**: Verify previous drafts while generating new ones
- **Request-level parallelism**: Multiple requests in pipeline simultaneously

## GitHub 介绍
PipeInfer: Accelerating LLM Inference using Asynchronous Pipelined Speculation. The implementation demonstrates significant throughput improvements over standard speculative decoding approaches.