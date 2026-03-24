# ALISE: Speculative Scheduling for LLM Serving

## 论文信息
- **作者**: Various
- **会议**: arXiv 2024
- **arXiv**: https://arxiv.org/abs/2410.19690
- **日期**: 2024.10

## 摘要 (Abstract)
ALISE (Accelerating LLM Serving with Speculative Scheduling) combines speculative decoding with intelligent scheduling:

1. **Speculative scheduling**: Intelligently schedules speculation
2. **Workload-aware**: Adapts to request characteristics
3. **Batch optimization**: Better batching decisions with speculation
4. **Improved throughput**: Significant gains over non-speculative systems

## 摘要中文
ALISE（通过投机调度加速LLM服务）将投机解码与智能调度相结合：

1. **投机调度**: 智能调度投机
2. **工作负载感知**: 适应请求特征
3. **批处理优化**: 更好的批处理决策与投机
4. **改进的吞吐量**: 比非投机系统显著的改进

## 引言 (Introduction)
Traditional serving systems:
- Treat speculation as add-on optimization
- Don't fully exploit speculation potential
- Suboptimal scheduling decisions

ALISE improves:
- **Integrated scheduling**: Full integration of speculation into scheduler
- **Predictive speculation**: Speculate based on predicted accept rates
- **Batch synergy**: Better batching when speculation is considered

## 总结
ALISE demonstrates that tight integration of speculation and scheduling yields significant improvements.