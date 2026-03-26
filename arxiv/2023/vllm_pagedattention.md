# vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention

## 论文信息

- **arXiv**: https://arxiv.org/abs/2309.06180
- **会议**: SOSP 2023 (ACM Symposium on Operating Systems Principles)
- **作者**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica

## 摘要

大语言模型 (LLM) 服务的效率高度依赖于内存的使用。现有的系统将KV缓存存储在连续的内存空间中，由于内存碎片化，导致：

- 批处理大小受限于可用内存
- 显存利用率低
- 难以处理长上下文

**PagedAttention** 是一种受操作系统分页启发的注意力机制，允许非连续的KV缓存存储。

基于PagedAttention构建的 **vLLM** 在LLaMA-7B上实现了 **2.4倍** 的吞吐量提升，在OPT-175B上实现了 **8.5倍** 的吞吐量提升。

## 核心贡献

1. **PagedAttention**: 受OS分页启发的非连续KV缓存管理
2. **vLLM系统**: 基于PagedAttention的高效LLM服务引擎
3. **连续批处理**: 优化请求调度

## 技术细节

### 传统方法的问题
- KV缓存连续存储
- 内存碎片化
- 预留大量内存

### PagedAttention解决方案
- 分页式KV缓存存储
- 按需分配内存
- 共享前缀优化

## 实验结果

| 模型 | 基线 | vLLM | 提升 |
|------|------|------|------|
| LLaMA-7B | 1.0x | 2.4x | 2.4倍 |
| OPT-175B | 1.0x | 8.5x | 8.5倍 |

## 相关资源

- [论文](https://arxiv.org/abs/2309.06180)
- [GitHub](https://github.com/vllm-project/vllm)
- [博客](https://blog.vllm.ai/2023/06/20/vllm.html)