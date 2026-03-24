# REST: Retrieval-Based Speculative Decoding

## 论文信息
- **作者**: FasterDecoding
- **会议**: NAACL 2024
- **GitHub**: https://github.com/FasterDecoding/REST
- **日期**: 2024

## 摘要 (Abstract)
REST (Retrieval-Based Speculative Decoding) introduces a novel approach that leverages previously computed contexts to accelerate inference. Instead of always using a smaller model as the drafter, REST:

1. **Retrieves similar contexts** from recent history
2. **Uses retrieved tokens as drafts** for verification
3. **Achieves high acceptance rates** when prompts share common patterns

This approach is particularly effective for tasks with repetitive patterns or structured outputs.

## 摘要中文
REST（基于检索的投机解码）引入了一种新方法，利用先前计算的上下文来加速推理。REST不总是使用较小的模型作为drafter，而是：

1. **从历史记录中检索相似上下文**
2. **使用检索到的tokens作为验证的draft**
3. **当prompt共享共同模式时实现高接受率**

这种方法对于具有重复模式或结构化输出的任务特别有效。

## 引言 (Introduction)
Key insight: Many inference requests share common patterns (e.g., system prompts, formatting instructions). REST exploits this by:

1. **Building a retrieval cache** of recent context chunks
2. **Querying the cache** for similar contexts when generating
3. **Using matched tokens** as speculative drafts
4. **Verifying with the full model** to ensure correctness

This eliminates the need for a separate draft model and can achieve better efficiency.

## GitHub 介绍
Official implementation of REST: Retrieval-Based Speculative Decoding (NAACL 2024). The system demonstrates significant speedups for various LLM inference scenarios.