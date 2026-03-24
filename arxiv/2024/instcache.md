# InstCache: Predictive Cache for LLM Serving

## 论文信息
- **作者**: Various
- **会议**: arXiv 2024
- **arXiv**: https://arxiv.org/abs/2411.12410
- **日期**: 2024.11

## 摘要 (Abstract)
InstCache introduces a predictive caching mechanism for LLM serving. The key idea is to anticipate what will be needed and proactively cache it:

1. **Prefix prediction**: Predicts common prompt prefixes
2. **Intent anticipation**: Anticipates user intent from context
3. **Smart pre-caching**: Loads likely-needed KV cache in advance
4. **High hit rates**: Achieves significant cache hit improvements

## 摘要中文
InstCache为LLM服务引入了一种预测性缓存机制。关键思想是预测需要什么并主动缓存它：

1. **前缀预测**: 预测常见的prompt前缀
2. **意图预测**: 从上下文预测用户意图
3. **智能预缓存**: 预先加载可能需要的KV缓存
4. **高命中率**: 实现显著的缓存命中率提升

## 引言 (Introduction)
LLM serving can benefit from caching:
- Similar prompts appear frequently
- System prompts are often reused
- Common patterns exist in queries

InstCache leverages this with:
- **Learning-based prediction**: ML model predicts what to cache
- **Prefix tree**: Efficient prefix matching
- **Eviction policy**: Optimal cache management

## 总结
InstCache demonstrates that predictive caching can significantly improve LLM serving efficiency.