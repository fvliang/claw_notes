# ZoomR: Memory Efficient Reasoning through Multi-Granularity Key Value Retrieval

**arXiv**: 2604.10898
**链接**: https://arxiv.org/abs/2604.10898
**作者**: David H. Yang, Yuxuan Zhu, Mohammad Mohammadi Amiri, Keerthiram Murugesan, Tejaswini Pedapati, Subhajit Chaudhury, Pin-Yu Chen
**会议**: arXiv 2026
**主题**: llm_serving / KV Cache Optimization

## 摘要 (Abstract)

Large language models (LLMs) have shown great performance on complex reasoning tasks but often require generating long intermediate thoughts before reaching a final answer. During generation, LLMs rely on a key-value (KV) cache for autoregressive decoding. However, the memory footprint of the KV cache grows with output length. Prior work on KV cache optimization mostly focus on compressing the long input context, while retaining the full KV cache for decoding. For tasks requiring long output generation, this leads to increased computational and memory costs. In this paper, we introduce ZoomR, a novel approach that enables LLMs to adaptively compress verbose reasoning thoughts into summaries and uses a dynamic KV cache selection policy that leverages these summaries while also strategically "zooming in" on fine-grained details. By using summary keys as a coarse-grained index during decoding, ZoomR uses the query to retrieve details for only the most important thoughts. This hierarchical strategy significantly reduces memory usage by avoiding full-cache attention at each step. Experiments across math and reasoning tasks show that our approach achieves competitive performance compared to baselines, while reducing inference memory requirements by more than 4x. These results demonstrate that a multi-granularity KV selection enables more memory efficient decoding, especially for long output generation.

## 摘要 (中文)

大型语言模型在复杂推理任务上表现出色，但通常需要在得出最终答案之前生成冗长的中间思考过程。在生成过程中，LLMs 依赖于 KV cache 进行自回归解码，但其内存占用随输出长度增长。之前的 KV cache 优化工作主要集中于压缩长输入上下文，而在解码时保留完整的 KV cache。对于需要长输出生成的任务，这会导致计算和内存成本增加。本文提出 ZoomR，一种新颖的方法，使 LLMs 能够自适应地将冗长的推理思考压缩为摘要，并使用动态 KV cache 选择策略，利用这些摘要同时策略性地"聚焦"细粒度细节。通过在解码时使用摘要键作为粗粒度索引，ZoomR 仅检索最重要的思考细节。这种分层策略通过避免每一步的全缓存注意力显著减少了内存使用。实验表明，该方法在数学和推理任务上实现了与基线相当的性能，同时将推理内存需求降低了 4 倍以上。

## 引言/核心思路

ZoomR 的核心创新在于多粒度 KV cache 选择机制：在解码阶段，利用摘要级别的粗粒度索引快速定位关键信息，然后仅对这些关键部分"zoom in"到细粒度细节。这避免了传统方法每步都需要全缓存注意力的开销，特别适合长输出推理场景（如 chain-of-thought）。

## 关键贡献

1. 提出自适应推理思考压缩机制
2. 动态 KV cache 选择策略：粗粒度摘要索引 + 细粒度选择性检索
3. 内存需求降低 4x+，性能保持竞争力