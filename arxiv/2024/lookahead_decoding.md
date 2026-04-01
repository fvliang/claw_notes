# Lookahead Decoding: Break the Sequential Dependency of LLM Inference

## 基本信息

- **标题**: Lookahead Decoding: Break the Sequential Dependency of LLM Inference Using Lookahead Decoding
- **作者**: (待补充)
- **arXiv**: (待补充)
- **会议**: ICML 2024
- **GitHub**: [hao-ai-lab/LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding)
- **Stars**: 1325

## 摘要 (Abstract)

Large Language Models (LLMs) generate tokens auto-regressively, which creates sequential dependency that limits parallelization during inference. This paper presents Lookahead Decoding, a technique that breaks this sequential dependency to enable parallel token generation.

## 摘要 (中文)

大型语言模型(LLM)以自回归方式生成token,这造成了顺序依赖性,限制了推理期间的并行化。本文提出了Lookahead Decoding技术,打破这种顺序依赖以实现并行token生成。

## 引言 (Introduction)

自回归解码是LLM生成的主要范式,但其顺序依赖性导致了推理延迟高、吞吐量低的问题。现有的加速技术如speculative decoding需要在draft和target模型之间切换,而Lookahead Decoding通过在单个模型内部实现并行化来解决问题。

## 原文链接

- arXiv: (待补充)
- GitHub: https://github.com/hao-ai-lab/LookaheadDecoding

## 相关博客

- (待补充)

---