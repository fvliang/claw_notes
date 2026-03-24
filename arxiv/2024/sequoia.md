# Sequoia: Tree-Based Speculative Decoding

## 论文信息
- **作者**: Infini-AI-Lab
- **会议**: arXiv 2024
- **GitHub**: https://github.com/Infini-AI-Lab/Sequoia
- **日期**: 2024

## 摘要 (Abstract)
Sequoia presents a scalable and robust tree-based speculative decoding algorithm. Key contributions:

1. **Efficient tree construction**: Optimal draft tree structure
2. **Robust verification**: Handles various acceptance scenarios
3. **Adaptive strategy**: Adjusts to different workload characteristics
4. **Significant speedups**: Up to 2.5x over baseline methods

## 摘要中文
Sequoia提出了一种可扩展的基于树的鲁棒投机解码算法。主要贡献：

1. **高效的树构建**: 最优的draft树结构
2. **鲁棒的验证**: 处理各种接受场景
3. **自适应策略**: 适应不同的工作负载特征
4. **显著加速**: 比基线方法快2.5倍

## 引言 (Introduction)
Tree-based speculative decoding has shown promise but faces challenges:
- Tree construction overhead
- Verification complexity
- Scalability to large trees

Sequoia addresses these with:
- **Optimal tree algorithms**: Minimizes expected verification cost
- **Efficient verification**: Parallel verification of tree nodes
- **Adaptive branching**: Dynamic tree size based on acceptance rates

## GitHub 介绍
Sequoia: scalable and robust tree-based speculative decoding algorithm. Provides efficient implementation for high-performance LLM inference.