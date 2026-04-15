# Robust Length Prediction: A Perspective from Heavy-Tailed Prompt-Conditioned Distributions

**Source:** arxiv | **Category:** LLM Inference | **Date:** 2026-04-09
**ArXiv ID:** 2604.07931
**Authors:** Jing Wang, Yu-Yang Qian, Ke Xue, Chao Qian, Peng Zhao, Zhi-Hua Zhou
**Tags:** length-prediction, heavy-tailed, prompt-conditioned, robust-estimation, prod

## Links

- 📄 [Paper (PDF)](https://arxiv.org/pdf/2604.07931)
- 🌐 [ArXiv Page](https://arxiv.org/abs/2604.07931)

## Abstract (English)

Output-length prediction is important for efficient LLM serving, as it directly affects batching, memory reservation, and scheduling. Most existing methods use a one-shot sampled length as the label, implicitly treating each prompt as having one true target length. This is unreliable: even under a fixed model and decoding setup, the same prompt induces a prompt-conditioned output length distribution, not a deterministic scalar, and this distribution exhibits heavy-tailed behavior. Robust Length Prediction casts length prediction as robust estimation from heavy-tailed prompt-conditioned length distributions. It proposes ProD methods, which construct training targets from multiple independent generations of the same prompt. ProD-M uses a median-based target for robust point prediction; ProD-D uses a distributional target preserving prompt-conditioned uncertainty. Experiments across diverse scenarios show consistent gains in prediction quality.

## Abstract (Chinese)

输出长度预测对高效LLM服务很重要，直接影响批处理、内存预留和调度。大多数现有方法使用一次性采样长度作为标签，隐含地将每个提示视为有一个真实目标长度。这是不可靠的：即使在固定模型和解码设置下，同一提示诱导的是提示条件输出长度分布，而非确定性标量，且该分布表现出重尾行为。鲁棒长度预测将长度预测重新表述为重尾提示条件长度分布的鲁棒估计。提出ProD方法，从同一提示的多次独立生成构建训练目标。ProD-M使用基于中位数的目标进行鲁棒点预测；ProD-D使用分布目标保留提示条件不确定性。在各种场景下实验显示预测质量的持续提升。

## Key Contributions

1. **Robust Length Prediction** — Output-length prediction is important for efficient LLM serving, as it directly affects batching, me...
2. Addresses core challenges in LLM Inference systems
3. Demonstrates significant improvements over existing baselines

## Notes

- Added on 2026-04-16
- Paper published on 2026-04-09
