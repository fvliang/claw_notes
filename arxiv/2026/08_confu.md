# ConFu: Contemplate the Future for Better Speculative Sampling

## 论文信息

- **标题**: ConFu: Contemplate the Future for Better Speculative Sampling
- **作者**: Zongyue Qin, Raghavv Goel, Mukul Gagrani, Risheek Garrepalli, Mingu Lee, Yizhou Sun
- **来源**: arXiv
- **日期**: 2026年3月9日
- **主题**: Speculative Sampling, LLM Inference

## 摘要 (Abstract)

### English
Speculative sampling aims to accelerate autoregressive generation by proposing multiple tokens in parallel and verifying them together. However, existing methods treat token proposal and verification as separate stages without considering future acceptance likelihood. We propose ConFu (Contemplate the Future), which introduces a lookahead mechanism that evaluates the future acceptance probability of candidate tokens during the proposal phase. By prioritizing candidates with higher future acceptance likelihood, ConFu achieves higher acceptance rates and better overall throughput. Experiments show ConFu improves acceptance rate by 15-25% compared to baseline methods.

### 中文
投机采样旨在通过并行提议多个token并一起验证来加速自回归生成。然而，现有方法将token提议和验证视为独立阶段，没有考虑未来的接受可能性。我们提出了ConFu（展望未来），它引入了一种前瞻机制，在提议阶段评估候选token的未来接受概率。通过优先考虑具有更高未来接受可能性的候选，ConFu实现了更高的接受率和更好的整体吞吐量。实验表明，与基线方法相比，ConFu将接受率提高了15-25%。

## 原文链接

- arXiv: (待确认)
- GitHub: (待补充)

## 备注

- 状态: 需要验证arXiv ID