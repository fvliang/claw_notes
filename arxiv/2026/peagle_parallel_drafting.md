# P-EAGLE: Parallel-Drafting EAGLE with Scalable Training

## 论文信息
- **标题**: P-EAGLE: Parallel-Drafting EAGLE with Scalable Training
- **作者**: Mude Hui, Xin Huang, Jaime Campos Salas, Yue Sun, Nathan Pemberton, Xiang Song, Ashish Khetan, George Karypis
- **arXiv**: [2602.01469](https://arxiv.org/abs/2602.01469)
- **提交时间**: 2026年2月1日
- **领域**: Machine Learning (cs.LG), Artificial Intelligence (cs.AI)

## 摘要 (Abstract)
Reasoning LLMs produce longer outputs, requiring speculative decoding drafters trained on extended sequences. Parallel drafting - predicting multiple tokens per forward pass - offers latency benefits over sequential generation, but training complexity scales quadratically with the product of sequence length and parallel positions, rendering long-context training impractical. We present P(arallel)-EAGLE, which transforms EAGLE from autoregressive to parallel multi-token prediction via a learnable shared hidden state. To scale training to long contexts, we develop a framework featuring attention mask pre-computation and sequence partitioning techniques, enabling gradient accumulation within individual sequences for parallel-prediction training. We implement P-EAGLE in vLLM and demonstrate speedups of 1.10-1.36x over autoregressive EAGLE-3 across GPT-OSS 120B, 20B, and Qwen3-Coder 30B.

## 摘要 (中文)
推理LLM产生更长的输出，需要在长序列上训练的投机解码drafters。并行drafting——每次前向传播预测多个tokens——比顺序生成提供延迟优势，但训练复杂度随序列长度和并行位置的乘积呈二次方增长，使得长上下文训练不切实际。我们提出了P(arallel)-EAGLE，它通过可学习的共享隐藏状态将EAGLE从自回归转变为并行多token预测。为了将训练扩展到长上下文，我们开发了一个具有注意力掩码预计算和序列分区技术的框架，使梯度能够在单个序列内积累，用于并行预测训练。我们在vLLM中实现了P-EAGLE，并在GPT-OSS 120B、20B和Qwen3-Coder 30B上展示了比自回归EAGLE-3高1.10-1.36倍的加速。

## 核心贡献
1. **并行多token预测**: 将EAGLE从自回归转变为并行多token预测
2. **可学习共享隐藏状态**: 通过可学习的共享隐藏状态实现并行预测
3. **长上下文训练**: 开发了注意力掩码预计算和序列分区技术，支持长上下文训练

## 技术细节
- **实现**: 在vLLM中实现
- **性能**: 在GPT-OSS 120B、20B和Qwen3-Coder 30B上比EAGLE-3快1.10-1.36倍

---

*更新时间: 2026-03-25*