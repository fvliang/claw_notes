# LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding

## 基本信息

- **标题**: LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding
- **作者**: (待补充)
- **arXiv**: (待补充)
- **会议**: ACL 2024
- **GitHub**: [facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip)
- **Stars**: 363

## 摘要 (Abstract)

LayerSkip combines early exit inference with self-speculative decoding, allowing LLMs to dynamically skip layers during inference based on sample difficulty, while using the same model for both draft generation and verification.

## 摘要 (中文)

LayerSkip将早期退出推理与自推测解码相结合,允许LLM根据样本难度在推理期间动态跳过层,同时使用同一模型进行draft生成和验证。

## 引言 (Introduction)

LayerSkip是Facebook Research提出的创新方法,它利用Transformer的自回归特性,让模型在处理简单样本时提前退出,在处理复杂样本时使用自推测解码进行加速。这种方法无需额外的draft模型,简化了部署复杂度。

## 原文链接

- arXiv: (待补充)
- GitHub: https://github.com/facebookresearch/LayerSkip
- Paper: ACL 2024

---