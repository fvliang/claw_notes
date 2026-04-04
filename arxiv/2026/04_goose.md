# Goose: Anisotropic Speculation Trees for Training-Free Speculative Decoding

## 论文信息

- **标题**: Goose: Anisotropic Speculation Trees for Training-Free Speculative Decoding
- **来源**: arXiv
- **日期**: 2026年4月2日
- **主题**: Speculative Decoding, LLM Inference

## 摘要 (Abstract)

### English
Speculative decoding accelerates large language model inference by drafting multiple candidate tokens and verifying them in a single forward pass. Candidates are organized as a tree: deeper trees accept more tokens per step, but adding depth requires sacrificing breadth (fallback options) under a fixed verification budget. We propose Goose, which introduces anisotropic speculation trees that adapt tree structure based on local token confidence. Goose achieves better acceptance rates by allocating more candidates to high-uncertainty positions while maintaining efficiency through adaptive tree pruning. Our experiments show that Goose achieves 1.8x speedup over baseline speculative decoding without any additional training.

### 中文
投机解码通过起草多个候选token并在单次前向传播中验证它们来加速大型语言模型推理。候选被组织成树状结构：更深的树每步接受更多token，但在固定验证预算下增加深度需要牺牲广度（回退选项）。我们提出了Goose，它引入了各向异性投机树，根据局部token置信度来适应树结构。Goose通过在高度不确定的位置分配更多候选来实现更好的接受率，同时通过自适应树剪枝保持效率。我们的实验表明，Goose在没有任何额外训练的情况下实现了比基线投机解码1.8倍的加速。

## 引言 (Introduction)

### English
Speculative decoding has become a key technique for accelerating LLM inference. The core idea is to use a smaller "draft" model to predict multiple tokens in parallel, which are then verified by the larger "target" model. However, traditional approaches use a fixed tree structure that doesn't adapt to the input.

Goose introduces anisotropy into speculation trees, meaning that the tree structure varies based on the local context. High-confidence tokens get fewer candidates (narrower branches), while uncertain positions get more candidates (wider branches).

### 中文
投机解码已成为加速LLM推理的关键技术。其核心思想是使用较小的"起草"模型并行预测多个token，然后由较大的"目标"模型验证。然而，传统方法使用固定的树结构，不能适应输入。

Goose将各向异性引入投机树，意味着树结构根据局部上下文变化。高置信度token获得更少候选（更窄的分支），而不确定的位置获得更多候选（更宽的分支）。

## 原文链接

- arXiv: (待确认)
- GitHub: (待补充)

## 补充材料

- 博客: (待补充)
- 相关GitHub: (待补充)

## 备注

- 状态: 需要验证arXiv ID
- 需要补充完整的GitHub链接和博客内容