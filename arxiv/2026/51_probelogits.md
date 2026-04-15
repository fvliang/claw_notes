---
title: ProbeLogits: Kernel-Level LLM Inference Primitives for AI-Native Operating Systems
authors: Daeyeon Son
arxiv_id: 
conference: arxiv
full_conference: ARXIV 2026
year: "2026"
topic: LLM Serving
url: 
pdf_url: 
added_date: 2026-04-15
---

# ProbeLogits: Kernel-Level LLM Inference Primitives for AI-Native Operating Systems

## 论文信息

- **arXiv**: 
- **会议**: ARXIV 2026
- **作者**: Daeyeon Son
- **主题**: LLM Serving

## 摘要 (Abstract)

An OS kernel that runs LLM inference internally can read logit distributions before any text is generated and act on them as a governance primitive. We present ProbeLogits, a kernel-level operation that performs a single forward pass and reads specific token logits to classify agent actions as safe or dangerous, with zero learned parameters. This approach enables real-time safety checks at the inference level without additional model overhead.

## 摘要中文

在内部运行LLM推理的操作系统内核可以在任何文本生成之前读取logit分布，并将其作为治理原语行动。我们提出了ProbeLogits，一种内核级操作，执行单次前向传播并读取特定token logit以将代理行动分类为安全或危险，无需学习参数。这种方法在推理级别实现实时安全检查，无需额外模型开销。

## 引言 (Introduction)

As LLMs become embedded in operating systems as native services, new primitives emerge for inference-level governance. ProbeLogits explores the concept of reading logits directly from the model's forward pass for safety classification.

## 引言中文

随着LLM作为原生服务嵌入操作系统，出现了推理级治理的新原语。ProbeLogits探索了直接从模型前向传播读取logit进行安全分类的概念。

## 主要贡献

1. (待补充)

## 原文链接

- arXiv: (待确认)
- GitHub: (待补充)

## 补充材料

- 博客: (待补充)
- 相关GitHub: (待补充)

## 备注
