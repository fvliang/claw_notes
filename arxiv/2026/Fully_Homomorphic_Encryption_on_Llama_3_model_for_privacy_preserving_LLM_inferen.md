# Fully Homomorphic Encryption on Llama 3 model for privacy preserving LLM inference

- **Arxiv ID**: 2604.12168
- **Conference**: arxiv 2026
- **Link**: https://arxiv.org/abs/2604.12168
- **GitHub**: 
- **Tags**: fhe, privacy-preserving, llm-inference, post-quantum

## Abstract (English)

We integrate Post-Quantum Cryptography based Lattice-based Homomorphic Encryption (HE) main functions in the LLM's inference pipeline to secure some of its layers against data privacy attacks. We modify the inference pipeline of the transformer architecture for the LLAMA-3 model while injecting homomorphic encryption operations provided by the concrete-ml library. We demonstrate high text generation accuracies (up to 98%) with reasonable latencies (237 ms) on an i9 CPU, reaching up to 80 tokens per second, proving the feasibility of running a FHE-secured LLAMA-3 inference model.

## Abstract (Chinese)

我们将基于后量子密码学的格同态加密(HE)主要功能集成到LLM推理管线中，以保护某些层免受数据隐私攻击。我们修改了LLAMA-3模型的Transformer架构推理管线，注入了由concrete-ml库提供的同态加密操作。在i9 CPU上展示了高文本生成准确率(高达98%)和合理延迟(237ms)，达到80 tokens/秒，证明了FHE安全LLAMA-3推理模型的可行性。

## Introduction (English)

The proliferation of LLM-dependent applications in fields such as healthcare, finance, and transportation raises serious concerns regarding data privacy. Processing data in plain (not encrypted) form within LLMs poses a great risk of breaching users' personal and sensitive data. Attackers can use sophisticated techniques including prompt injection, jailbreaks, model and data poisoning, and data extraction. A promising direction is to use encryption algorithms to secure data at different levels and stages. In this work, we focus on securing the LLM's inference stage by leveraging post-quantum algorithms in fully homomorphic encryption (FHE) to secure the inference process of Llama-3.

## Introduction (Chinese)

LLM在医疗、金融和交通等领域的广泛应用引发了严重的数据隐私担忧。在LLM中以明文形式处理数据存在泄露用户个人和敏感数据的重大风险。攻击者可以使用提示注入、越狱、模型和数据投毒以及数据提取等复杂技术。一个有前景的方向是使用加密算法在不同级别和阶段保护数据。在本工作中，我们专注于通过利用全同态加密(FHE)中的后量子算法来保护Llama-3的推理过程。

## GitHub Introduction

N/A - No GitHub repository found for this paper.

## Blog Content

N/A - No blog post found for this paper.

---
*Auto-collected on 2026-04-21*
