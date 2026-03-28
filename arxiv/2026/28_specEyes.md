# SpecEyes: Accelerating Agentic Multimodal LLMs via Speculative Perception and Planning

## 论文信息

- **作者**: Haoyu Huang, Jinfa Huang, Zhongwei Wan, Xiawu Zheng, Rongrong Ji, Jiebo Luo
- **arXiv**: [2603.23483](https://arxiv.org/abs/2603.23483)
- **提交日期**: 2026年3月24日
- **领域**: Computer Vision and Pattern Recognition (cs.CV); Computation and Language (cs.CL); Machine Learning (cs.LG)

## 摘要 (Abstract)

Agentic multimodal large language models (MLLMs) (e.g., OpenAI o3 and Gemini Agentic Vision) achieve remarkable reasoning capabilities through iterative visual tool invocation. However, the cascaded pipeline of tool use introduces significant latency, making real-time deployment challenging. To address this, we propose SpecEyes, a novel speculative perception and planning framework for agentic MLLMs. Specifically, we first reveal a stateful bottleneck in current agentic MLLMs: the sequential tool-use dependency limits both latency and concurrency. Based on this insight, we propose speculative reasoning that can skip full tool invocation loops for easy queries. To decide whether to bypass the tool-using model, we introduce answer separability gating, a new confidence metric based on top-K logit gaps. Extensive experiments on multiple benchmarks (including V* Bench, HR-Bench, POPE, DeepEyes, and Thyme) demonstrate that SpecEyes can achieve up to 2.9× speedup while maintaining comparable accuracy.

## 摘要 (中文)

代理多模态大语言模型(MLLM)(如OpenAI o3和Gemini Agentic Vision)通过迭代视觉工具调用实现了卓越的推理能力。然而，工具使用的级联管道引入了显著的延迟，使其难以实现实时部署。为了解决这个问题，我们提出了SpecEyes，一种用于代理MLLM的新型投机感知和规划框架。具体来说，我们首先揭示了当前代理MLLM中的一个有状态瓶颈：顺序的工具有用依赖限制了延迟和并发性。基于这一见解，我们提出了投机推理，可以跳过简单查询的完整工具调用循环。为了决定是否绕过工具使用模型，我们引入了答案可分离门控，这是一种基于top-K logit间隙的新置信度指标。在多个基准测试(包括V* Bench、HR-Bench、POPE、DeepEyes和Thyme)上的广泛实验表明，SpecEyes可以在保持相当准确度的同时实现高达2.9倍的加速。

## 引言 (Introduction)

代理多模态大语言模型(MLLM)通过迭代调用视觉工具来实现复杂的推理能力。然而，这种级联管道存在严重的延迟问题，限制了实时部署的可能性。

本文的主要贡献：
1. 揭示了当前代理MLLM中的有状态瓶颈
2. 提出投机推理框架，可以跳过简单查询的完整工具调用
3. 引入答案可分离门控机制，基于top-K logit间隙判断是否绕过工具使用模型

## GitHub

- **官方仓库**: [MAC-AutoML/SpecEyes](https://github.com/MAC-AutoML/SpecEyes)

### 核心特性

- **投机感知与规划**: 使用轻量级VLM快速筛选视觉输入和问题
- **答案可分离门控**: 基于top-K logit间隙的置信度指标
- **状态瓶颈分析**: 揭示代理MLLM中顺序工具使用依赖问题

### 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# Deepeyes baseline
python eval_code_deepeyes/SpecEyes.py --baseline

# 带置信度门控
python eval_code_deepeyes/SpecEyes.py --score_threshold 0.98
```

## 博客

暂无公开博客。