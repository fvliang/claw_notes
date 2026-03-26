# PowerInfer

## 项目信息

- **项目名称**: PowerInfer
- **GitHub**: https://github.com/Tiiny-AI/PowerInfer
- **Stars**: 9.1k+
- **语言**: C++
- **维护者**: SJTU-IPADS

## 简介

PowerInfer是一个高速大语言模型本地部署推理引擎，利用激活局部性在消费级设备上运行LLM。

## 核心创新

PowerInfer的核心设计利用LLM推理中固有的高局部性，表现为神经元激活的幂律分布。

- **热神经元 (Hot Neurons)**: 在不同输入中持续激活的一小部分神经元
- **冷神经元 (Cold Neurons)**: 根据特定输入变化的多数神经元

PowerInfer利用这一洞察设计GPU-CPU混合推理引擎：
- 热神经元预加载到GPU以实现快速访问
- 冷神经元在CPU上计算
- 大幅减少GPU内存需求和CPU-GPU数据传输

## 主要特性

### 高性能
- 利用稀疏激活和"热/冷"神经元概念的高效推理
- 混合CPU/GPU利用：平衡工作负载和更快处理
- 自适应预测器和神经元感知稀疏算子

### 灵活性
- 兼容流行的ReLU稀疏模型
- 为消费级硬件深度优化
- 支持AMD ROCm
- 支持Windows GPU推理

## 性能数据

在单张RTX 4090 (24GB)上运行Falcon(ReLU)-40B-FP16：
- 平均token生成速率: 13.20 tokens/s
- 峰值: 29.08 tokens/s
- 比llama.cpp快 **11.69倍**
- 仅比顶级服务器级A100 GPU低18%

## 项目版本

### PowerInfer-2 (2024/6)
专为智能手机优化的高推理框架
- 使用TurboSparse-Mixtral-47B达到11.68 tokens/s
- 比其他SOTA框架快达22倍

### SmallThinker (2025/7)
- SmallThinker-21BA3B-Instruct
- SmallThinker-4BA0.6B-Instruct

### Bamboo LLM (2024/3)
- 高性能高速度的LLM

### Turbo Sparse (2024/6)
- 稀疏化Mistral和Mixtral模型至约90%稀疏度
- Mixtral级模型仅激活4B参数

## 安装

```bash
git clone https://github.com/Tiiny-AI/PowerInfer.git
cd PowerInfer
pip install -r requirements.txt
```

## 在线演示

- [Gradio Demo](https://powerinfer-gradio.vercel.app/) - 运行Falcon(ReLU)-40B-FP16

## 相关资源

- [论文](https://arxiv.org/abs/2406.06282)
- [TurboSparse论文](https://arxiv.org/abs/2406.05955)
- [HuggingFace模型](https://huggingface.co/PowerInfer)
- [Tiiny AI Pocket Lab](https://tiiny.ai/) - 世界首款口袋超算