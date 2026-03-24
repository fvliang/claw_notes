# PowerInfer: 本地部署高速LLM推理引擎

## 原文链接
- GitHub: https://github.com/Tiiny-AI/PowerInfer
- Stars: 9k
- 官方网站: https://tiiny.ai/

## 概述
PowerInfer是一款利用激活局部性的高速大型语言模型（LLM）推理引擎，专为个人电脑本地部署设计。它采用GPU-CPU混合架构，利用LLM推理中的高局部性特征（神经元激活的幂律分布）来实现高效推理。

## 核心设计理念

PowerInfer的核心设计利用了LLM推理中固有的高局部性，其特征是神经元激活呈幂律分布：

- **热神经元（Hot Neurons）**：跨输入持续激活的一小部分神经元
- **冷神经元（Cold Neurons）**：根据特定输入而变化的大部分神经元

PowerInfer利用这一洞察设计了GPU-CPU混合推理引擎：
- 热激活神经元预加载到GPU上以实现快速访问
- 冷激活神经元在CPU上计算
- 大大减少GPU内存需求和CPU-GPU数据传输

PowerInfer还集成了自适应预测器和神经元感知的稀疏算子，优化了神经元激活和计算稀疏性的效率。

## 主要特性

### 高速度
- **以局部性为中心的设计**：利用稀疏激活和"热"/"冷"神经元概念实现高效LLM推理，确保高速和低资源需求
- **混合CPU/GPU利用**：无缝集成CPU和GPU的内存/计算能力，实现平衡的工作负载和更快的处理

### 灵活易用
- **易于集成**：兼容流行的ReLU稀疏模型
- **本地部署便捷**：专为消费级硬件的本地部署而设计，实现单GPU上的低延迟LLM推理和服务
- **向后兼容**：与llama.cpp使用方式兼容（server和batch generation）

## 性能表现
- 在单张NVIDIA RTX 4090 GPU上，PowerInfer实现了：
  - 平均token生成速率：13.20 tokens/s
  - 峰值：29.08 tokens/s
- 比llama.cpp快**高达11.69倍**
- 对于OPT-175B等大型模型，仅比顶级服务器级A100 GPU低18%

## 支持的模型
- Falcon-40B
- Llama2 家族
- ProSparse Llama2 家族
- Bamboo-7B
- TurboSparse-Mixtral-47B（手机上可达11.68 tokens/s）

## 更新日志
- [2026/01] 发布Tiiny AI Pocket Lab，世上首款口袋超级计算机，可在本地以20 tokens/s运行GPT-OSS-120B (int4)
- [2025/07] 发布SmallThinker-21BA3B-Instruct和SmallThinker-4BA0.6B-Instruct
- [2024/06] 发布PowerInfer-2，专为智能手机设计的高优化推理框架
- [2024/06] 发布TurboSparse，将Mistral和Mixtral模型稀疏化到近90%稀疏度
- [2024/05] 支持AMD设备（ROCm）
- [2024/03] 发布Bamboo LLM，实现顶级性能和卓越速度
- [2024/01] 支持Windows GPU推理

## 支持的平台
- Linux：x86-64 CPU（AVX2）+ NVIDIA GPU
- Windows：x86-64 CPU（AVX2）+ NVIDIA GPU
- macOS：Apple M系列芯片（仅CPU）

## 技术细节

PowerInfer的关键技术包括：
1. **自适应预测器**：预测神经元激活模式
2. **神经元感知稀疏算子**：优化稀疏计算效率
3. **智能热/冷神经元划分**：平衡GPU和CPU工作负载
4. **动态FFN卸载**：根据实时激活模式动态调整

## 安装

```bash
git clone https://github.com/Tiiny-AI/PowerInfer
cd PowerInfer
pip install -r requirements.txt

# 使用CMake构建
# NVIDIA GPU:
cmake -S . -B build -DLLAMA_CUBLAS=ON
cmake --build build --config Release

# AMD GPU:
CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ cmake -S . -B build -DLLAMA_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build --config Release

# 仅CPU:
cmake -S . -B build
cmake --build build --config Release
```

## 在线演示
尝试在RTX 4090上托管Falcon(ReLU)-40B-FP16的[Gradio服务器](https://powerinfer-gradio.vercel.app/)（实验性）

---

*本文档由自动化任务生成于 2026-03-24*