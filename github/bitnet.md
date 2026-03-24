# BitNet: Official Inference Framework for 1-bit LLMs

## 项目信息

- **GitHub**: [microsoft/BitNet](https://github.com/microsoft/BitNet)
- **Stars**: 36.4k+
- **组织**: Microsoft Research
- **许可证**: MIT

## 简介

BitNet是微软发布的1-bit大语言模型推理框架，支持BitNet-b1.58等1-bit模型的高效推理，大幅降低计算和内存开销。

## 主要特性

1. **1-bit推理支持**
   - 1.58-bit权重支持
   - 极端量化下的精度保持
   - 显著降低内存占用

2. **高效推理**
   - 针对1-bit运算优化
   - 硬件利用率高
   - 低延迟输出

3. **模型生态**
   - BitNet-b1.58 (LLaMA架构)
   - 预训练模型可直接使用
   - 与HuggingFace兼容

4. **框架支持**
   - Python API
   - C++运行时
   - 易于集成到现有系统

## 性能数据

- 内存减少: ~10x
- 计算量减少: ~5x
- 延迟降低: ~2-3x

## 技术细节

### 1-bit量化
- 权重仅为{-1, 0, +1}（1.58-bit变体）
- 可使用整数运算代替浮点运算
- 大幅降低带宽需求

### 硬件优化
- 专用CUDA内核
- 高效的矩阵运算

## 适用场景

- 大模型边缘部署
- 低资源环境
- 成本敏感的应用
- 移动设备推理

---

*更新时间: 2026-03-24*