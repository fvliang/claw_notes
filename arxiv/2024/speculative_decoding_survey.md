#投机解码 (Speculative Decoding) 研究综述

## 什么是投机解码?

投机解码是一种加速LLM推理的技术，通过使用一个较小的"draft"模型快速生成多个token候选，然后使用较大的"target"模型并行验证这些候选。

## 工作原理

1. **Draft阶段**: 小型draft模型自回归生成多个token
2. **Verify阶段**: 大型target模型并行验证draft的token
3. **接受**: 验证通过的token被接受
4. **拒绝**: 验证失败的token重新采样

## 优势

- **加速比**: 可实现2-3倍甚至更高的加速
- **无损**: 保证与原生自回归解码相同的输出
- **灵活**: 可使用不同大小的draft/target模型组合

## 变体

### 1. 自投机解码 (Self-Speculative Decoding)
使用同一模型既做draft又做verify

### 2. 树状投机解码 (Tree Speculative Decoding)
生成token树而非线性序列

### 3. 批处理投机解码 (Batch Speculative Decoding)
支持多个序列并行投机

### 4. 协作投机解码 (Collaborative SD)
边缘-云协作场景

## 相关项目

- vLLM内置投机解码
- FlashInfer: Chain speculative sampling
- SpecForge: 训练框架
- MineDraft: 批量并行

## 挑战

- Draft接受率
- 内存带宽
- 调度开销