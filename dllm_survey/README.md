# D-LLM Survey 笔记

## 核心内容
- **主题**: 分布式大语言模型（D-LLM）技术综述
- **来源**: [Fuliang Liu 的2025年调查报告](https://fvliang.github.io)
- **资料**: [完整演示文稿](images/dllmsurvey2025.pptx)

## 关键技术点
1. **分布式训练架构**
   - 数据并行 vs 模型并行
   - 流水线并行优化
2. **推理优化**
   - KV缓存机制（见下图）
   - 注意力掩码策略
3. **通信效率**
   - 梯度压缩技术
   - 异步更新策略

## 技术图示
### KV缓存机制
![KV缓存](images/kvcache.png)

### 训练流程
![训练流程](images/train.png)

### 推理流程
![推理流程](images/inference.png)

> 注：图片需从原始PDF转换，当前仅保留PPT文件