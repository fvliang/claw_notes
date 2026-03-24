# Mooncake: Kimi的LLM服务平台

## 原文链接
- GitHub: https://github.com/kvcache-ai/Mooncake
- Stars: 5k
- 技术报告: https://arxiv.org/abs/2407.00079

## 概述
Mooncake是Kimi的LLM服务平台，由Moonshot AI（Moonshot AI）提供支持。Mooncake采用了以KVCache为中心的解聚架构，将prefill（预填充）和decoding（解码）集群分离。它还利用了GPU集群中未充分利用的CPU、DRAM和SSD资源来实现KVCache的缓存和预取。

## 核心特性

### KVCache中心化架构
- 将prefill和decoding集群解聚
- 利用CPU、DRAM和SSD资源进行KVCache缓存和预取
- 实现高效的KVCache传输

### Transfer Engine（传输引擎）
- Mooncae的核心组件
- 支持高性能P2P存储
- 已集成到vLLM、TensorRT-LLM、SGLang等主流推理框架

### Mooncake Store（分布式KVCache存储）
- 基于Transfer Engine的分布式KVCache
- 支持多层次的KVCache存储（设备、主机、远程存储）
- 与vLLM、SGLang、LMDeploy等深度集成

### PD解聚（Prefill-Disaggregation）
- 支持Prefill和Decoding的分离部署
- 实现大规模专家并行
- 支持多节点推理

## 重要更新

### 2026年
- [2026/03] TorchSpec开源，使用Mooncake进行高效的隐藏状态管理，解耦推理和训练
- [2026/03] LightX2V支持基于Mooncake的解聚部署
- [2026/02] SGLang引入基于Mooncake的编码器全局缓存管理器
- [2026/02] vLLM-Omni引入解聚推理连接器，支持Mooncake
- [2026/02] Mooncake正式加入PyTorch生态系统
- [2026/01] FlexKV支持与Mooncake Transfer Engine的分布式KVCache复用

### 2025年
- [2025/12] SGLang引入Encode-Prefill-Decode (EPD)解聚
- [2025/12] Mooncake Transfer Engine集成到TensorRT-LLM
- [2025/12] Mooncake Transfer Engine直接集成到vLLM v1作为KV Connector
- [2025/09] SGLang正式支持Mooncake Store作为分层KV缓存存储后端
- [2025/08] xLLM高性能推理引擎基于Mooncake构建混合KVCache管理
- [2025/07] Mooncake支持Kimi K2在128个H200 GPU上的部署，实现224k tokens/s预填充吞吐量和288k tokens/s解码吞吐量
- [2025/06] Mooncake成为LMDeploy的PD解聚后端
- [2025/04] LMCache正式支持Mooncake Store作为远程连接器
- [2025/03] 开源Mooncake Store

## 获得的荣誉
- **FAST 2025最佳论文奖**

## 集成和合作

Mooncake已与多个主流框架深度集成：
- **vLLM**: KV Connector和PD解聚支持
- **TensorRT-LLM**: KVCache传输
- **SGLang**: 分层KV缓存和PD解聚
- **LMDeploy**: PD解聚后端
- **LMCache**: 远程连接器
- **NIXL**: 后端插件
- **FlexKV**: 分布式KVCache复用
- **vLLM-Ascend**: 华为昇腾NPU支持
- **DeepSeek**: PD解聚部署支持

## 性能数据
- Kimi K2（1T参数）部署：
  - 128个H200 GPU
  - PD解聚和大规模专家并行
  - 预填充吞吐量：224k tokens/s
  - 解码吞吐量：288k tokens/s

## 技术特点

1. **分层KVCache存储**
   - 设备层（GPU显存）
   - 主机层（CPU DRAM）
   - 远程存储层（SSD）

2. **高效传输**
   - RDMA引擎
   - 零拷贝传输大型多模态嵌入

3. **全局缓存**
   - 跨实例共享ViT嵌入
   - 避免冗余GPU计算

## 安装

```bash
git clone https://github.com/kvcache-ai/Mooncake
cd Mooncake
pip install -r requirements.txt
```

## 相关链接
- 技术报告：https://arxiv.org/abs/2407.00079
- 中文博客：https://zhuanlan.zhihu.com/p/705754254
- vLLM集成文档：https://docs.vllm.ai/en/latest/features/mooncake_connector_usage/
- SGLang集成：https://lmsys.org/blog/2025-09-10-sglang-hicache/

---

*本文档由自动化任务生成于 2026-03-24*