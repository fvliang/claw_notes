# LoRAX: Multi-LoRA Inference Server

## 基本信息

- **仓库**: [predibase/lorax](https://github.com/predibase/lorax)
- **描述**: Multi-LoRA inference server that scales to 1000s of fine-tuned LLMs
- **语言**: Python
- **Stars**: 3741
- **更新时间**: 2026-04-01

## 主要特性

- **Multi-LoRA支持**: 支持同时运行数千个LoRA适配器
- **高性能**: 针对大规模LoRA推理优化
- **动态加载**: 运行时动态加载和卸载LoRA权重
- **多模型支持**: 支持多种基础模型

## 原文链接

- GitHub: https://github.com/predibase/lorax

## 介绍

LoRAX (Large Language Model LoRA eXchange)是Predibase推出的多LoRA推理服务器,专门针对大规模LoRA微调模型的部署场景设计。在实际生产环境中,企业往往需要同时运行多个针对不同任务微调的LLM,LoRAX通过共享基础模型权重的方式,大幅降低了多模型部署的成本。

---