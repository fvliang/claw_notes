---
title: Blink: CPU-Free LLM Inference by Delegating the Serving Stack to GPU and SmartNIC
authors: Mohammad Siavashi, Mariano Scazzariello, Gerald Q. Maguire Jr., Dejan Kostić, Marco Chiesa
arxiv_id: 
conference: arxiv
full_conference: ARXIV 2026
year: "2026"
topic: LLM Serving
url: 
pdf_url: 
added_date: 2026-04-15
---

# Blink: CPU-Free LLM Inference by Delegating the Serving Stack to GPU and SmartNIC

## 论文信息

- **arXiv**: 
- **会议**: ARXIV 2026
- **作者**: Mohammad Siavashi, Mariano Scazzariello, Gerald Q. Maguire Jr., Dejan Kostić, Marco Chiesa
- **主题**: LLM Serving

## 摘要 (Abstract)

Large Language Model (LLM) inference is rapidly becoming a core datacenter service, yet current serving systems heavily rely on the CPU for orchestrating the inference pipeline. We present Blink, a system that delegates the entire LLM serving stack—including scheduling, memory management, and network communication—to the GPU and SmartNIC, eliminating CPU involvement in the inference pipeline. This approach enables tighter integration between compute and communication, reducing overhead and improving throughput.

## 摘要中文

大语言模型推理正迅速成为数据中心核心服务，然而当前服务系统严重依赖CPU来协调推理流水线。我们提出了Blink，一个将整个LLM服务栈（包括调度、内存管理和网络通信）委托给GPU和SmartNIC的系统，消除了CPU在推理流水线中的参与。这种方法使计算和通信之间更紧密集成，减少开销并提高吞吐量。

## 引言 (Introduction)

Current LLM serving systems like vLLM and TensorRT-LLM rely on the CPU for critical orchestration tasks, including request scheduling, KV cache management, and inter-GPU communication. This CPU-centric design introduces overhead that limits achievable throughput, especially at high batch sizes.

## 引言中文

当前的LLM服务系统（如vLLM和TensorRT-LLM）依赖CPU执行关键协调任务，包括请求调度、KV缓存管理和GPU间通信。这种以CPU为中心的设计引入了限制可达到吞吐量的开销，特别是在高批量大小下。

## 主要贡献

1. (待补充)

## 原文链接

- arXiv: (待确认)
- GitHub: (待补充)

## 补充材料

- 博客: (待补充)
- 相关GitHub: (待补充)

## 备注
