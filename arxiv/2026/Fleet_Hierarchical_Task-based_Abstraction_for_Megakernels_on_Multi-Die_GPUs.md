# Fleet: Hierarchical Task-based Abstraction for Megakernels on Multi-Die GPUs

- **Arxiv ID**: 2604.15379
- **Conference**: arxiv 2026
- **Link**: https://arxiv.org/abs/2604.15379
- **GitHub**: 
- **Tags**: megakernel, chiplet-gpu, llm-inference, persistent-kernel

## Abstract (English)

Modern GPUs adopt chiplet-based designs with multiple private cache hierarchies, but current programming models (CUDA/HIP) expose a flat execution hierarchy that cannot express chiplet-level locality or synchronization. Fleet presents a multi-level task model that maps computation to memory scopes, introducing Chiplet-tasks—a new abstraction that binds work and data to a chiplet and enables coordination through its shared L2 cache. On AMD Instinct MI350 with Qwen3-8B, Fleet achieves 1.3–1.5× lower decode latency than vLLM at batch sizes 1–8 through persistent kernel execution and per-chiplet scheduling. At larger batch sizes, cooperative weight tiling increases L2 hit rate (from 12% to 54% at batch 32), reducing HBM traffic by up to 37% and delivering 1.27–1.30× speedup over a chiplet-unaware megakernel baseline.

## Abstract (Chinese)

现代GPU采用chiplet设计，具有多个私有缓存层次，但当前编程模型(CUDA/HIP)暴露的是扁平执行层次，无法表达chiplet级别的局部性或同步。Fleet提出了一个多层任务模型，将计算映射到内存范围，引入Chiplet-task——一种新的抽象，将工作和数据绑定到chiplet并通过共享L2缓存实现协调。在AMD Instinct MI350上运行Qwen3-8B时，Fleet通过持久内核执行和chiplet级调度，在batch 1-8下比vLLM实现1.3-1.5倍更低的解码延迟。在大batch下，协作权重分块将L2命中率从12%提升至54%，减少37%的HBM流量。

## Introduction (English)

Modern GPU architectures have moved toward chiplet-based, multi-die designs for improving compute density. AMD's Instinct MI300X and MI350/MI355 implement eight XCDs with private 4MB L2 caches while NVIDIA's Blackwell integrates two dies on a single package. Yet the CUDA/HIP execution model has not adapted to this architectural shift. There is no direct way to express data affinity between groups of workgroups, or to scope work to a specific chiplet's memory hierarchy. This gap is especially acute for LLM inference, where interactive chatbots, coding assistants, and voice agents demand sub-10ms per-token decode, a regime dominated by memory bandwidth. Fleet presents a multi-level task model that maps operators to memory scopes matching their working sets.

## Introduction (Chinese)

现代GPU架构已转向chiplet多芯片设计以提高计算密度。AMD的MI300X和MI350实现8个XCD（各有私有4MB L2缓存），NVIDIA的Blackwell在单个封装上集成两个芯片。然而CUDA/HIP执行模型尚未适应这一架构转变。无法直接表达工作组间的数据亲和性或将工作限定到特定chiplet的内存层次。这对LLM推理尤为关键——交互式聊天机器人、编程助手和语音代理要求低于10ms的每token解码延迟。Fleet提出了一个多层任务模型，将算子映射到匹配其工作集的内存范围。

## GitHub Introduction

N/A - No GitHub repository found for this paper.

## Blog Content

N/A - No blog post found for this paper.

---
*Auto-collected on 2026-04-21*
