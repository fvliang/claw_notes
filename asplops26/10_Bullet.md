# Bullet: Boosting GPU Utilization for LLM Serving through Spatial-Temporal GPU Resource Sharing

**论文链接**: [arXiv:2504.19516](https://arxiv.org/abs/2504.19516)

**作者**: Zejia Lin, Hongxin Xu, Guanyi Chen, Zhiguang Chen, Yutong Lu, Xianwei Zhang (中山大学)

**会议**: ASPLOS 2026

---

## Abstract (摘要)

Modern LLM serving systems confront inefficient GPU utilization due to the fundamental mismatch between compute-intensive prefill and memory-bound decode phases. While current practices attempt to address this by organizing these phases into hybrid batches, such solutions create an inefficient tradeoff that sacrifices either throughput or latency, leaving substantial GPU resources underutilized. We identify two key root causes: 1) the prefill phase suffers from suboptimal compute utilization due to wave quantization and attention bottlenecks. 2) hybrid batches disproportionately prioritize latency over throughput, resulting in wasted compute and memory bandwidth. To mitigate the issues, we present Bullet, a novel spatial-temporal orchestration system that eliminates these inefficiencies through precise phase coordination. Bullet enables concurrent execution of prefill and decode phases, while dynamically provisioning GPU resources using real-time performance modeling. By integrating SLO-aware scheduling and adaptive resource allocation, Bullet maximizes utilization without compromising latency targets. Experimental evaluations on real-world workloads demonstrate that Bullet delivers 1.26× average throughput gains (up to 1.55×) over state-of-the-arts, while consistently meeting latency constraints.

---

现代LLM服务系统由于计算密集型预填充阶段和内存密集型解码阶段之间的根本性不匹配而面临GPU利用率低效的问题。虽然当前的做法尝试通过将这些阶段组织成混合批次来解决这个问题，但这种解决方案产生了低效的权衡，牺牲了吞吐量或延迟，导致大量GPU资源未被充分利用。我们确定了两个关键根本原因：1）预填充阶段由于波前量化（wave quantization）和注意力瓶颈导致计算利用率不佳。2）混合批次过度优先考虑延迟而非吞吐量，导致计算和内存带宽浪费。为了缓解这些问题，我们提出了Bullet，这是一种通过精确阶段协调消除这些低效率的新型时空编排系统。Bullet支持预填充和解码阶段的并发执行，同时使用实时性能建模动态配置GPU资源。通过集成SLO感知调度和自适应资源分配，Bullet在不牺牲延迟目标的情况下最大化利用率。在实际工作负载上的实验评估表明，Bullet比最先进的技术提供1.26倍的平均吞吐量提升（最高1.55倍），同时始终满足延迟约束。

---

## 1. Introduction (引言)

GPUs have become the predominant computing platform for large language model (LLM) services, powering a wide range of applications with varying computational and latency demands. As these applications continue to grow in scale and complexity, maximizing GPU utilization has become crucial for elevating service quality. In response, a plethora of systems have been developed to optimize different aspects of LLM serving, such as kernel-level performance, scheduling strategies, and parallelization techniques.

However, the divergent computational characteristics of LLM inference make high GPU resource utilization particularly challenging. In detail, the workflow consists of a computationally intensive prefill phase that processes all inputs in parallel, succeeded by a memory-bound decode phase that generates tokens sequentially. These two contrasting phases lead to a structural imbalance in the utilization of computational resources and memory bandwidth. The dynamic nature of workload patterns further exacerbates this imbalance, forcing GPUs to alternate between these complementary resource utilization states.

Prior attempts have sought to address phase imbalance either by isolating prefill and decode across different GPUs, or by coordinating them on the same device through spatial multiplexing. Prefill-decode disaggregation represents the first approach, physically separating the two phases across dedicated GPU groups. However, such systems require careful tuning of GPU allocations for each phase, tailored to specific workload patterns, and struggle to reconfigure quickly under fluctuating request loads. In practice, chunked prefill has been pervasively adopted in production systems. This method leverages a fixed token budget to combine prefill and decode requests into hybrid batches, with longer sequences being partitioned into chunks to fit within capacity.

We identify a critical limitation that existing approaches often underutilize GPUs, resulting in suboptimal balance between latency and hardware efficiency. First, the attention exhibits relatively low compute utilization compared to linear layers. Second, the throughput-latency tradeoff in chunked prefill exhibits sub-linear scaling with chunk sizes that prolongs execution time for successive chunks. Third, while spatial sharing has the potential to exploit the natural complementarity between compute-intensive prefill and memory-bound decode phases through concurrent execution, effective orchestration is required.

To address GPU under-utilization and balance the tradeoff between throughput and latency, we propose Bullet, an LLM serving system that saturates GPU resources through spatial-temporal orchestration with fine-grained resource provisioning. Bullet proactively monitors request progress and dynamically adjusts resource provision to sustain high utilization while satisfying latency requirements. The system achieves such efficient execution with four key components: 1) a performance estimator with an accurate analytical model of low profile and runtime overhead; 2) an SLO-aware task scheduler that dynamically adjusts prefill and decode requests to balance throughput and latency; 3) a computational resource manager that offers lightning yet precise resource configuration; 4) a concurrent execution engine that enables asynchronous CPU control flow and GPU execution.

---

GPU已成为大型语言模型（LLM）服务的主导计算平台，为具有不同计算和延迟需求的广泛应用程序提供动力。随着这些应用在规模和复杂性上持续增长，最大化GPU利用率对于提升服务质量至关重要。作为回应，大量系统已被开发出来优化LLM服务的不同方面，如内核级性能、调度策略和并行化技术。

然而，LLM推理的不同计算特性使得高GPU资源利用率特别具有挑战性。具体来说，工作流包括计算密集型预填充阶段，并行处理所有输入，随后是内存密集型解码阶段，顺序生成token。这两个形成对比的阶段导致计算资源和内存带宽利用率的结构性不平衡。工作负载模式的动态特性进一步加剧了这种不平衡，迫使GPU在这些互补的资源利用状态之间交替。

先前的尝试试图通过跨不同GPU隔离预填充和解码，或通过空间复用在同一设备上协调它们来解决阶段不平衡。预填充-解码分解代表了第一种方法，在专用GPU组上物理分离两个阶段。然而，这样的系统需要仔细调整每个阶段的GPU分配，以适应特定的工作负载模式，并且在请求负载波动时难以快速重新配置。实际上，分块预填充已在生产系统中被广泛采用。这种方法利用固定的token预算将预填充和解码请求组合成混合批次，较长的序列被分区为块以适应容量。

我们确定了一个关键限制，即现有方法通常未能充分利用GPU，导致延迟和硬件效率之间的次优平衡。首先，与线性层相比，注意力表现出相对较低的计算利用率。其次，分块预填充中的吞吐量-延迟权衡与chunk大小呈次线性缩放，延长了连续块的执行时间。第三，虽然空间共享有可能通过并发执行利用计算密集型预填充和内存密集型解码阶段之间的自然互补性，但需要有效的协调。

为了解决GPU利用不足并平衡吞吐量和延迟之间的权衡，我们提出了Bullet，这是一种通过细粒度资源调配的时空编排来饱和GPU资源的LLM服务系统。Bullet主动监控请求进度并动态调整资源分配，以在满足延迟要求的同时保持高利用率。系统通过四个关键组件实现这种高效执行：1）一个具有低开销和运行时开销的准确分析模型的性能估计器；2）一个SLO感知的任务调度器，动态调整预填充和解码请求以平衡吞吐量和延迟；3）一个计算资源管理器，提供快速而精确的资源配置；4）一个并发执行引擎，支持异步CPU控制流和GPU执行。

---

## 主要贡献

1. **识别低效问题**：识别现有LLM服务系统中阻碍GPU利用率的低效问题，同时处理吞吐量-延迟权衡
2. **时空建模**：为空间-时间共享阶段建立准确建模，引入延迟感知资源调配的细粒度控制机制
3. **实现Bullet系统**：设计和实现LLM服务系统，将所提出的技术有效集成到现有框架中

---

## 实验结果

- 平均 **1.26×** 吞吐量提升
- 最高 **1.55×** 吞吐量提升

**代码**: https://github.com/zejia-lin/Bullet