# Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

**论文链接**: [arXiv:2504.14489](https://arxiv.org/abs/2504.14489)

**作者**: Yukang Chen, Weihao Cui, Han Zhao, Ziyi Xu, Xiaoze Fan, Xusheng Chen, Yangjie Zhou, Shixuan Sun, Bingsheng He, Quan Chen

**会议**: ASPLOS 2026

---

## Abstract (摘要)

Large Language Model (LLM) serving must meet stringent Service Level Objectives (SLOs) for both the prefill and decode phases. Some existing solutions disaggregate the two phases, causing potential resource idleness or compute redundancy. Others split the prefill phase into chunks and fuse it with decode iteration, creating a dilemma between SLO compliance and high utilization. To address these issues, an efficient serving system should dynamically adapt compute allocation, decouple compute from memory management, and execute prefill and decode independently. We present MuxWise, an LLM serving framework that adopts a new paradigm, intra-GPU prefill-decode multiplexing, to meet these requirements. To fully exploit the paradigm, MuxWise integrates a bubble-less multiplex engine, a contention-tolerant estimator, and an SLO-aware dispatcher. Evaluation shows that MuxWise improves peak throughput under SLO guarantees by an average of 2.20× (up to 3.06×) over state-of-the-art baselines.

---

大型语言模型（LLM）服务必须为预填充和解码两个阶段满足严格的服务水平目标（SLO）。一些现有解决方案将两个阶段分离，导致潜在的资源闲置或计算冗余。其他方案将预填充阶段分块并与解码迭代融合，在SLO合规性和高利用率之间产生两难境地。为了解决这些问题，高效的服务系统应该动态适应计算分配，将计算与内存管理解耦，并独立执行预填充和解码。我们提出了MuxWise，这是一种采用新范式的LLM服务框架——GPU内预填充-解码复用（intra-GPU prefill-decode multiplexing）。为了充分利用这一范式，MuxWise集成了无气泡复用引擎、抗争容限估计器和SLO感知调度器。评估表明，MuxWise在SLO保证下将峰值吞吐量比最先进的基线平均提高2.20倍（最高3.06倍）。

---

## 1. Introduction (引言)

Large language models (LLM) services now perform well across diverse workloads. At the request level, an LLM processes input in two phases: a prefill phase that produces the first token, followed by a decode phase that iteratively generates the remaining tokens. The ratio of input length (prefill) to output length (decode) varies across tasks. At the application level, tasks such as chatbot services or agent-based workloads often consist of multiple turns of requests with shared context.

To achieve high throughput for serving these workloads, existing LLM serving systems employ several optimizations. While requests arrive at different times, inflight batching stalls the ongoing decode phase to prefill new requests and then processes all decode iterations together in a single batch. It greatly improves compute utilization for the memory-intensive decode phase. Since multi-turn requests share context, LLM serving systems reuse intermediate results (i.e., the KV cache) both within and across requests through a KV cache pool.

LLM services also impose stringent Service Level Objectives (SLOs). For instance, chatbot typically requires Time-To-First-Token (TTFT) under 500 ms for prefill and Time-Between-Tokens (TBT) under 100 ms for decode. Since prefill and decode interleave in an LLM serving system, SLO violations may arise.

To sustain high goodput–peak throughput with SLO guarantees–existing methods fall into two categories: disaggregated serving and chunked prefill. Disaggregated serving separates the prefill and decode phases into distinct instances, but cannot adapt to serving dynamics and decreases goodput due to shrinking the KV cache pool. Chunked prefill splits the prefill phase into chunks and fuses each chunk with a decode iteration, but creates a dilemma between SLO compliance and high utilization.

We propose intra-GPU prefill-decode (PD) multiplexing as a promising new serving paradigm. Specifically, the prefill and decode phases are executed on different streaming multiprocessors (SMs) within the GPUs. In the new paradigm: 1) compute partitions can be reconfigured with low overhead to adapt to serving dynamics; 2) multiplexed phases share GPU memory, keeping the KV cache pool efficient; 3) with spatial sharing, prefill and decode execute independently, avoiding the tradeoff between SLO compliance and utilization.

To this end, we propose MuxWise, an LLM serving framework that achieves high goodput across diverse workloads. MuxWise comprises three modules: a bubble-less multiplex engine, a contention-tolerant estimator, and an SLO-aware dispatcher. Experiments show that MuxWise achieves an average 2.20× goodput improvement (up to 3.06×) over state-of-the-art solutions.

---

大型语言模型（LLM）服务目前在不同工作负载上表现良好。在请求级别，LLM分两个阶段处理输入：预填充阶段产生第一个token，然后是解码阶段迭代生成剩余token。输入长度（预填充）与输出长度（解码）的比率因任务而异。在应用层面，聊天机器人服务或基于agent的工作负载等任务通常包含多轮具有共享上下文的请求。

为了实现这些工作负载的高吞吐量，现有的LLM服务系统采用了几种优化措施。虽然请求在不同时间到达，但飞行批处理会暂停正在进行的解码阶段以预填充新请求，然后在一个批次中一起处理所有解码迭代。它大大提高了内存密集型解码阶段的计算利用率。由于多轮请求共享上下文，LLM服务系统通过KV缓存池在请求内部和跨请求重用中间结果（即KV缓存）。

LLM服务还要求严格的服务级别目标（SLO）。例如，聊天机器人通常要求预填充的首个token时间（TTFT）低于500ms，解码的token间隔时间（TBT）低于100ms。由于预填充和解码在LLM服务系统中交错，可能会出现SLO违规。

为了保持高goodput——在SLO保证下的峰值吞吐量——现有方法分为两类：分解服务和分块预填充。分解服务将预填充和解码阶段分离到不同的实例中，但无法适应服务动态，并因KV缓存池缩小而降低goodput。分块预填充将预填充阶段分块并将每个块与解码迭代融合，但在SLO合规性和高利用率之间产生两难境地。

我们提出了GPU内预填充-解码（PD）复用作为一种有前途的新服务范式。具体来说，预填充和解码阶段在GPU内的不同流多处理器（SM）上执行。在新范式中：1）计算分区可以低开销重新配置以适应服务动态；2）复用阶段共享GPU内存，保持KV缓存池高效；3）通过空间共享，预填充和解码独立执行，避免SLO合规性和利用率之间的权衡。

为此，我们提出了MuxWise，一个在各种工作负载下实现高goodput的LLM服务框架。MuxWise包含三个模块：无气泡复用引擎、抗争容限估计器和SLO感知调度器。实验表明，MuxWise比最先进的解决方案实现了平均2.20倍的goodput提升（最高3.06倍）。

---

## 主要贡献

1. **识别关键需求**：通过详细分析先前工作，识别高goodput LLM服务的关键需求
2. **提出新范式**：提出PD复用这一新的LLM服务范式
3. **实现MuxWise**：无气泡复用引擎 + 抗争容限估计器 + SLO感知调度器

---

## 实验结果

- 平均 **2.20×** goodput提升
- 最高 **3.06×** goodput提升

**关键词**: LLM Serving, PD-Multiplexing, Goodput