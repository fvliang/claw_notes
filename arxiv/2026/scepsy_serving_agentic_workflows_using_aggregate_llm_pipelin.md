# ScePsy: Serving Agentic Workflows Using Aggregate LLM Pipelines

**Authors:** Marcel Wagenländer, Otto White, Britannio Jarrett, Pedro Silvestre, Yanda Tao, Guo Li, Huanzhou Zhu, Llúis Vilanova, Peter Pietzuch

**Conference:** arXiv 2026

**Year:** 2026

**ArXiv:** [2604.15186](<https://arxiv.org/abs/2604.15186>)

**Topic:** LLM Serving

---

## Abstract (English)

Agentic workflows carry out complex tasks by orchestrating multiple large language models (LLMs) and tools. Serving such workflows at a target throughput with low latency is challenging because they can be defined using arbitrary agentic frameworks and exhibit unpredictable execution times: execution may branch, fan-out, or recur in data-dependent ways. Since LLMs in workflows often outnumber available GPUs, their execution also leads to GPU oversubscription. We describe ScePsy, a new agentic serving system that efficiently schedules arbitrary multi-LLM agentic workflows onto a GPU cluster. ScePsy exploits the insight that, while agentic workflows have unpredictable end-to-end latencies, the shares of each LLM's total execution times are comparatively stable across executions. ScePsy decides on GPU allocations based on these aggregate shares: first, it profiles the LLMs under different parallelism degrees. It then uses these statistics to construct an Aggregate LLM Pipeline, which is a lightweight latency/throughput predictor for allocations. To find a GPU allocation that minimizes latency while achieving a target throughput, ScePsy uses the Aggregate LLM Pipeline to explore a search space over fractional GPU shares, tensor parallelism degrees, and replica counts. It uses a hierarchical heuristic to place the best allocation onto the GPU cluster, minimizing fragmentation, while respecting network topology constraints. Our evaluation on realistic agentic workflows shows that ScePsy achieves up to 2.4x higher throughput and 27x lower latency compared to systems that optimize LLMs independently or rely on user-specified allocations.

## Abstract (Chinese / 中文摘要)

Agentic workflows通过编排多个大语言模型(LLM)和工具来执行复杂任务。在目标吞吐量下以低延迟服务这些工作流具有挑战性，因为它们可以使用任意的agentic框架定义，并表现出不可预测的执行时间：执行可能分支、扇出或以数据依赖的方式递归。由于工作流中的LLM通常多于可用GPU，其执行也会导致GPU超额订阅。我们描述了ScePsy，一个新的agentic服务系统，可以高效地将任意多LLM agentic工作流调度到GPU集群上。ScePsy利用了一个洞察：虽然agentic工作流具有不可预测的端到端延迟，但每个LLM的总执行时间份额在不同执行中相对稳定。ScePsy基于这些聚合份额决定GPU分配：首先，在不同并行度下对LLM进行性能分析，然后使用这些统计数据构建Aggregate LLM Pipeline——一个轻量级的延迟/吞吐量预测器。为了找到最小化延迟同时实现目标吞吐量的GPU分配，ScePsy使用Aggregate LLM Pipeline探索分数GPU份额、张量并行度和副本计数的搜索空间，并使用分层启发式方法将最佳分配放置到GPU集群上，最小化碎片化同时尊重网络拓扑约束。在真实的agentic工作流上的评估显示，ScePsy比独立优化LLM或依赖用户指定分配的系统实现了高达2.4倍的吞吐量和27倍的更低延迟。

---

*Auto-collected from arXiv on 2026-04-17*
