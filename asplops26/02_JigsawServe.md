# Serving Compound Inference Systems on Datacenter GPUs

**论文链接**: [arXiv:2603.08797](https://arxiv.org/abs/2603.08797)

**作者**: Sriram Devata, Rahul Sukthankar, Saurabh Adya

**会议**: ASPLOS '26 (Extended version from HCDS workshop)

---

## Abstract (摘要)

Applications in emerging domains such as XR are being built as compound inference systems, where multiple ML models are composed in the form of a task graph to service each request. Serving these compound systems efficiently raises two questions: how to apportion end-to-end latency and accuracy budgets between different tasks in a compound inference system, and how to allocate resources effectively for different models with varying resource requirements. We present JigsawServe, the first serving framework that jointly optimizes for latency, accuracy, and cost in terms of GPU resources by adaptively choosing model variants and performing fine-grained resource allocation by spatially partitioning the GPUs for each task of a compound inference system. Analytical evaluation of a system with a large number of GPUs shows that JigsawServe can increase the maximum serviceable demand (in requests per second) by 11.3x when compared to the closest prior work. Our empirical evaluation shows that for a large range of scenarios, JigsawServe consumes only 43.3% of the available GPU resources while meeting accuracy SLOs with less than 0.6% latency SLO violations. All of the features in JigsawServe contribute to this high efficiency -- sacrificing any one feature of accuracy scaling, GPU spatial partitioning, or task-graph-informed resource budgeting significantly reduces efficiency.

---

在XR等新兴领域的应用正被构建为复合推理系统，其中多个ML模型以任务图的形式组合来服务每个请求。高效地服务这些复合系统提出了两个问题：如何在复合推理系统的不同任务之间分配端到端延迟和精度预算，以及如何有效地为具有不同资源需求的模型分配资源。我们提出了JigsawServe，这是第一个通过为复合推理系统的每个任务自适应选择模型变体并执行细粒度资源分配（通过空间分割GPU）来共同优化延迟、精度和GPU资源成本的推理服务框架。对拥有大量GPU的系统进行分析评估表明，与最接近的先前工作相比，JigsawServe可以将最大可服务需求（每秒请求数）提高11.3倍。我们的实证评估表明，在大范围的场景中，JigsawServe仅消耗43.3%的可用GPU资源，同时满足精度SLO，延迟SLO违规率低于0.6%。JigsawServe的所有特性都有助于实现这一高效率——牺牲精度扩展、GPU空间分割或任务图知情的资源预算中的任何一个特性都会显著降低效率。

---

## 1. Introduction (引言)

As machine learning (ML) inference becomes a dominant workload for datacenter resources, there is growing emphasis on maximizing the efficiency of inference serving systems. Compound inference systems are becoming increasingly important in many emerging domains such as multi-agent systems and extended reality (XR). Such systems compose multiple ML models, each performing a specific task, to provide complex functionalities, and present new challenges and opportunities for increased efficiency. This work focuses on efficient serving of compound inference systems on datacenter GPUs, where each request requires invoking a directed acyclic graph (DAG) of ML inference tasks, which are potentially dynamically determined.

We focus on two aspects that are unique to compound inference systems. First, we expect that any latency and accuracy Service Level Objectives (SLOs) are provided for the end-to-end compound inference system and need to be apportioned among the individual tasks. This affords new sources of flexibility for the serving system to choose among multiple model variants (with different latencies and accuracies) and GPU resource allocation for each task, when compared to a single task system with a given SLO constraint. Second, the ML model variants across the various tasks in a compound inference system have different resource requirements, motivating finer-grained heterogeneity in the optimal resource choice for the different tasks. Recent GPU hardware provide spatial partitioning mechanisms that enable fine-grained allocation of GPU resources. For example, NVIDIA GPUs provide spatial partitioning mechanisms such as Multi-Process Service (MPS) and Multi-Instance GPU (MIG).

We present JigsawServe, the first work to jointly optimize for latency, accuracy, and fine-grained cost (in terms of GPU resources required) by adaptively choosing model variants and allocating fine-grained spatial partitions of GPUs for each task of a compound inference system, while preserving end-to-end accuracy and latency SLOs.

There is a plethora of literature on increasing the efficiency of serving systems for requests with a single model inference, including systems that perform accuracy scaling and spatial partitioning. Compound inference systems have received relatively less attention. Table 1 summarizes key prior works that consider resource budgeting for compound inference systems, accuracy scaling through different model variants of a task, and/or spatial partitioning of GPU resources.

Specifically, we make the following contributions:

- 1. We introduce JigsawServe, a novel framework to jointly optimize latency, accuracy, and fine-grained cost for compound inference systems on datacenter GPUs. JigsawServe uniquely combines: (1) per-task accuracy scaling via choosing model variants, (2) GPU spatial partitioning, and (3) task-graph-informed latency and accuracy SLO budgeting.

- 2. Our analytical evaluation compares the various combinations of the features. We find that spatial partitioning (S) delivers the highest standalone gain in serving capacity for the same GPU resources (5.25× over Unopt), compared to accuracy scaling (A: 1.6×) and task-graph-informed budgeting (T: 1.1×). JigsawServe's full integration of the three features (A+S+T) achieves 21.6× capacity, surpassing combinations of S with A or T in S+A (12.1×) and S+T (7.8×).

- 3. We emphasize that no prior work has evaluated GPU spatial partitioning for compound inference systems. Systems without S show significantly lower serving capacity than any system with S. A+T, in particular, is closest to (and explores a larger search space than) prior work of Loki. We find that JigsawServe allows 11.3× higher capacity than A+T.

- 4. Our empirical evaluation compares the four top performing systems under a wide range of compound inference systems and workload demand conditions. JigsawServe shows the best overall performance, using just 43.3% of the available GPU resources on average, respecting the accuracy SLO, and showing less than 0.6% average SLO violations. Alternatives suffer significantly: S+T/A+T exceed 10% violations and have 2× resource usage in at least one case; S+A requires 33% more resources than JigsawServe with 6.7% SLO violations.

---

随着机器学习（ML）推理成为数据中心资源的主导工作负载，人们越来越强调最大化推理服务系统的效率。复合推理系统在许多新兴领域（如多代理系统和扩展现实（XR））变得越来越重要。这类系统组合多个ML模型，每个模型执行特定任务，以提供复杂功能，并为提高效率带来新的挑战和机遇。本工作专注于在数据中心GPU上高效服务复合推理系统，其中每个请求都需要调用可能动态确定的ML推理任务的有向无环图（DAG）。

我们关注复合推理系统特有的两个方面。首先，我们预计端到端复合推理系统会提供延迟和精度服务级别目标（SLO），需要在各个任务之间分配。这为服务系统提供了新的灵活性来源，与具有给定SLO约束的单任务系统相比，可以为每个任务选择多个模型变体（具有不同的延迟和精度）和GPU资源分配。其次，复合推理系统中各种任务的ML模型变体具有不同的资源需求，促使不同任务的最佳资源选择具有更细粒度的异构性最近的GPU硬件提供空间分割机制，实现GPU资源的细粒度分配。例如，NVIDIA GPU提供空间分割机制，如多进程服务（MPS）和多实例GPU（MIG）。

我们提出了JigsawServe，这是第一个通过为复合推理系统的每个任务自适应选择模型变体和分配细粒度GPU空间分区来共同优化延迟、精度和细粒度成本（以所需GPU资源计）的工作，同时保持端到端精度和延迟SLO。

有大量文献关于提高单模型推理请求的服务系统效率，包括执行精度扩展和空间分割的系统。复合推理系统受到的关注相对较少。表1总结了考虑复合推理系统资源预算、任务模型变体精度扩展和/或GPU资源空间分割的关键先前工作。

具体而言，我们做出以下贡献：

1. 我们介绍了JigsawServe，这是一个新颖的框架，可在数据中心GPU上为复合推理系统共同优化延迟、精度和细粒度成本。JigsawServe独特地结合了：(1)通过选择模型变体进行每任务精度扩展，(2)GPU空间分割，以及(3)任务图知情的延迟和精度SLO预算。

2. 我们的分析评估比较了表中各种特性的组合。我们发现空间分割（S）在相同GPU资源下提供了最高的独立服务容量增益（比Unopt高5.25倍），而精度扩展（A：1.6倍）和任务图知情预算（T：1.1倍）。JigsawServe的三个特性的完整集成（A+S+T）实现了21.6倍的容量，超过了S与A或T的组合（S+A为12.1倍，S+T为7.8倍）。

3. 我们强调先前没有任何工作评估过复合推理系统的GPU空间分割。没有S的系统明显比任何有S的系统显示更低的服务容量。特别是A+T最接近（并探索比先前Loki工作更大的搜索空间）。我们发现JigsawServe允许比A+T高11.3倍的容量。

4. 我们的实证评估在广泛的复合推理系统和工作负载需求条件下比较了四个表现最好的系统。JigsawServe显示出最佳的整体性能，平均仅使用43.3%的可用GPU资源，尊重精度SLO，延迟SLO违规率低于0.6%。替代方案表现显著：S+T/A+T在至少一种情况下超过10%的违规率和2倍的资源使用；S+A需要比JigsawServe多33%的资源，违规率为6.7%。

---

## 主要贡献

1. **JigsawServe框架**：首个为数据中心GPU上的复合推理系统共同优化延迟、精度和细粒度成本的服务框架，结合了每任务精度扩展、GPU空间分割和任务图知情的SLO预算。

2. **分析评估**：显示空间分割提供最高的独立服务容量增益（5.25倍），完整集成实现21.6倍容量。

3. **GPU空间分割**：首次评估复合推理系统的GPU空间分割，比最接近的先前工作（Loki）提供11.3倍更高的容量。

4. **实证评估**：JigsawServe仅使用43.3%的GPU资源，延迟SLO违规率低于0.6%，显著优于所有替代方案。