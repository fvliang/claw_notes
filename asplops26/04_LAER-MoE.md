# LAER-MoE: Load-Adaptive Expert Re-layout for Efficient Mixture-of-Experts Training

**论文链接**: [arXiv:2602.11686](https://arxiv.org/abs/2602.11686)

**作者**: Xinyi Liu, Zijian Zhang, YongLi Zhu, Jiale Zhang, Peng Sun, XuanWang, Qi Qi, Jingren Zhou, Tong Yang, Bin Cui

**会议**: ASPLOS 2026

---

## Abstract (摘要)

Expert parallelism is vital for effectively training Mixture-of-Experts (MoE) models, enabling different devices to host distinct experts, with each device processing different input data. However, during expert parallel training, dynamic routing results in significant load imbalance among experts: a handful of overloaded experts hinder overall iteration, emerging as a training bottleneck.

In this paper, we introduce LAER-MoE, an efficient MoE training framework. The core of LAER-MoE is a novel parallel paradigm, Fully Sharded Expert Parallel (FSEP), which fully partitions each expert parameter by the number of devices and restores partial experts at expert granularity through All-to-All communication during training. This allows for flexible re-layout of expert parameters during training to enhance load balancing. In particular, we perform fine-grained scheduling of communication operations to minimize communication overhead. Additionally, we develop a load balancing planner to formulate re-layout strategies of experts and routing schemes for tokens during training. We perform experiments on an A100 cluster, and the results indicate that our system achieves up to 1.69x acceleration compared to the current state-of-the-art training systems. Source code available at this https URL.

---

专家并行对于有效训练混合专家（MoE）模型至关重要，使不同设备托管不同的专家，每个设备处理不同的输入数据。然而，在专家并行训练期间，动态路由导致专家之间严重的负载不平衡：少数过载的专家阻碍整体迭代，成为训练瓶颈。

在本文中，我们介绍了LAER-MoE，一个高效的MoE训练框架。LAER-MoE的核心是一种新颖的并行范式——完全分片专家并行（FSEP），它通过设备数量完全分割每个专家参数，并在训练期间通过All-to-All通信在专家粒度上恢复部分专家。这允许在训练期间灵活重新布局专家参数以增强负载平衡。特别是，我们执行细粒度的通信操作调度以最小化通信开销。此外，我们开发了一个负载平衡规划器，用于制定训练期间专家的重新布局策略和token路由方案。我们在A100集群上进行了实验，结果表明我们的系统比当前最先进的训练系统实现了高达1.69倍的加速。源代码可在https://github.com/PKU-DAIR/Hetu-Galvatron/tree/laer-moe获取。

---

## 主要贡献

1. **Fully Sharded Expert Parallel (FSEP)**：一种新的并行范式，完全分割每个专家参数
2. **动态专家重新布局**：在训练期间灵活调整专家布局以实现负载均衡
3. **细粒度通信调度**：最小化通信开销
4. **负载平衡规划器**：制定专家重新布局策略和token路由方案

---

## 实验结果

- 在A100集群上实现最高 **1.69倍** 加速 compared to state-of-the-art

**代码**: https://github.com/PKU-DAIR/Hetu-Galvatron/tree/laer-moe