# ReviveMoE: Fast Recovery for Hardware Failures in Large-Scale MoE LLM Inference Deployments

**arXiv**: 2602.21140
**链接**: https://arxiv.org/abs/2602.21140
**作者**: Haley Li, Xinglu Wang, Cong Feng, Chunxu Zuo, Yanan Wang, Hei Lo, Yufei Cui, Bingji Wang, Duo Cui, Shuming Jing, Yizhou Shan, Ying Xiong, Jiannan Wang, Yong Zhang, Zhenan Fan
**会议**: arXiv 2026
**主题**: llm_serving / MoE / Fault Tolerance
**部署**: Huawei Cloud MaaS, xDeepServe

## 摘要 (Abstract)

As LLM deployments scale over more hardware, the probability of a single failure in a system increases significantly, and cloud operators must consider robust countermeasures to handle these inevitable failures. A common recovery approach is to simply restart the LLM serving instance; however, this is costly in model-as-a-service (MaaS) inference settings, where reloading model weights and recompiling computation graphs can introduce significant delays to incoming requests. We propose ReviveMoE, a method for rapid failure recovery in large-scale LLM deployments without restarting the serving instance. ReviveMoE is designed to support both the traditional LLM architecture, which collocates MoE and attention on the same hardware, and the disaggregated architectures, which separate MoE from attention. Integrated into Huawei Cloud's MaaS, ReviveMoE is built on top of Huawei's xDeepServe serving platform and the XCCL communications library.

## 摘要 (中文)

随着 LLM 部署规模扩展到更多硬件，系统中单点故障的概率显著增加，云运营商必须考虑强有力的对策来处理这些不可避免的故障。常见的恢复方法是简单重启 LLM 服务实例；然而在 MaaS 推理设置中，重新加载模型权重和重新编译计算图会对传入请求引入显著延迟。我们提出 ReviveMoE，一种在大规模 LLM 部署中无需重启服务实例即可快速故障恢复的方法。ReviveMoE 支持传统 LLM 架构（MoE 和 attention 同位部署）和解耦架构（MoE 与 attention 分离部署）。已集成到华为云的 MaaS 平台，基于华为的 xDeepServe 服务平台和 XCCL 通信库构建。

## 关键贡献

1. 无需重启服务实例的快速故障恢复方法
2. 支持传统同位架构和解耦架构
3. 已在华为云 MaaS 生产环境部署