# Trinity: Disaggregating Vector Search from Prefill-Decode Disaggregation in LLM Serving

**arXiv**: 2512.02281
**链接**: https://arxiv.org/abs/2512.02281
**作者**: Yi Liu, Chen Qian
**会议**: arXiv 2025
**主题**: llm_serving / Disaggregated Serving / Vector Search

## 摘要 (Abstract)

Prefill and decode (PD) disaggregation separates prompt prefill and token-by-token decode stages into distinct GPU pools and has become the dominant architecture for large-scale LLM serving in industry. Also, retrieval tasks via vector search remains entangled with the model inference process, like heterogeneous RAG requests and prompt answer caches, inflating tail latency. We are motivated to investigate how vector search should be orchestrated along with PD disaggregation with a dedicated deployment architecture without violating SLOs in various retrieval workloads. We present Trinity, a practical framework that consolidates all retrieval into a single, shared vector-search GPU pool and make it work with PD disaggregated LLM serving in match. Trinity introduces (1) a novel architecture for deploying GPU-based vector search service in PD disaggregation. (2) Continuous batching for vector search that make full used of GPUs under heterogeneous queries; (3) Stage-aware scheduling that preempts vector search requests between both decode and prefill tasks.

## 摘要 (中文)

Prefill-decode (PD) 解耦将 prompt prefill 和逐 token decode 分离到不同 GPU 池，已成为大规模 LLM 服务的主流架构。同时，通过向量搜索进行的检索任务仍然与模型推理过程纠缠在一起，如异构 RAG 请求和 prompt 答案缓存，导致尾部延迟膨胀。我们研究如何在 PD 解耦架构中编排向量搜索的专用部署架构。提出 Trinity，一个实用框架，将所有检索整合到单一共享的向量搜索 GPU 池中，并与 PD 解耦 LLM 服务协同工作。Trinity 引入：（1）PD 解耦中部署 GPU 向量搜索服务的新架构；（2）向量搜索的连续批处理，在异构查询下充分利用 GPU；（3）Stage-aware 调度，在 decode 和 prefill 任务之间抢占向量搜索请求。

## 关键贡献

1. 三层解耦架构：Prefill + Decode + Vector Search 各有独立 GPU 池
2. 向量搜索连续批处理机制
3. Stage-aware 调度抢占机制