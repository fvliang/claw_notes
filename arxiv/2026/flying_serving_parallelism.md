# FLYING SERVING: On-the-Fly Parallelism Switching for Large Language Model Serving

## 论文信息
- **标题**: FLYING SERVING: On-the-Fly Parallelism Switching for Large Language Model Serving
- **作者**: Shouwei Gao, Junqi Yin, Feiyi Wang, Wenqian Dong
- **arXiv**: [2602.22593](https://arxiv.org/abs/2602.22593)
- **会议**: ACM International Conference on Supercomputing (ICS 2026)
- **提交时间**: 2026年2月25日 (v1), 2026年3月2日 (v2)
- **领域**: Distributed, Parallel, and Cluster Computing (cs.DC)

## 摘要 (Abstract)
Production LLM serving must simultaneously deliver high throughput, low latency, and sufficient context capacity under non-stationary traffic and mixed request requirements. Data parallelism (DP) maximizes throughput by running independent replicas, while tensor parallelism (TP) reduces per-request latency and pools memory for long-context inference. However, existing serving stacks typically commit to a static parallelism configuration at deployment; adapting to bursts, priorities, or long-context requests is often disruptive and slow. We present Flying Serving, a vLLM-based system that enables online DP-TP switching without restarting engine workers. Flying Serving makes reconfiguration practical by virtualizing the state that would otherwise force data movement: (i) a zero-copy Model Weights Manager that exposes TP shard views on demand, (ii) a KV Cache Adaptor that preserves request KV state across DP/TP layouts, (iii) an eagerly initialized Communicator Pool to amortize collective setup, and (iv) a deadlock-free scheduler that coordinates safe transitions under execution skew. Across three popular LLMs and realistic serving scenarios, Flying Serving improves performance by up to 4.79x under high load and 3.47x under low load while supporting latency- and memory-driven requests.

## 摘要 (中文)
生产级LLM服务必须在非平稳流量和混合请求需求下同时提供高吞吐量、低延迟和足够的上下文容量。数据并行（DP）通过运行独立副本来最大化吞吐量，而张量并行（TP）减少每请求延迟并聚合内存用于长上下文推理。然而，现有服务堆栈通常在部署时静态配置并行性；适应突发、优先级或长上下文请求通常具有破坏性且缓慢。我们提出了Flying Serving，这是一个基于vLLM的系统，可以在不重启引擎 workers 的情况下实现在线DP-TP切换。Flying Serving通过虚拟化原本会强制数据移动的状态来使重新配置变得实用：(i) 零拷贝模型权重管理器，可按需暴露TP分片视图，(ii) KV Cache适配器，可在DP/TP布局之间保留请求KV状态，(iii) 积极初始化的通信器池，以分摊集体设置，以及 (iv) 无死锁调度器，在执行偏斜下协调安全转换。在三个流行的LLM和实际服务场景中，Flying Serving在高负载下将性能提升至4.79倍，在低负载下提升至3.47倍，同时支持延迟驱动和内存驱动的请求。

## 核心贡献
1. **在线DP-TP切换**: 首次实现无需重启引擎workers的在线数据并行-张量并行切换
2. **状态虚拟化**: 通过虚拟化模型权重和KV Cache状态避免数据移动
3. **零拷贝权重管理**: 按需暴露TP分片视图

## 技术细节
- **基础**: 基于vLLM构建
- **性能**: 高负载下提升4.79倍，低负载下提升3.47倍
- **接受会议**: ACM ICS 2026

---

*更新时间: 2026-03-25*