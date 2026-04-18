# From Inference Routing to Agent Orchestration: Declarative Policy Compilation with Cross-Layer Verification

**arXiv**: 2603.27299
**链接**: https://arxiv.org/abs/2603.27299
**作者**: Huamin Chen, Xunzhuo Liu, Bowei He, Xue Liu
**会议**: arXiv 2026 (Position Paper)
**主题**: llm_serving / Inference Routing / Agent Orchestration

## 摘要 (Abstract)

The Semantic Router DSL is a non-Turing-complete policy language deployed in production for per-request LLM inference routing: content signals (embedding similarity, PII detection, jailbreak scoring) feed into weighted projections and priority-ordered decision trees that select a model, enforce privacy policies, and produce structured audit traces -- all from a single declarative source file. Prior work established conflict-free compilation for probabilistic predicates and positioned the DSL within the Workload-Router-Pool inference architecture.

This paper extends the same language from stateless, per-request routing to multi-step agent workflows -- the full path from inference gateway to agent orchestration to infrastructure deployment. The DSL compiler emits verified decision nodes for orchestration frameworks (LangGraph, OpenClaw), Kubernetes artifacts (NetworkPolicy, Sandbox CRD, ConfigMap), YANG/NETCONF payloads, and protocol-boundary gates (MCP, A2A) -- all from the same source.

Because the language is non-Turing-complete, the compiler guarantees exhaustive routing, conflict-free branching, referential integrity, and audit traces structurally coupled to the decision logic. Because signal definitions are shared across targets, a threshold change propagates from inference gateway to agent gate to infrastructure artifact in one compilation step -- eliminating cross-team coordination as the primary source of policy drift. We ground the approach in four pillars -- auditability, cost efficiency, verifiability, and tunability -- and identify the verification boundary at each layer.

## 摘要 (中文)

Semantic Router DSL 是一种非 Turing 完备的策略语言，已在生产中部署用于 LLM 推理路由：内容信号（嵌入相似度、PII 检测、jailbreak 评分）馈入加权投影和优先级有序决策树，选择模型、执行隐私策略、生成结构化审计追踪——全部来自单一声明式源文件。

本文将同一语言从无状态的单请求路由扩展到多步 agent 工作流——从推理网关到 agent 编排到基础设施部署的完整路径。DSL 编译器为编排框架（LangGraph、OpenClaw）、Kubernetes 制品（NetworkPolicy、Sandbox CRD、ConfigMap）、YANG/NETCONF 负载和协议边界门（MCP、A2A）发出经过验证的决策节点——全部来自同一源文件。

由于语言是非 Turing 完备的，编译器保证穷举路由、无冲突分支、引用完整性和与决策逻辑结构耦合的审计追踪。四个支柱：可审计性、成本效率、可验证性和可调性。

## 关键贡献

1. 将声明式 DSL 从单请求路由扩展到多步 agent 工作流
2. 单一源文件编译出推理网关、agent 编排、基础设施部署的多层制品
3. 非 Turing 完备语言保证无冲突分支和引用完整性
4. 四支柱框架：可审计性、成本效率、可验证性、可调性