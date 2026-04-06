# PROTEUS: SLA-Aware Routing via Lagrangian RL for Multi-LLM Serving Systems

**作者**: Amit Singh Bhatti, Vishal Vaddina, Dagnachew Birru

**arXiv**: 2601.19402

**年份**: 2026

**会议**: arXiv (EuroMLSys 2026)

**主题**: LLM Serving

## 摘要 (Abstract)

Production LLM deployments increasingly leverage multiple specialized models to handle diverse query types, necessitating intelligent routing mechanisms that direct requests to appropriate backend models while meeting Service Level Objectives (SLOs). We present PROTEUS, a novel reinforcement learning framework for SLA-aware request routing in multi-LLM serving systems. Unlike prior work that treats routing as a static optimization problem, PROTEUS formulates it as a sequential decision-making task and employs Lagrangian-based RL to jointly optimize for both SLO compliance and system throughput. Our approach dynamically adapts to varying query workloads and model performance characteristics, learned through continuous interaction with the serving system. We evaluate PROTEUS on a production-like multi-LLM serving cluster handling diverse query patterns. Results show that PROTEUS reduces SLO violations by 34% while maintaining 28% higher throughput compared to state-of-the-art routing policies.

## 摘要中文

生产环境的LLM部署越来越多地利用多个专业模型来处理不同的查询类型，需要智能路由机制将请求引导到适当的后端模型，同时满足服务等级目标（SLO）。我们提出了PROTEUS，这是一种用于多LLM服务系统中SLO感知请求路由的新型强化学习框架。与将路由视为静态优化问题的先前工作不同，PROTEUS将其表述为序贯决策任务，并采用基于拉格朗日的RL来联合优化SLO合规性和系统吞吐量。我们的方法动态适应不同的查询工作负载和模型性能特征，通过与服务系统的持续交互学习。我们在处理多样化查询模式的生产级多LLM服务集群上评估PROTEUS。结果表明，PROTEUS在保持28%更高吞吐量的情况下，将SLO违规减少了34%。

