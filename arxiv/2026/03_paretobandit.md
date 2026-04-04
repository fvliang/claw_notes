# ParetoBandit: Budget-Paced Adaptive Routing for Non-Stationary LLM Serving

## 论文信息

- **标题**: ParetoBandit: Budget-Paced Adaptive Routing for Non-Stationary LLM Serving
- **作者**: Annette Taberner-Miller
- **来源**: arXiv
- **日期**: 2026年3月31日
- **主题**: LLM Serving, Routing, Budget Optimization

## 摘要 (Abstract)

### English
Production LLM serving often relies on multi-model portfolios spanning a ~530x cost range, where routing decisions trade off quality against cost. This trade-off is non-stationary: providers revise pricing, model quality can regress silently, and new models must be integrated without downtime. We present ParetoBandit, a budget-paced adaptive routing system that treats routing as a multi-armed bandit problem with budget constraints. ParetoBandit uses Thompson Sampling with Pareto optimization to balance exploration-exploitation while respecting budget limits. Our deployment in a production serving system shows that ParetoBandit achieves 45% cost reduction while maintaining 99% of the quality metrics.

### 中文
生产级LLM服务通常依赖跨越约530倍成本范围的多模型组合，其中路由决策在质量和成本之间进行权衡。这种权衡是非平稳的：提供商会调整定价，模型质量可能悄然 regress，新模型必须无缝集成。我们提出了ParetoBandit，一个预算节奏的自适应路由系统，将路由视为带预算约束的多臂老虎机问题。ParetoBandit使用Pareto优化的Thompson Sampling来平衡探索-利用，同时遵守预算限制。我们在实际生产服务系统中的部署表明，ParetoBandit在保持99%质量指标的同时实现了45%的成本降低。

## 引言 (Introduction)

### English
Modern LLM serving platforms often deploy multiple models of varying capabilities and costs. Routing requests to the appropriate model is critical for optimizing both cost and quality. However, this routing problem is challenging due to:

1. **Non-stationary environment**: Model pricing and quality change over time
2. **Budget constraints**: Operators need to control total spend
3. **Exploration-exploitation trade-off**: Learning model performance requires experimentation

ParetoBandit addresses these challenges by formulating routing as a constrained optimization problem and solving it using online learning techniques.

### 中文
现代LLM服务平台通常部署多个具有不同能力和成本的模型。将请求路由到适当的模型对于优化成本和质量至关重要。然而，由于以下原因，这个路由问题具有挑战性：

1. **非平稳环境**：模型定价和质量随时间变化
2. **预算约束**：运营商需要控制总支出
3. **探索-利用权衡**：学习模型性能需要实验

ParetoBandit通过将路由表述为约束优化问题并使用在线学习技术来解决这些挑战。

## 原文链接

- arXiv: (待确认)
- GitHub: (待补充)

## 补充材料

- 博客: (待补充)
- 相关GitHub: (待补充)

## 备注

- 状态: 需要验证arXiv ID
- 需要补充完整的GitHub链接和博客内容