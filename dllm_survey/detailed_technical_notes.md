# Diffusion Large Language Models (DLLM) 详细调查报告

## 1. 引言：为什么需要扩散语言模型？

### 1.1 自回归模型的局限性
- **序列生成限制**：自回归(AR)语言模型采用从左到右的逐token生成方式，本质上是串行的
- **推理效率瓶颈**：无法并行生成多个token，限制了推理吞吐量
- **上下文利用不足**：在生成过程中只能利用左侧上下文，无法利用双向信息

### 1.2 扩散模型的优势
- **并行生成能力**：可以同时更新多个位置的token
- **双向上下文利用**：在生成过程中可以利用完整的双向上下文信息
- **更好的全局一致性**：通过迭代去噪过程，能够更好地保持文本的全局一致性

## 2. 扩散模型基础理论

### 2.1 连续扩散模型核心思想
扩散模型的核心洞察是**噪声条件化分数学习**，而不是直接学习原始数据分数：

#### 2.1.1 噪声条件化分数
- 定义高斯噪声破坏过程：$x_\sigma = x_0 + \sigma\epsilon$
- 学习噪声条件化分数：$s_\sigma(x) = \nabla_x \log p_\sigma(x)$
- **关键优势**：将不可行的分数学习问题转化为可扩展的去噪回归问题

#### 2.1.2 去噪分数匹配(DSM)
DSM目标函数：
$$\mathcal{L}_{\text{DSM}}(\theta) = \mathbb{E}_{x_0,\epsilon} \left[ \left\| s_\theta(x_0+\sigma\epsilon,\sigma) + \frac{\epsilon}{\sigma} \right\|_2^2 \right]$$

这个公式揭示了核心洞察：
- 不需要直接学习$\nabla_x \log p_{\text{data}}(x)$
- 添加噪声后，最优分数估计器有简单的回归目标$-\epsilon/\sigma$
- 训练简化为标准的监督学习任务

### 2.2 离散语言扩散模型
对于离散的token序列，需要将连续扩散的思想进行适配：

#### 2.2.1 吸收式前向过程
SEDD采用吸收式前向过程，token被替换为特殊掩码符号[M]：
$$q(x_t | x_0) = \prod_{i=1}^L \left[ \alpha_t \mathbf{1}(x_t^i=x_0^i) + (1-\alpha_t) \mathbf{1}(x_t^i=[\mathrm{M}]) \right]$$

#### 2.2.2 离散"分数"的定义
由于离散token没有梯度，SEDD引入**比率视图**作为离散分数的替代：
$$r_t(i,v | x_t) \propto \frac{p_t(\hat{x}_t^{(i\leftarrow v)})}{p_t(x_t)}$$

这个比率衡量了如果将掩码位置具体化为token $v$，序列的可能性会如何变化。

## 3. SEDD：首个离散扩散语言模型

### 3.1 核心贡献：分数熵
SEDD引入**分数熵**作为离散情况下的分数匹配替代方案：

#### 3.1.1 分数熵损失
$$\mathcal{L}_{\mathrm{SE}} = \mathbb{E}_{x\sim p} \left[ \sum_{y\neq x} w_{xy} \left( s_\theta(x)_y - \frac{p(y)}{p(x)}\log s_\theta(x)_y + K\left(\frac{p(y)}{p(x)}\right) \right) \right]$$

#### 3.1.2 可扩展的去噪分数熵
通过扰动视角，SEDD证明分数熵等价于去噪分数熵：
$$\mathcal{L}_{\mathrm{DSE}} = \mathbb{E}_{x_0\sim p_0} \mathbb{E}_{x\sim p(\cdot|x_0)} \left[ \sum_{y\neq x} w_{xy} \left( s_\theta(x)_y - \frac{p(y|x_0)}{p(x|x_0)}\log s_\theta(x)_y \right) \right]$$

### 3.2 实践挑战
- **时间步依赖性**：SEDD的比率估计本质上是时间步依赖的
- **需要时间条件网络**：通常需要显式的$t$嵌入来表示不同时间步的效果

## 4. RADD：简化离散扩散的关键洞察

### 4.1 核心理论突破
RADD提供了关键的理论澄清：**在吸收扩散中，具体分数可以分解为时间无关条件概率和解析时间相关标量的乘积**：

$$r_t(i,v | x_t) = c(t) \cdot p(x_0^i=v | x_t)$$

### 4.2 实际意义
- **移除时间条件**：不再需要Transformer接受$t$作为输入来表示扩散时间步效应
- **简化训练目标**：训练目标简化为类似AR/MLM的去噪交叉熵
- **更好的缓存性**：输出可以在采样过程中重用，提高推理效率

### 4.3 损失函数简化
RADD的损失函数：
$$\mathcal{L}_{\text{RADD}}(\theta) = \mathbb{E}_{t} \mathbb{E}_{x_0, x_t\sim q(\cdot|x_0,t)} \left[ -\sum_{i: x_t^i=[\mathrm{M}]} w(t)\,\log \pi_\theta(x_0^i | x_t) \right]$$

## 5. SMDM：大规模扩散语言模型的可扩展性

### 5.1 扩展定律证据
SMDM建立了首个针对文本掩码扩散模型(MDM)的扩展定律研究：
- **训练可扩展性**：证明扩散LM在训练方面可以与AR模型相媲美
- **计算差距**：在受控设置下，与AR模型相比只有相对较小的计算差距

### 5.2 推理效率挑战
- **NFE问题**：生成通常需要多次去噪迭代，延迟随步骤数线性增长
- **系统级加速需求**：需要专门的系统优化来减少推理成本

### 5.3 简单指导机制
SMDM提出使用类似分类器自由指导的简单指导机制来改善生成质量：
$$\text{logits} \leftarrow (1+\gamma)\,\text{logits}_{\text{cond}} - \gamma\,\text{logits}_{\text{uncond}}$$

## 6. Fast-dLLM：解决KV缓存问题

### 6.1 KV缓存失效的根本原因
在全注意力扩散LM中，标准KV缓存失效的原因：

1. **双向依赖性**：每个位置的表示可能依赖于所有其他位置，包括未来位置
2. **重复掩码变化**：序列内容因掩码/取消掩码而反复变化，导致隐藏状态不稳定

### 6.2 Fast-dLLM解决方案
- **块状近似KV缓存**：为扩散LM设计的块状近似KV缓存
- **置信度引导的并行解码**：使用置信度感知选择，只接受足够可靠的预测

## 7. AR→块扩散转换：最佳实践

### 7.1 块扩散的基本思想
将长度为$L$的序列划分为$K$个连续块：
$$x = (b_1,\dots,b_k),\quad b_k \in \mathcal{V}^D, \quad KD=L$$

块扩散保持块间的AR结构：
$$P_\theta(x) = \prod_{k=1}^{K} P_\theta(b_k | b_{<k})$$

但在每个块内使用**块内掩码扩散/迭代去噪**，允许并行token更新。

### 7.2 专门的注意力掩码设计
Fast-dLLM v2和SDAR使用专门的$2L\times 2L$注意力掩码：

#### 7.2.1 训练时掩码结构
将噪声序列$x_t$和干净序列$x_0$连接成长度为$2L$的输入：
$$\tilde{x} = [x_t; x_0] \in \mathcal{V}^{2L}$$

构建$2\times 2$块形式的完整掩码：
$$\mathcal{M}_{\text{full}} = \begin{bmatrix} \mathcal{M}_{\mathrm{BD}} & \mathcal{M}_{\mathrm{OBC}} \\ 0 & \mathcal{M}_{\mathrm{BC}} \end{bmatrix}$$

- **$\mathcal{M}_{\mathrm{BD}}$**：块对角双向注意力（在$x_t$内部）
- **$\mathcal{M}_{\mathrm{OBC}}$**：偏移块因果注意力（从$x_t$到$x_0$前缀）
- **$0$**：防止从$x_t$泄漏回$x_0$
- **$\mathcal{M}_{\mathrm{BC}}$**：块因果注意力（在$x_0$内部）

### 7.3 SDAR vs Fast-dLLM v2

#### 7.3.1 SDAR特点
- 轻量级AR到块扩散转换
- 保留预训练AR模型的强大全局一致性
- 在块内实现并行性和双向细化

#### 7.3.2 Fast-dLLM v2改进
- **移位预测**：使用AR对齐的预测路径，保持AR表示稳定性
- **互补掩码**：提供更强、更平衡的监督信号
- **分层缓存**：支持高吞吐量推理

## 8. 结论与未来方向

### 8.1 核心洞察总结
1. **连续扩散**：噪声条件化使分数学习可扩展
2. **离散扩散**：SEDD用比率替代梯度，RADD简化时间步依赖
3. **可扩展性**：SMDM证明扩散LM可以遵循有竞争力的扩展趋势
4. **部署优化**：系统和转换技术弥合了部署差距

### 8.2 实践建议
- **小规模实验**：从RADD开始，避免SEDD的时间条件复杂性
- **大规模训练**：参考SMDM的扩展定律指导
- **生产部署**：优先考虑块扩散转换方法（SDAR/Fast-dLLM v2）
- **推理优化**：结合KV缓存和置信度引导的并行解码

### 8.3 未来研究方向
- **更高效的采样算法**：减少去噪步骤数而不损失质量
- **混合架构**：结合AR和扩散优势的新型架构
- **多模态扩散**：将扩散思想扩展到多模态生成
- **理论分析**：深入理解扩散LM的表示能力和泛化性能

---
**资料来源**：Fuliang Liu的DLLM调查报告（2025）
**PPT链接**：https://fvliang.github.io/ppt/dllmsurvey2025.pptx
**LaTeX源码**：https://fvliang.github.io/tex/dllmsurvey2025.zip