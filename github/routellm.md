# RouteLLM: LLM路由服务与评估框架

## 原文链接
- GitHub: https://github.com/lm-sys/RouteLLM
- Stars: 4.7k
- 博客: http://lmsys.org/blog/2024-07-01-routellm/
- 论文: https://arxiv.org/abs/2406.18665

## 概述
RouteLLM是一个用于服务和服务LLM路由器的框架。它可以智能地将简单查询路由到更便宜的模型，同时将复杂查询路由到更强大的模型，从而在降低成本的同时保持响应质量。

## 核心特性

### 1. 易于使用
- 可作为OpenAI客户端的直接替代品
- 或启动OpenAI兼容服务器进行路由

### 2. 开箱即用的训练路由器
- 可降低高达**85%**的成本
- 同时保持**95%**的GPT-4性能
- 在MT Bench等广泛使用的基准测试上表现优异

### 3. 高性价比
- 路由器实现与商业产品相同的性能
- 但成本低**40%以上**

### 4. 可扩展性
- 轻松扩展框架以包含新的路由器
- 跨多个基准测试比较路由器性能

## 性能数据
- 在MT Bench等常用基准测试上：
  - 成本降低：高达85%
  - 性能保持：95% GPT-4水平
  - 相比商业解决方案：便宜40%以上

## 技术架构

### MF路由器（Matrix Factorization）
- 默认使用最佳性能配置
- 基于成本阈值进行路由决策

### 工作原理
1. 路由器分析输入查询的复杂度
2. 简单查询路由到较弱/便宜的模型
3. 复杂查询路由到较强/昂贵的模型
4. 通过成本阈值控制成本和质量之间的权衡

## 安装

### 从PyPI安装
```bash
pip install "routellm[serve,eval]"
```

### 从源码安装
```bash
git clone https://github.com/lm-sys/RouteLLM.git
cd RouteLLM
pip install -e .[serve,eval]
```

## 使用示例

### Python SDK
```python
import os
from routellm.controller import Controller

os.environ["OPENAI_API_KEY"] = "sk-XXXXXX"
os.environ["ANYSCALE_API_KEY"] = "esecret_XXXXXX"

client = Controller(
    routers=["mf"],
    strong_model="gpt-4-1106-preview",
    weak_model="anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1",
)

# 设置成本阈值
response = client.chat.completions.create(
    model="router-mf-0.11593",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
```

### 启动OpenAI兼容服务器
```bash
export OPENAI_API_KEY=sk-XXXXXX
export ANYSCALE_API_KEY=esecret_XXXXXX
python -m routellm.openai_server --routers mf \
    --strong-model gpt-4-1106-preview \
    --weak-model anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1
```

### 阈值校准
```bash
python -m routellm.calibrate_threshold \
    --routers mf \
    --strong-model-pct 0.5 \
    --config config.example.yaml
```

## 支持的模型
RouteLLM利用LiteLLM支持广泛的开源和闭源模型的聊天完成功能：
- 支持任意OpenAI兼容端点
- 支持本地模型路由
- 支持多种模型提供商

## 路由策略
- **成本阈值控制**：通过设置成本阈值来控制成本和质量之间的权衡
- **性能校准**：根据接收到的查询类型校准阈值以最大化路由性能

---

*本文档由自动化任务生成于 2026-03-24*