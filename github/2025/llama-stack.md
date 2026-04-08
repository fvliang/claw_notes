# Llama Stack

- **GitHub**: https://github.com/llamastack/llama-stack
- **Stars**: ⭐
- **Conference/Source**: GitHub
- **Year**: 2025

## 摘要 (CN)

Llama Stack是一个开源的AI应用代理API服务器，提供OpenAI兼容的API，可以在任何地方运行——笔记本电脑、数据中心或云端。使用任何OpenAI兼容的客户端或代理框架。可以在不更改应用代码的情况下在Llama、GPT、Gemini、Mistral或任何模型之间切换。

## 摘要 (EN)

Llama Stack is an open-source agentic API server for building AI applications. OpenAI-compatible. Any model, any infrastructure.

## 特性

- **OpenAI兼容API**: 完全兼容OpenAI API，可以作为drop-in replacement
- **多模型支持**: 支持Llama、GPT、Gemini、Mistral等模型
- **可插拔Provider架构**: 支持Ollama、vLLM等多种后端
- **Chat Completions & Embeddings**: 标准API端点
- **Responses API**: 服务端代理编排，支持工具调用和MCP集成
- **Vector Stores & Files**: 托管文档存储和搜索
- **Batches**: 离线批处理支持

## 安装

```bash
# One-line install
curl -LsSf https://github.com/llamastack/llama-stack/raw/main/scripts/install.sh | bash

# Or install via uv
uv pip install llama-stack

# Start the server
llama stack run
```

## 使用示例

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")
response = client.chat.completions.create(
    model="llama-3.3-70b",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Provider支持

- Ollama (本地开发)
- vLLM (生产部署)
- 托管服务

## 适用场景

Llama Stack适合构建AI应用，特别是需要多模型支持和代理功能的场景。它提供了统一的API接口，可以方便地在不同模型和基础设施之间切换。