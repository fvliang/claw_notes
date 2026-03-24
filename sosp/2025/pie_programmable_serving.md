# Pie: A Programmable Serving System for Emerging LLM Applications

- **会议**: SOSP 2025
- **arXiv**: [2510.24051](https://arxiv.org/abs/2510.24051)
- **GitHub**: [pie-project/pie](https://github.com/pie-project/pie)
- **作者**: In Gim 等
- **年份**: 2025

## 摘要

Emerging large language model (LLM) applications involve diverse reasoning strategies and agentic workflows, straining the capabilities of existing serving systems built on a monolithic token generation loop. This paper introduces **Pie**, a programmable LLM serving system designed for flexibility and efficiency. 

Pie decomposes the traditional generation loop into fine-grained service handlers exposed via an API and delegates control of the generation process to user-provided programs, called **inferlets**. This enables applications to implement new KV cache strategies, bespoke generation logic, and seamlessly integrate computation and I/O—entirely within the application, without requiring modifications to the serving system.

Pie executes inferlets using WebAssembly, benefiting from its lightweight sandboxing.

## 核心贡献

1. **可编程的LLM服务系统**: 将传统的生成循环分解为细粒度的服务处理器
2. **Inferlets**: 用户提供的程序，可以控制生成过程
3. **WebAssembly沙箱**: 轻量级沙箱执行

## 实验结果

- 在标准任务上与SOTA性能相当（3-12%延迟开销）
- 在agentic工作负载上显著提升延迟和吞吐量（1.3x-3.4x更高）

---

*论文来源：SOSP 2025，数据真实可验证*