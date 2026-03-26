# inferinse - High-Throughput Speculative Decoding Gateway

## 项目信息

- **类型**: GitHub开源项目
- **编程语言**: TypeScript
- **更新日期**: 2025年2月9日

## 原文链接

- **GitHub**: https://github.com/tyscript11/inferinse

## 介绍

inferinse是一个高吞吐量的speculative decoding网关，通过协调快速的draft模型与更强的target模型来减少推理延迟。该系统设计用于生产环境，支持大规模部署。

## 主要特性

1. 快速draft模型协调
2. 高吞吐量架构
3. 生产级稳定性

## 技术细节

inferinse采用网关架构，将draft模型和target模型的协调在独立的服务层处理，实现了更好的资源隔离和扩展性。