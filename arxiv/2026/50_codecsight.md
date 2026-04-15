---
title: CodecSight: Leveraging Video Codec Signals for Efficient Streaming VLM Inference
authors: Yulin Zou, Yan Chen, Wenyan Chen, JooYoung Park, Shivaraman Nitin, Luo Tao, Francisco Romero, Dmitrii Ustiugov
arxiv_id: 
conference: arxiv
full_conference: ARXIV 2026
year: "2026"
topic: LLM Serving
url: 
pdf_url: 
added_date: 2026-04-15
---

# CodecSight: Leveraging Video Codec Signals for Efficient Streaming VLM Inference

## 论文信息

- **arXiv**: 
- **会议**: ARXIV 2026
- **作者**: Yulin Zou, Yan Chen, Wenyan Chen, JooYoung Park, Shivaraman Nitin, Luo Tao, Francisco Romero, Dmitrii Ustiugov
- **主题**: LLM Serving

## 摘要 (Abstract)

Video streaming analytics is a crucial workload for vision-language model serving, but the high cost of multimodal token generation creates significant inference overhead. We present CodecSight, a system that leverages video codec signals (motion vectors, residual frames, and macroblock partitioning) already computed during video encoding to skip redundant visual token generation, reducing the computational cost of VLM inference for streaming video workloads.

## 摘要中文

视频流分析是视觉语言模型服务的关键工作负载，但多模态token生成的高成本创造了显著的推理开销。我们提出了CodecSight，一个利用视频编码期间已计算的视频编解码信号（运动向量、残差帧和宏块分区）来跳过冗余视觉token生成的系统，降低了流视频工作负载的VLM推理计算成本。

## 引言 (Introduction)

Streaming video analysis with VLMs is extremely costly because each frame requires full visual tokenization. However, consecutive frames in video streams share substantial visual similarity—information already captured by video codecs.

## 引言中文

使用VLM的流视频分析极其昂贵，因为每帧需要完整的视觉token化。然而，视频流中的连续帧共享大量视觉相似性——这些信息已由视频编解码器捕获。

## 主要贡献

1. (待补充)

## 原文链接

- arXiv: (待确认)
- GitHub: (待补充)

## 补充材料

- 博客: (待补充)
- 相关GitHub: (待补充)

## 备注
