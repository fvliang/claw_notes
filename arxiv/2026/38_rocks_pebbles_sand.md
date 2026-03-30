# Rocks, Pebbles and Sand: Modality-aware Scheduling for Multimodal Large Language Model Inference

## 论文信息

- **标题**: Rocks, Pebbles and Sand: Modality-aware Scheduling for Multimodal Large Language Model Inference
- **作者**: Konstantinos Papaioannou, Thaleia Dimitra Doudali, et al.
- **arXiv**: 2603.26498
- **会议/来源**: arXiv
- **年份**: 2026
- **主题**: LLM Serving
- **提交日期**: 2026年3月27日

## 原文链接

- arXiv: https://arxiv.org/abs/2603.26498
- PDF: https://arxiv.org/pdf/2603.26498

## 摘要 (英文)

Multimodal Large Language Models (MLLMs) power platforms like ChatGPT, Gemini, and Copilot, enabling richer interactions with text, images, and videos. These heterogeneous workloads introduce additional inference stages, such as vision preprocessing and encoding, that inflate latency and memory demand. Existing LLM serving systems, optimized for text-only workloads, fail under multimodality: large requests (e.g., videos) monopolize resources, causing severe head-of-line blocking and performance degradation. Our key insight is that multimodal requests differ by orders of magnitude in resource demands, which we capture through a simple abstraction: videos behave like rocks, images like pebbles, and text like sand. We design RPS-Serve, a modality-aware scheduler that lets sand flow quickly through pebbles and rocks, ensuring interactive responsiveness while avoiding starvation. RPS-Serve classifies requests, prioritizes them dynamically, and applies aging to avoid starvation. Evaluation across state-of-the-art MLLMs shows that RPS-Serve reduces, on average, time-to-first-token (TTFT) by 54% overall, and by 78.5% for latency-critical requests, compared to current systems.

## 摘要 (中文)

多模态大型语言模型(MLLM)为ChatGPT、Gemini和Copilot等平台提供支持,实现更丰富的文本、图像和视频交互。这些异构工作负载引入了额外的推理阶段,如视觉预处理和编码,从而增加延迟和内存需求。现有优化仅针对纯文本工作负载的LLM服务系统在多模态场景下表现不佳:大型请求(如视频)垄断资源,导致严重的队头阻塞和性能下降。我们的关键洞察是,多模态请求的资源需求相差数个数量级,我们通过一个简单的抽象来捕捉:视频像石头,图像像鹅卵石,文本像沙子。我们设计RPS-Serve,一种模态感知调度器,让沙子快速流过鹅卵石和石头,确保交互响应性同时避免饥饿。RPS-Serve对请求进行分类,动态优先级排序,并应用老化机制以避免饥饿。在最先进的MLLM上的评估表明,与当前系统相比,RPS-Serve平均将首token时间(TTFT)减少54%,对于延迟关键请求减少78.5%。

## 引言 (英文)

The landscape of AI interactions has fundamentally shifted with the advent of Multimodal Large Language Models (MLLMs). Platforms like ChatGPT, Gemini, and Copilot now seamlessly integrate text, images, and videos, enabling richer and more natural user experiences. However, this multimodality introduces significant challenges for serving infrastructure. Unlike text-only workloads, multimodal requests exhibit orders of magnitude difference in resource demands - processing a video query requires vastly more computation than a text query.

Current LLM serving systems were designed with text-only workloads in mind. They assume relatively uniform request sizes and processing times. When faced with multimodal workloads, these systems suffer from severe performance degradation. Large requests like videos monopolize GPU resources, causing smaller text requests to wait excessively - a phenomenon known as head-of-line blocking.

## 引言 (中文)

多模态大型语言模型(MLLM)的出现从根本上改变了AI交互格局。ChatGPT、Gemini和Copilot等平台现在无缝集成文本、图像和视频,实现更丰富、更自然的用户体验。然而,这种多模态给服务基础设施带来了重大挑战。与纯文本工作负载不同,多模态请求的资源需求相差数个数量级——处理视频查询比文本查询需要更多的计算资源。

当前的LLM服务系统是针对纯文本工作负载设计的。它们假设请求大小和处理时间相对均匀。当面对多模态工作负载时,这些系统会出现严重的性能下降。大型请求(如视频)垄断GPU资源,导致较小的文本请求过度等待——这是一种被称为队头阻塞的现象。

## 核心贡献

1. **模态感知抽象**: 提出"石头、鹅卵石和沙子"的抽象,用于刻画不同模态请求的资源需求差异
2. **RPS-Serve调度器**: 设计模态感知调度器,让沙子快速流过鹅卵石和石头
3. **显著性能提升**: 相比当前系统,平均TTFT减少54%,延迟关键请求减少78.5%