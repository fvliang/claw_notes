# Efficiently Aligning Draft Models via Parameter- and Data-Efficient Adaptation (EDA)

**arXiv**: 2603.09527
**链接**: https://arxiv.org/abs/2603.09527
**作者**: Luxi Lin, Zhihang Lin, Zhanpeng Zeng, Yuhao Chen, Qingyu Zhang, Jixiang Luo, Xuelong Li, Rongrong Ji
**会议**: arXiv 2026
**主题**: llm_serving / Speculative Decoding / Draft Model Adaptation
**GitHub**: https://github.com/Lyn-Lucy/Efficient-Draft-Adaptation

## 摘要 (Abstract)

Speculative decoding accelerates LLM inference but suffers from performance degradation when target models are fine-tuned for specific domains. A naive solution is to retrain draft models for every target model, which is costly and inefficient. To address this, we introduce a parameter- and data-efficient framework named Efficient Draft Adaptation (EDA), for efficiently adapting draft models. EDA introduces three innovations: (1) a decoupled architecture that utilizes shared and private components to model the shared and target-specific output distributions separately, enabling parameter-efficient adaptation by updating only the lightweight private component; (2) a data regeneration strategy that utilizes the fine-tuned target model to regenerate training data, thereby improving the alignment between training and speculative decoding, leading to higher average acceptance length; (3) a sample selection mechanism that prioritizes high-value data for efficient adaptation. Our experiments show that EDA effectively restores speculative performance on fine-tuned models, achieving superior average acceptance lengths with significantly reduced training costs compared to full retraining.

## 摘要 (中文)

推测解码加速 LLM 推理，但当目标模型针对特定领域微调时性能下降。朴素的解决方案是为每个目标模型重新训练 draft 模型，成本高且低效。为此我们提出 EDA（Efficient Draft Adaptation），一个参数和数据高效的框架用于高效适配 draft 模型。EDA 引入三个创新：（1）解耦架构，利用共享和私有组件分别建模共享和目标特定的输出分布，通过仅更新轻量级私有组件实现参数高效适配；（2）数据再生策略，利用微调后的目标模型再生训练数据，改善训练与推测解码之间的对齐，提高平均接受长度；（3）样本选择机制，优先选择高价值数据以实现高效适配。实验表明 EDA 有效恢复了微调模型上的推测性能，以显著降低的训练成本实现了更优的平均接受长度。

## 关键贡献

1. 解耦架构：共享 + 私有组件分离建模
2. 数据再生策略：利用目标模型再生训练数据
3. 样本选择机制：优先高价值数据
4. GitHub 开源：https://github.com/Lyn-Lucy/Efficient-Draft-Adaptation