# ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

**论文链接**: [arXiv:2603.17435](https://arxiv.org/abs/2603.17435)

**作者**: Ruibo Fan, Xiangrui Yu, Xinglin Pan, Zeyu Li, Weile Luo, Qiang Wang, Wei Wang, Xiaowen Chu

**会议**: ASPLOS '26

---

## Abstract (摘要)

Lossless model compression holds tremendous promise for alleviating the memory and bandwidth bottlenecks in bit-exact Large Language Model (LLM) serving. However, existing approaches often result in substantial inference slowdowns due to fundamental design mismatches with GPU architectures: at the kernel level, variable-length bitstreams produced by traditional entropy codecs break SIMT parallelism; at the system level, decoupled pipelines lead to redundant memory traffic. We present ZipServ, a lossless compression framework co-designed for efficient LLM inference. ZipServ introduces Tensor-Core-Aware Triple Bitmap Encoding (TCA-TBE), a novel fixed-length format that enables constant-time, parallel decoding, together with a fused decompression-GEMM (ZipGEMM) kernel that decompresses weights on-the-fly directly into Tensor Core registers. This "load-compressed, compute-decompressed" design eliminates intermediate buffers and maximizes compute intensity. Experiments show that ZipServ reduces the model size by up to 30%, achieves up to 2.21x kernel-level speedup over NVIDIA's cuBLAS, and expedites end-to-end inference by an average of 1.22x over vLLM. ZipServ is the first lossless compression system that provides both storage savings and substantial acceleration for LLM inference on GPUs.

---

无压缩模型压缩在实现位精确的大型语言模型（LLM）服务方面具有巨大潜力，可缓解内存和带宽瓶颈。然而，现有方法往往会导致显著的推理减速，这是由于与GPU架构的根本设计不匹配：在内核级别，传统熵编码器产生的可变长度位流破坏了SIMT并行性；在系统级别，解耦的管道导致冗余内存流量。我们提出了ZipServ，这是一种针对高效LLM推理共同设计的无损压缩框架。ZipServ引入了张量核感知的“三重位图编码”（TCA-TBE），这是一种新型的固定长度格式，可实现常量时间并行解码，同时还有一个融合的解压缩GEMM（ZipGEMM）内核，可即时将权重解压缩到张量核寄存器中。这种"加载压缩、计算解压缩"的设计消除了中间缓冲区并最大化了计算强度。实验表明，ZipServ可将模型大小减少高达30%，在NVIDIA的cuBLAS上实现高达2.21倍的内核级加速，并在vLLM上实现平均1.22倍的端到端加速。ZipServe是第一个在GPU上为LLM推理同时提供存储节省和显著加速的无损压缩系统。

---

## 1. Introduction (引言)

The transformative power of Large Language Models (LLMs) like GPT-4, LLaMA-3, and Qwen-3 is rooted in their massive scale, enabling a new paradigm of AI applications. However, this immense scale creates significant deployment challenges, making GPU memory and memory bandwidth the primary bottlenecks for LLM serving, especially in resource-constrained environments.

Model compression offers a promising solution for efficient LLM deployment. Most existing approaches are lossy, reducing size by approximating model weights via quantization (e.g., GPTQ, AWQ) or pruning (e.g., SparseGPT). However, such approximations risk accuracy loss. For instance, aggressive 4-bit quantization (e.g., MXFP4) slashes accuracy from 56.0% to 36.2% on LiveCodeBench, while even robust int8 quantization (GPTQ-int8) can cause up to 11.1% loss in long-context reasoning (NOCHA). These risks undermine reliability in safety-critical and user-facing settings, motivating approaches that guarantee bit-exact reproducibility and numerical integrity.

Lossless compression offers a compelling alternative by providing bit-exact model representation without accuracy loss. To date, its benefits have largely targeted storage and training workflows. For example, LMC and ZipNN employ Huffman to compress model checkpoints for efficient storage and distribution, while NeuZip and DietGPU mitigate memory and communication overhead during training. Although recent efforts, notably DFloat11, aim to extend these gains to inference, practical efficiency remains elusive. When integrated into serving pipelines, existing lossless techniques incur significant runtime overhead. The decoupled decompression step alone takes 1.56–3.44× the time of the core inference computation. This overhead forces an unpleasant tradeoff between memory efficiency and runtime efficiency.

We contend that this tradeoff is not fundamental but arises from a mismatch between conventional compression algorithms and modern GPU architectures. The issue manifests at two levels. At the kernel level, traditional entropy codecs (e.g., Huffman or ANS) produce variable-length bitstreams, whose decoding demands serialized, data-dependent operations. These are ill-suited to the lockstep, parallel SIMT execution model of GPU warps, resulting in severe control-flow divergence and compute underutilization. At the system level, most frameworks employ a decoupled inference pipeline: weights are fully decompressed into a global-memory buffer before kernel consumption. This staged execution results in redundant, high-latency memory accesses, eroding compression-provided bandwidth savings and reducing arithmetic intensity during inference.

To rectify these fundamental algorithm-hardware mismatches, we present ZipServ, the first lossless compression framework co-designed for high-performance LLM inference on GPUs. Our key observation is that the exponent bits of BFloat16 weights in LLMs exhibit a highly skewed, low-entropy distributions in contemporary models. Exploiting this statistical redundancy, we propose Tensor-Core-Aware Triple Bitmap Encoding (TCA-TBE), a fixed-length, bitmap-based weight format tailored to GPU architectures. Unlike variable-length entropy codecs, TCA-TBE enables constant-time, parallel decoding using lightweight bitwise operations, thereby eliminating control-flow divergence and aligning with the GPU's SIMT execution model. Paired with TCA-TBE, ZipServ devises a fused decompression-GEMM kernel (ZipGEMM). Rather than decompressing weights into global memory as an intermediate step, ZipGEMM performs on-the-fly decoding, delivering compressed weights directly into the register files that feed Tensor Core matrix multiplication units. This "load-compressed, compute-decompressed" design eliminates intermediate buffers, reduces data movement, and maximizes the overlap between computation and memory access. By jointly addressing both the kernel-level and system-level mismatches, ZipServ transforms the theoretical storage savings of lossless compression into tangible performance gains on inference-optimized GPUs.

We demonstrate ZipServ's effectiveness through comprehensive benchmarking against state-of-the-art lossless approaches, including DietGPU, vendor-optimized nvCOMP, and the Huffman-based DFloat11. Compared to these baselines, which uniformly suffer significant runtime overhead, ZipServ consistently delivers substantial accelerations at both the kernel and system level on various inference-optimized GPUs, including RTX4090, L40S, and RTX5090. Our fused ZipGEMM achieves speedups of up to 2.21× over NVIDIA's cuBLAS, and up to 5.53× over DFloat11, the fastest lossless compression pipeline. These kernel-level improvements translate into an average 1.22× end-to-end speedup compared to leading systems like vLLM. Our results demonstrate for the first time that when co-designed with hardware, lossless compression can provide both storage savings and substantial LLM inference acceleration.

---

像GPT-4、LLaMA-3和Qwen-3这样的大型语言模型（LLM）的变革力量源于其巨大规模，为人工智能应用开辟了新范式。然而，这种巨大的规模带来了显著的部署挑战，使得GPU内存和内存带宽成为LLM服务的主要瓶颈，特别是在资源受限的环境中。

模型压缩为高效的LLM部署提供了有希望的解决方案。大多数现有方法都是有损的，通过量化（例如GPTQ、AWQ）或剪枝（例如SparseGPT）来近似模型权重来减小尺寸。然而，这种近似存在精度损失的风险。例如，激进的4位量化（例如MXFP4）将LiveCodeBench上的准确率从56.0%降至36.2%，而即使是稳健的int8量化（GPTQ-int8）也可能导致长上下文推理（NOCHA）损失高达11.1%。这些风险损害了安全关键和面向用户场景的可靠性，促使人们寻求能够保证位精确可重复性和数值完整性的方法。

无损压缩通过提供位精确的模型表示而不损失精度，提供了一个引人注目的替代方案。迄今为止，其优势主要针对存储和训练工作流。例如，LMC和ZipNN使用霍夫曼压缩模型检查点以实现高效的存储和分发，而NeuZip和DietGPU减轻了训练期间的内存和通信开销。尽管最近的努力，特别是DFloat11，旨在将这些收益扩展到推理，但实际效率仍然难以捉摸。当集成到服务管道中时，现有无损技术会产生显著的运行时开销。仅解耦的解压缩步骤就需要核心推理计算时间的1.56-3.44倍。这种开销迫使人们在内存效率和运行时效率之间做出令人不快的权衡。

我们认为这种权衡并非根本性，而是源于传统压缩算法与现代GPU架构之间的不匹配。问题表现在两个层面。在内核级别，传统熵编码器（如霍夫曼或ANS）产生可变长度位流，其解码需要序列化的、数据依赖的操作。这些操作不适合GPU warp的锁定步进并行SIMT执行模型，导致严重的控制流分歧和计算利用不足。在系统级别，大多数框架采用解耦推理管道：权重在内核消费之前完全解压缩到全局内存缓冲区。这种分阶段执行导致冗余的高延迟内存访问，削弱了压缩提供的带宽节省并降低了推理期间的算术强度。

为了纠正这些基本的算法-硬件不匹配，我们提出了ZipServ，这是第一个为高性能GPU上的LLM推理共同设计的无损压缩框架。我们的关键观察是，当代模型中BFloat16权重的指数位呈现出高度倾斜的低熵分布。利用这种统计冗余，我们提出了张量核感知的“三重位图编码”（TCA-TBE），这是一种针对GPU架构量身定制的固定长度、基于位图的权重格式。与可变长度熵编码器不同，TCA-TBE使用轻量级位操作实现常量时间并行解码，从而消除控制流分歧并与GPU的SIMT执行模型保持一致。与TCA-TBE配合，ZipServ设计了一个融合的解压缩-GEMM内核（ZipGEMM）。ZipGEMM不是将权重解压缩到全局内存作为中间步骤，而是执行即时解码，将压缩权重直接传送到为张量核矩阵乘法单元提供数据的寄存器文件中。这种"加载压缩、计算解压缩"设计消除了中间缓冲区，减少了数据移动，并最大化了计算和内存访问之间的重叠。通过共同解决内核级和系统级的不匹配，ZipServ将无损压缩的理论存储节省转化为推理优化GPU上的切实性能提升。

我们通过对最先进的无损方法（包括DietGPU、供应商优化的nvCOMP和基于霍夫曼的DFloat11）进行全面基准测试来展示ZipServ的有效性。与这些基线（这些基线普遍存在显著的运行时开销）相比，ZipServ在各种推理优化GPU（包括RTX4090、L40S和RTX5090）上始终在内核和系统级别提供实质性加速。我们的融合ZipGEMM实现了比NVIDIA的cuBLAS高达2.21倍的加速，比最快的无损压缩管道DFloat11高达5.53倍的加速。这些内核级改进转化为与vLLM等领先系统相比平均1.22倍的端到端加速。我们的结果首次证明，当与硬件共同设计时，无损压缩可以同时提供存储节省和显著的LLM推理加速。

---

## 主要贡献

- 我们识别了传统基于熵的压缩与GPU架构之间的基本不匹配，揭示了阻碍高效推理的内核级和系统级瓶颈。
- 我们提出了TCA-TBE，这是一种针对SIMT执行和张量核平铺量身定制的固定长度、基于位图的编码，能够实现常量时间并行解码。
- 我们设计了ZipGEMM，这是一种新型内核，可直接在张量核寄存器中即时执行解压缩，消除中间内存缓冲区并最大化计算强度。
- 我们展示并评估了ZipServ，这是一种无损压缩的LLM推理框架，可在各种LLM和GPU上实现端到端加速，这是无损压缩可直接加速LLM服务的首个实际证据。