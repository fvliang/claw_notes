# P/D-Serve: Serving Disaggregated Large Language Model at Scale

**Authors:** Yibo Jin, Tao Wang, Huimin Lin, Mingyang Song, Peiyang Li, Yipeng Ma, Yicheng Shan, Zhengfan Yuan, Cailong Li, Yajing Sun, Tiandeng Wu, Xing Chu, Ruizhi Huan, Li Ma, Xiao You, Wenting Zhou, Yunpeng Ye, Wen Liu, Xiangkun Xu, Yongsheng Zhang, Tiantian Dong, Jiawei Zhu, Zhe Wang, Xijian Ju, Jianxun Song 等

**Conference:** arXiv 2024

**Year:** 2024

**ArXiv:** [2408.08147](<https://arxiv.org/abs/2408.08147>)

**Topic:** Disaggregated Serving

---

## Abstract (English)

Serving disaggregated large language models (LLMs) over tens of thousands of xPU devices (GPUs or NPUs) with reliable performance faces multiple challenges. 1) Ignoring the diversity (various prefixes and tidal requests), treating all the prompts in a mixed pool is inadequate. To facilitate the similarity per scenario and minimize the inner mismatch on P/D (prefill and decoding) processing, fine-grained organization is required, dynamically adjusting P/D ratios for better performance. 2) Due to inaccurate estimation on workload (queue status or maintained connections), the global scheduler easily incurs unnecessary timeouts in prefill. 3) Block-fixed device-to-device (D2D) KVCache transfer over cluster-level RDMA (remote direct memory access) fails to achieve desired D2D utilization as expected. To overcome previous problems, this paper proposes an end-to-end system P/D-Serve, which models end-to-end (E2E) P/D performance and enables: 1) fine-grained P/D organization, mapping the service with RoCE as needed; 2) on-demand forwarding upon rejections for idle prefill; and 3) efficient KVCache transfer via optimized D2D access. P/D-Serve is implemented upon Ascend and MindSpore, has been deployed over tens of thousands of NPUs for more than eight months in commercial use, and further achieves 60%, 42% and 46% improvements on E2E throughput, TTFT SLO and D2D transfer time.

## Abstract (Chinese / 中文摘要)

在数万个xPU设备(GPU或NPU)上服务解耦的大语言模型(LLM)并保持可靠性能面临多重挑战。1)忽略多样性（各种前缀和潮汐请求），将所有提示放在混合池中是不充分的。为了促进每个场景的相似性并最小化P/D(prefill和decoding)处理的内部不匹配，需要细粒度组织，动态调整P/D比率以获得更好的性能。2)由于对工作负载的不准确估计，全局调度器容易在prefill中产生不必要的超时。3)块固定的设备到设备(D2D) KVCache传输通过集群级RDMA无法实现预期的D2D利用率。为了克服这些问题，本文提出端到端系统P/D-Serve，它建模端到端(E2E)P/D性能并启用：1)细粒度P/D组织；2)空闲prefill的按需转发；3)通过优化的D2D访问的高效KVCache传输。P/D-Serve在Ascend和MindSpore上实现，已在数万个NPU上商业部署超过八个月。

---

*Auto-collected from arXiv on 2026-04-17*
