"""Configuration manager."""
import os
from pathlib import Path

# Resolve repo root dynamically
REPO_ROOT = Path(__file__).parent.parent.resolve()
DB_PATH = REPO_ROOT / "database.json"
PAPERS_DIR = REPO_ROOT / "papers"
INDEX_HTML = REPO_ROOT / "index.html"

# arXiv API settings
ARXIV_API_BASE = "http://export.arxiv.org/api/query"
ARXIV_MAX_RESULTS = 100
ARXIV_DELAY_BETWEEN_QUERIES = 3.0  # seconds
ARXIV_BATCH_SIZE = 10  # IDs per batch fetch
ARXIV_BATCH_DELAY = 6.0  # seconds between batches
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 5.0

# Search keywords (ordered by precision, high-precision first)
SEARCH_QUERIES = [
    "speculative decoding",
    "llm serving",
    "kv cache optimization",
    "llm inference system",
    "paged attention",
    "continuous batching llm",
    "llm inference acceleration",
    "prefill decode disaggregation",
]

SERVING_KEYWORDS = [
    "serving", "inference", "speculative decoding", "kv cache", "kv-cache",
    "prefill", "decode", "batching", "scheduling", "throughput", "latency",
    "memory management", "paged attention", "vllm", "continuous batching",
    "disaggregation", "offloading", "acceleration", "ttft", "tpot",
    "inference speedup", "inference latency", "inference throughput",
    "inference optimization", "efficient inference", "inference system",
    "inference framework", "inference engine", "inference acceleration",
    "self-speculative", "early exit", "layer skipping",
    "parallel decoding", "speculative execution",
    "distributed inference", "edge inference", "on-device inference",
    "gpu memory", "draft model", "verification", "acceptance rate",
    "load balancing", "request routing",
    "flash attention", "attention kernel",
    "moe inference", "moe serving",
    "long-context inference", "generation latency",
    "cost-efficient inference", "model deployment",
    "speculative sampling", "prefix caching", "token eviction",
    "kv compression", "cache compression",
    "lora serving", "adapter serving",
    "token budget", "request scheduling",
    "attention sink", "streaming llm",
    "chunked prefill", "microbatch",
    "speculative drafting", "verification head",
    "distillation inference", "model compression serving",
    "tensor parallel", "pipeline parallel", "data parallel",
    "speculative", "draft", "medusa", "eagle",
]

TRAINING_EXCLUSIONS = [
    "training system", "distributed training", "fine-tuning system",
    "pre-training", "gradient", "optimizer",
    "safety alignment", "rlhf", "dpo",
    "question answering", "retrieval augmented generation",
    "prompt tuning", "sentiment analysis",
    "code generation benchmark", "math reasoning benchmark",
    "drug discovery", "protein", "bioinformatics",
    "medical diagnosis", "clinical",
    "autonomous driving", "robotics manipulation",
    "image generation", "video generation",
    "speech recognition", "asr", "tts",
    "music generation", "art generation",
]

# LLM-related terms (must have at least one)
LLM_TERMS = [
    "llm", "large language model",
    "language model inference", "language model serving",
    "transformer inference", "transformer serving",
    "foundation model inference", "autoregressive inference",
    "autoregressive model", "generative model inference",
    "diffusion llm", "mamba inference", "state space model inference",
]

# Strong signals (if present, high confidence)
STRONG_SIGNALS = [
    "serving system", "inference system", "inference framework", "inference engine",
    "speculative decoding", "kv cache", "prefill", "decode phase",
    "inference latency", "inference throughput", "inference speedup",
    "inference acceleration", "inference optimization", "efficient inference",
    "parallel decoding", "batched inference", "continuous batching",
    "paged attention", "vllm", "sglang", "tensorrt-llm",
    "memory footprint", "gpu utilization", "request scheduling",
    "disaggregated serving", "prefill-decode",
]
