"""AI summary generation for papers."""
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import requests

from .models import Paper

logger = logging.getLogger("clawbot.summarizer")

API_KEY = os.environ.get("BAILIAN_API_KEY", "")
API_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
API_MODEL = "qwen3.8-max-preview"
MAX_WORKERS = 200

SUMMARY_PROMPT_EN = """You are an AI research assistant specialized in LLM systems. Summarize the following academic paper in 2-3 concise sentences in English. Focus on: 1) What problem it solves, 2) Key method/insight, 3) Main results or impact. Be technical and precise."""

SUMMARY_PROMPT_CN = """你是一位专精于大模型系统的研究助理。请用2-3句简洁的中文总结以下学术论文。重点关注：1）解决了什么问题，2）核心方法/洞见，3）主要结果或影响。要求技术准确、表述精炼。"""


def _call_api(messages, max_retries=3, timeout=90):
    """Call Bailian API with retry."""
    if not API_KEY:
        logger.warning("BAILIAN_API_KEY not set, skipping AI summary generation")
        return None

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": API_MODEL,
        "messages": messages,
        "temperature": 0.3
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=data, timeout=timeout)
            resp.raise_for_status()
            result = resp.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning("API call failed (attempt %d): %s", attempt + 1, e)
            time.sleep(2 ** attempt)
    return None


def generate_summary_en(paper: Paper) -> Optional[str]:
    """Generate English AI summary for a paper."""
    abstract = paper.abstract_en or ""
    if not abstract or len(abstract) < 50:
        return None
    messages = [
        {"role": "system", "content": SUMMARY_PROMPT_EN},
        {"role": "user", "content": f"Title: {paper.title}\nTopic: {paper.topic}\nAbstract: {abstract[:2500]}"}
    ]
    return _call_api(messages)


def generate_summary_cn(paper: Paper) -> Optional[str]:
    """Generate Chinese AI summary for a paper."""
    abstract = paper.abstract_en or ""
    if not abstract or len(abstract) < 50:
        return None
    messages = [
        {"role": "system", "content": SUMMARY_PROMPT_CN},
        {"role": "user", "content": f"Title: {paper.title}\nTopic: {paper.topic}\nAbstract: {abstract[:2500]}"}
    ]
    return _call_api(messages)


def _process_single_paper(paper: Paper):
    """Generate both EN and CN summaries for one paper."""
    needs_en = not paper.ai_summary_en
    needs_cn = not paper.ai_summary_cn

    if not needs_en and not needs_cn:
        return paper.id, None, None

    en_result = None
    cn_result = None

    with ThreadPoolExecutor(max_workers=2) as ex:
        future_en = ex.submit(generate_summary_en, paper) if needs_en else None
        future_cn = ex.submit(generate_summary_cn, paper) if needs_cn else None

        if future_en:
            try:
                en_result = future_en.result(timeout=120)
            except Exception as e:
                logger.warning("EN summary failed for paper %d: %s", paper.id, e)
        if future_cn:
            try:
                cn_result = future_cn.result(timeout=120)
            except Exception as e:
                logger.warning("CN summary failed for paper %d: %s", paper.id, e)

    return paper.id, en_result, cn_result


def generate_summaries_for_papers(papers: List[Paper], batch_size: int = 50) -> int:
    """Generate AI summaries for papers missing them. Returns count processed."""
    need_summary = []
    for p in papers:
        has_abstract = bool(p.abstract_en) and len(p.abstract_en) > 50
        needs_en = not p.ai_summary_en
        needs_cn = not p.ai_summary_cn
        if has_abstract and (needs_en or needs_cn):
            need_summary.append(p)

    if not need_summary:
        logger.info("No papers need AI summaries")
        return 0

    total = len(need_summary)
    logger.info("Generating AI summaries for %d papers", total)

    processed = 0
    success = 0

    # Process in batches
    for chunk_start in range(0, total, batch_size):
        chunk = need_summary[chunk_start:chunk_start + batch_size]
        chunk_num = chunk_start // batch_size + 1
        total_chunks = (total - 1) // batch_size + 1
        logger.info("Summary batch %d/%d (%d papers)", chunk_num, total_chunks, len(chunk))

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_paper = {executor.submit(_process_single_paper, p): p for p in chunk}

            for future in as_completed(future_to_paper):
                paper = future_to_paper[future]
                try:
                    pid, en_summary, cn_summary = future.result(timeout=180)
                    if en_summary:
                        paper.ai_summary_en = en_summary.strip()
                    if cn_summary:
                        paper.ai_summary_cn = cn_summary.strip()
                    if en_summary or cn_summary:
                        success += 1
                except Exception as e:
                    logger.warning("Failed to process paper %d: %s", paper.id, e)

                processed += 1
                if processed % 10 == 0:
                    logger.info("Summary progress: %d/%d", processed, total)

    logger.info("AI summaries complete: %d/%d papers processed", success, total)
    return success
