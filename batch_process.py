#!/usr/bin/env python3
"""Batch generate bilingual AI summaries using Bailian API with parallel requests."""

import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

DB_PATH = Path("database.json")
API_KEY = os.environ.get("BAILIAN_API_KEY", "")
API_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
API_MODEL = "qwen3.8-max-preview"
MAX_WORKERS = 200  # Parallel API calls

SUMMARY_PROMPT_EN = """You are an AI research assistant specialized in LLM systems. Summarize the following academic paper in 2-3 concise sentences in English. Focus on: 1) What problem it solves, 2) Key method/insight, 3) Main results or impact. Be technical and precise."""

SUMMARY_PROMPT_CN = """你是一位专精于大模型系统的研究助理。请用2-3句简洁的中文总结以下学术论文。重点关注：1）解决了什么问题，2）核心方法/洞见，3）主要结果或影响。要求技术准确、表述精炼。"""


def load_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def call_api(messages, max_retries=3, timeout=90):
    """Call Bailian API with retry."""
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
            time.sleep(2 ** attempt)
    return None


def generate_summary_en(paper):
    """Generate English AI summary."""
    abstract = paper.get("abstract_en", "")
    if not abstract or len(abstract) < 50:
        return None
    messages = [
        {"role": "system", "content": SUMMARY_PROMPT_EN},
        {"role": "user", "content": f"Title: {paper.get('title', '')}\nTopic: {paper.get('topic', '')}\nAbstract: {abstract[:2500]}"}
    ]
    return call_api(messages)


def generate_summary_cn(paper):
    """Generate Chinese AI summary."""
    abstract = paper.get("abstract_en", "")
    if not abstract or len(abstract) < 50:
        return None
    messages = [
        {"role": "system", "content": SUMMARY_PROMPT_CN},
        {"role": "user", "content": f"Title: {paper.get('title', '')}\nTopic: {paper.get('topic', '')}\nAbstract: {abstract[:2500]}"}
    ]
    return call_api(messages)


def process_single_paper(paper):
    """Process one paper: generate EN and CN summaries in parallel threads."""
    pid = paper['id']
    title = paper.get('title', '')[:40]
    needs_en = not paper.get("ai_summary_en")
    needs_cn = not paper.get("ai_summary_cn")

    if not needs_en and not needs_cn:
        return pid, None, None

    en_result = None
    cn_result = None

    # Launch EN and CN in parallel
    with ThreadPoolExecutor(max_workers=2) as ex:
        future_en = ex.submit(generate_summary_en, paper) if needs_en else None
        future_cn = ex.submit(generate_summary_cn, paper) if needs_cn else None

        if future_en:
            try:
                en_result = future_en.result(timeout=120)
            except Exception as e:
                pass
        if future_cn:
            try:
                cn_result = future_cn.result(timeout=120)
            except Exception as e:
                pass

    return pid, en_result, cn_result


def process_papers_parallel(batch_size=200, max_papers=None):
    """Process papers with parallel API calls."""
    db = load_db()
    papers = db["papers"]

    # Find papers needing processing
    need_processing = []
    for p in papers:
        has_en = bool(p.get("abstract_en")) and len(p.get("abstract_en", "")) > 50
        if not has_en:
            continue
        needs_en = not p.get("ai_summary_en")
        needs_cn = not p.get("ai_summary_cn")
        if not needs_en and not needs_cn:
            continue
        need_processing.append(p)

    if max_papers:
        need_processing = need_processing[:max_papers]

    total = len(need_processing)
    print(f"Found {total} papers to process, max_workers={MAX_WORKERS}")

    completed = 0
    processed_ids = set()

    # Process in chunks to allow periodic DB saves
    for chunk_start in range(0, total, batch_size):
        chunk = need_processing[chunk_start:chunk_start + batch_size]
        chunk_num = chunk_start // batch_size + 1
        total_chunks = (total - 1) // batch_size + 1
        print(f"\n--- Chunk {chunk_num}/{total_chunks} ({len(chunk)} papers) ---")

        # Build a map for quick lookup
        paper_map = {p['id']: p for p in papers}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_paper = {executor.submit(process_single_paper, p): p for p in chunk}

            for future in as_completed(future_to_paper):
                p = future_to_paper[future]
                try:
                    pid, en_summary, cn_summary = future.result(timeout=180)
                    paper = paper_map.get(pid)
                    if paper:
                        if en_summary:
                            paper["ai_summary_en"] = en_summary.strip()
                        if cn_summary:
                            paper["ai_summary_cn"] = cn_summary.strip()
                        processed_ids.add(pid)
                except Exception as e:
                    print(f"  Error processing paper {p['id']}: {e}")

                completed += 1
                if completed % 10 == 0:
                    print(f"  Progress: {completed}/{total}")

        # Save after each chunk
        save_db(db)
        print(f"Saved progress: {len(processed_ids)}/{total}")

    print(f"\nDone! Processed {len(processed_ids)} papers")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--max-papers", type=int, default=None)
    args = parser.parse_args()

    process_papers_parallel(batch_size=args.batch_size, max_papers=args.max_papers)
