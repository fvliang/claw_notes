#!/usr/bin/env python3
"""Batch translate abstracts and generate bilingual AI summaries using Kimi API."""

import json
import os
import time
from pathlib import Path

import requests

DB_PATH = Path("database.json")
API_KEY = os.environ.get("KIMI_API_KEY", "")
API_URL = "https://api.moonshot.cn/v1/chat/completions"

SUMMARY_PROMPT_EN = """You are an AI research assistant specialized in LLM systems. Summarize the following academic paper in 2-3 concise sentences in English. Focus on: 1) What problem it solves, 2) Key method/insight, 3) Main results or impact. Be technical and precise."""

SUMMARY_PROMPT_CN = """你是一位专精于大模型系统的研究助理。请用2-3句简洁的中文总结以下学术论文。重点关注：1）解决了什么问题，2）核心方法/洞见，3）主要结果或影响。要求技术准确、表述精炼。"""


def load_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def call_kimi(messages, max_retries=3, timeout=60):
    """Call Kimi API with retry."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "moonshot-v1-8k",
        "messages": messages,
        "temperature": 0.3
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=data, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None


def generate_summary_en(abstract_en, title, topic):
    """Generate English AI summary of the paper."""
    if not abstract_en or len(abstract_en) < 50:
        return None

    messages = [
        {"role": "system", "content": SUMMARY_PROMPT_EN},
        {"role": "user", "content": f"Title: {title}\nTopic: {topic}\nAbstract: {abstract_en[:2500]}"}
    ]
    return call_kimi(messages)


def generate_summary_cn(abstract_en, title, topic):
    """Generate Chinese AI summary of the paper."""
    if not abstract_en or len(abstract_en) < 50:
        return None

    messages = [
        {"role": "system", "content": SUMMARY_PROMPT_CN},
        {"role": "user", "content": f"Title: {title}\nTopic: {topic}\nAbstract: {abstract_en[:2500]}"}
    ]
    return call_kimi(messages)


def translate_abstract(abstract_en):
    """Translate English abstract to Chinese."""
    if not abstract_en or len(abstract_en) < 50:
        return None

    messages = [
        {"role": "system", "content": "You are a professional academic translator. Translate the following academic paper abstract from English to Chinese. Maintain technical accuracy and academic tone. Only output the translation, no explanations."},
        {"role": "user", "content": abstract_en[:3000]}
    ]
    return call_kimi(messages)


def process_papers(batch_size=3, max_papers=None, skip_existing=True):
    """Process papers in batches: generate bilingual AI summaries + translate abstracts."""
    db = load_db()
    papers = db["papers"]

    # Find papers needing processing
    need_processing = []
    for p in papers:
        has_en = bool(p.get("abstract_en")) and len(p.get("abstract_en", "")) > 50
        if not has_en:
            continue

        needs_en_summary = not p.get("ai_summary_en")
        needs_cn_summary = not p.get("ai_summary_cn")
        needs_cn_abstract = not p.get("abstract_cn") or "[中文翻译待补充]" in str(p.get("abstract_cn", "")) or "[自动翻译生成中...]" in str(p.get("abstract_cn", ""))

        if skip_existing and not needs_en_summary and not needs_cn_summary and not needs_cn_abstract:
            continue

        need_processing.append((p, needs_en_summary, needs_cn_summary, needs_cn_abstract))

    if max_papers:
        need_processing = need_processing[:max_papers]

    print(f"Found {len(need_processing)} papers to process")

    total = len(need_processing)
    processed = 0

    for i in range(0, total, batch_size):
        batch = need_processing[i:i+batch_size]
        print(f"\n--- Batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ---")

        for p, needs_en, needs_cn, needs_abstract_cn in batch:
            print(f"Processing [{p['id']}]: {p['title'][:50]}...")

            # Generate English summary
            if needs_en:
                print("  Generating EN summary...")
                summary = generate_summary_en(p.get("abstract_en", ""), p.get("title", ""), p.get("topic", ""))
                if summary:
                    p["ai_summary_en"] = summary.strip()
                    print(f"  EN: {summary[:80]}...")
                else:
                    print("  EN summary failed")
                time.sleep(1)

            # Generate Chinese summary
            if needs_cn:
                print("  Generating CN summary...")
                summary = generate_summary_cn(p.get("abstract_en", ""), p.get("title", ""), p.get("topic", ""))
                if summary:
                    p["ai_summary_cn"] = summary.strip()
                    print(f"  CN: {summary[:80]}...")
                else:
                    print("  CN summary failed")
                time.sleep(1)

            # Translate abstract to Chinese
            if needs_abstract_cn:
                print("  Translating abstract...")
                cn = translate_abstract(p.get("abstract_en", ""))
                if cn:
                    p["abstract_cn"] = cn.strip()
                    print(f"  Abstract translated ({len(cn)} chars)")
                else:
                    print("  Translation failed")
                time.sleep(1)

            processed += 1

        # Save progress every batch
        save_db(db)
        print(f"Saved progress: {processed}/{total}")

        # Rate limit between batches
        if i + batch_size < total:
            time.sleep(3)

    print(f"\nDone! Processed {processed} papers")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch process papers for bilingual AI summaries")
    parser.add_argument("--batch-size", type=int, default=3, help="Batch size (default: 3)")
    parser.add_argument("--max-papers", type=int, default=None, help="Max papers to process")
    parser.add_argument("--no-skip", action="store_true", help="Re-process papers that already have summaries")
    args = parser.parse_args()

    process_papers(batch_size=args.batch_size, max_papers=args.max_papers, skip_existing=not args.no_skip)
