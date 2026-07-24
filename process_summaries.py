#!/usr/bin/env python3
"""Batch translate abstracts and generate AI summaries using LLM API."""

import json
import os
import time
from pathlib import Path

DB_PATH = Path("database.json")

def load_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def translate_and_summarize_papers(batch_size=10):
    """Process papers in batches: translate CN + generate AI summary."""
    db = load_db()
    papers = db["papers"]
    
    # Find papers needing processing
    need_processing = []
    for p in papers:
        needs_cn = not p.get("abstract_cn") or "[中文翻译待补充]" in str(p.get("abstract_cn", ""))
        needs_summary = not p.get("ai_summary")
        if needs_cn or needs_summary:
            need_processing.append((p, needs_cn, needs_summary))
    
    print(f"Found {len(need_processing)} papers needing processing")
    print(f"Batch size: {batch_size}")
    
    # Process in batches
    total = len(need_processing)
    processed = 0
    
    for i in range(0, total, batch_size):
        batch = need_processing[i:i+batch_size]
        print(f"\n--- Batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ---")
        
        for p, needs_cn, needs_summary in batch:
            print(f"Processing: {p['title'][:60]}...")
            
            # TODO: Call LLM API here
            # For now, just mark placeholder
            if needs_cn and p.get("abstract_en"):
                p["abstract_cn"] = f"[自动翻译生成中...] {p['abstract_en'][:100]}..."
            
            if needs_summary and p.get("abstract_en"):
                p["ai_summary"] = f"[AI总结生成中...] 本文研究了{p['topic']}相关的问题。"
            
            processed += 1
        
        # Save progress every batch
        save_db(db)
        print(f"Saved progress: {processed}/{total}")
        
        # Rate limit
        time.sleep(1)
    
    print(f"\nDone! Processed {processed} papers")


if __name__ == "__main__":
    translate_and_summarize_papers()
