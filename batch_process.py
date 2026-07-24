#!/usr/bin/env python3
"""Batch translate abstracts and generate AI summaries using Kimi API."""

import json
import os
import time
from pathlib import Path

import requests

DB_PATH = Path("database.json")
API_KEY = os.environ.get("KIMI_API_KEY", "")
API_URL = "https://api.moonshot.cn/v1/chat/completions"

def load_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def call_kimi(messages, max_retries=3):
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
            resp = requests.post(API_URL, headers=headers, json=data, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None


def translate_abstract(abstract_en):
    """Translate English abstract to Chinese."""
    if not abstract_en or len(abstract_en) < 50:
        return None
    
    messages = [
        {"role": "system", "content": "You are a professional academic translator. Translate the following academic paper abstract from English to Chinese. Maintain technical accuracy and academic tone. Only output the translation, no explanations."},
        {"role": "user", "content": abstract_en[:3000]}  # Limit length
    ]
    return call_kimi(messages)


def generate_summary(abstract_en, title, topic):
    """Generate AI summary of the paper."""
    if not abstract_en or len(abstract_en) < 50:
        return None
    
    messages = [
        {"role": "system", "content": "You are an AI research assistant. Summarize the following academic paper in 2-3 sentences in Chinese. Focus on: 1) What problem it solves, 2) Key method/insight, 3) Main results. Be concise and technical."},
        {"role": "user", "content": f"Title: {title}\nTopic: {topic}\nAbstract: {abstract_en[:2000]}"}
    ]
    return call_kimi(messages)


def process_papers(batch_size=5, max_papers=None):
    """Process papers in batches."""
    db = load_db()
    papers = db["papers"]
    
    # Find papers needing processing
    need_processing = []
    for p in papers:
        needs_cn = not p.get("abstract_cn") or "[中文翻译待补充]" in str(p.get("abstract_cn", "")) or "[自动翻译生成中...]" in str(p.get("abstract_cn", ""))
        needs_summary = not p.get("ai_summary") or "[AI总结生成中...]" in str(p.get("ai_summary", ""))
        has_en = bool(p.get("abstract_en")) and len(p.get("abstract_en", "")) > 50
        
        if (needs_cn or needs_summary) and has_en:
            need_processing.append((p, needs_cn, needs_summary))
    
    if max_papers:
        need_processing = need_processing[:max_papers]
    
    print(f"Found {len(need_processing)} papers to process")
    
    total = len(need_processing)
    processed = 0
    
    for i in range(0, total, batch_size):
        batch = need_processing[i:i+batch_size]
        print(f"\n--- Batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ---")
        
        for p, needs_cn, needs_summary in batch:
            print(f"Processing [{p['id']}]: {p['title'][:50]}...")
            
            # Translate to Chinese
            if needs_cn:
                print("  Translating...")
                cn = translate_abstract(p.get("abstract_en", ""))
                if cn:
                    p["abstract_cn"] = cn
                    print(f"  ✓ Translated ({len(cn)} chars)")
                else:
                    print("  ✗ Translation failed")
                time.sleep(1)
            
            # Generate AI summary
            if needs_summary:
                print("  Generating summary...")
                summary = generate_summary(p.get("abstract_en", ""), p.get("title", ""), p.get("topic", ""))
                if summary:
                    p["ai_summary"] = summary
                    print(f"  ✓ Summary generated ({len(summary)} chars)")
                else:
                    print("  ✗ Summary failed")
                time.sleep(1)
            
            processed += 1
        
        # Save progress every batch
        save_db(db)
        print(f"Saved progress: {processed}/{total}")
        
        # Rate limit between batches
        time.sleep(2)
    
    print(f"\nDone! Processed {processed} papers")


if __name__ == "__main__":
    # Process first 50 papers as a test
    process_papers(batch_size=3, max_papers=50)
