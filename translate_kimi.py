#!/usr/bin/env python3
"""Batch translate and summarize using KIMI API."""

import json
import os
import time
from pathlib import Path

import requests

DB_PATH = Path("database.json")
API_KEY = os.environ.get("KIMI_API_KEY", "")
API_URL = "https://api.moonshot.cn/v1/chat/completions"
MODEL = "kimi-k2p6"

def load_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def kimi_chat(prompt):
    """Call KIMI API."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 2048
    }
    
    try:
        resp = requests.post(API_URL, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  API error: {e}")
        return None


def translate_abstract(abstract_en, title):
    """Translate abstract to Chinese."""
    if not abstract_en or len(abstract_en) < 50:
        return None
    
    prompt = f"""请将以下学术论文摘要翻译成中文。保持学术准确性，使用专业术语。

论文标题：{title}

英文摘要：
{abstract_en}

请只输出中文翻译，不要添加额外说明："""
    
    result = kimi_chat(prompt)
    if result:
        # Clean up
        result = result.strip()
        if result.startswith("中文翻译："):
            result = result[5:].strip()
        if result.startswith("翻译："):
            result = result[3:].strip()
    return result


def generate_summary(abstract_en, title, topic):
    """Generate AI summary."""
    if not abstract_en or len(abstract_en) < 50:
        return None
    
    prompt = f"""请为以下学术论文生成一段简短的中文总结（200字以内）。总结应包含：研究问题、方法、主要贡献。

论文标题：{title}
方向：{topic}

英文摘要：
{abstract_en}

请只输出总结，不要添加额外说明："""
    
    result = kimi_chat(prompt)
    if result:
        result = result.strip()
        if len(result) > 500:
            result = result[:500] + "..."
    return result


def process_batch(papers_batch):
    """Process a batch of papers."""
    db = load_db()
    papers = {p["id"]: p for p in db["papers"]}
    
    for p in papers_batch:
        paper = papers.get(p["id"])
        if not paper:
            continue
        
        print(f"\n[{p['id']}] {paper['title'][:40]}...")
        
        # Translate
        if not paper.get("abstract_cn") or "[中文翻译" in str(paper.get("abstract_cn", "")) or "[自动翻译" in str(paper.get("abstract_cn", "")):
            print("  Translating...")
            cn = translate_abstract(paper.get("abstract_en", ""), paper.get("title", ""))
            if cn:
                paper["abstract_cn"] = cn
                print(f"  ✓ CN ({len(cn)} chars)")
            else:
                print("  ✗ Failed")
            time.sleep(1)
        
        # Summary
        if not paper.get("ai_summary") or "[AI总结" in str(paper.get("ai_summary", "")):
            print("  Summarizing...")
            summary = generate_summary(paper.get("abstract_en", ""), paper.get("title", ""), paper.get("topic", ""))
            if summary:
                paper["ai_summary"] = summary
                print(f"  ✓ Summary ({len(summary)} chars)")
            else:
                print("  ✗ Failed")
            time.sleep(1)
    
    save_db(db)
    print(f"\nSaved {len(papers_batch)} papers")


def main():
    db = load_db()
    papers = db["papers"]
    
    # Find papers needing processing
    need_processing = []
    for p in papers:
        needs_cn = not p.get("abstract_cn") or "[中文翻译待补充]" in str(p.get("abstract_cn", "")) or "[自动翻译生成中...]" in str(p.get("abstract_cn", ""))
        needs_summary = not p.get("ai_summary") or "[AI总结生成中...]" in str(p.get("ai_summary", ""))
        has_en = bool(p.get("abstract_en")) and len(p.get("abstract_en", "")) > 50
        
        if (needs_cn or needs_summary) and has_en:
            need_processing.append(p)
    
    print(f"Found {len(need_processing)} papers to process")
    
    # Process in batches of 5
    batch_size = 5
    for i in range(0, len(need_processing), batch_size):
        batch = need_processing[i:i+batch_size]
        print(f"\n=== Batch {i//batch_size + 1}/{(len(need_processing)-1)//batch_size + 1} ===")
        process_batch(batch)
        time.sleep(2)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
