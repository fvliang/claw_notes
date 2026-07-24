#!/usr/bin/env python3
"""Complete fix: fetch missing abstracts + translate + summarize + regen web."""
import json
import urllib.request
import os
import time

API_KEY = os.environ.get('KIMI_API_KEY', '')

db = json.load(open('database.json'))
papers = db['papers']

# Revert bad match for ID 548
for p in papers:
    if p['id'] == 548 and p.get('arxiv_id') == '2407.11310v2':
        p['arxiv_id'] = None
        p['abstract_en'] = None
        print("Reverted bad match for ID 548")

missing = [p for p in papers if not p.get('abstract_en')]
print(f"Missing abstracts: {len(missing)}")

# Try Semantic Scholar with longer delays
def fetch_ss(title):
    q = urllib.parse.quote(title)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}&fields=title,abstract,externalIds&limit=3"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "clawbot/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        return None
    for paper in data.get('data', []):
        abstract = paper.get('abstract', '')
        if abstract:
            ptitle = paper.get('title', '')
            t_norm = title.lower().replace('-', ' ')
            p_norm = ptitle.lower().replace('-', ' ')
            if t_norm in p_norm or p_norm in t_norm or len(set(t_norm.split()) & set(p_norm.split())) >= 3:
                return abstract
    return None

# Fetch missing abstracts
for p in missing:
    print(f"\n[{p['id']}] {p['title'][:50]}...")
    abstract = fetch_ss(p['title'])
    if abstract:
        print(f"  Found ({len(abstract)} chars)")
        p['abstract_en'] = abstract
    else:
        print("  Not found")
    time.sleep(5)

# Now translate and summarize all newly fetched
needs_work = [p for p in papers if p.get('abstract_en') and not p.get('abstract_cn')]
print(f"\nNeed translation: {len(needs_work)}")

def kimi_chat(messages):
    data = json.dumps({
        "model": "kimi-k2p6",
        "messages": messages,
        "temperature": 0.3
    }).encode('utf-8')
    req = urllib.request.Request(
        "https://api.moonshot.cn/v1/chat/completions",
        data=data,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"  API error: {e}")
        return None

for p in needs_work:
    print(f"\n[{p['id']}] {p['title'][:40]}...")
    
    # Translate
    if not p.get('abstract_cn') or p.get('abstract_cn') == '[中文翻译待补充] ...':
        print("  Translating...")
        cn = kimi_chat([
            {"role": "system", "content": "将以下英文学术论文摘要翻译成简洁准确的中文。保持学术语气。只返回翻译结果。"},
            {"role": "user", "content": p['abstract_en']}
        ])
        if cn:
            p['abstract_cn'] = cn
            print(f"  CN: {cn[:60]}...")
        time.sleep(1)
    
    # AI Summary
    if not p.get('ai_summary'):
        print("  Summarizing...")
        summary = kimi_chat([
            {"role": "system", "content": "用1-2句话总结这篇论文的核心贡献。中文，简洁，不超过100字。"},
            {"role": "user", "content": f"Title: {p['title']}\nAbstract: {p['abstract_en']}"}
        ])
        if summary:
            p['ai_summary'] = summary
            print(f"  Summary: {summary[:60]}...")
        time.sleep(1)

with open('database.json', 'w', encoding='utf-8') as f:
    json.dump(db, f, ensure_ascii=False, indent=2)

# Regenerate website
print("\nRegenerating website...")
import subprocess
subprocess.run(['python3', 'gen_web.py'], cwd='/root/.openclaw/workspace/claw_notes')

print("\nAll done!")
