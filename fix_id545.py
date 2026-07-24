#!/usr/bin/env python3
"""Translate and summarize the newly fetched abstract for ID 545."""
import json
import urllib.request

db = json.load(open('database.json'))
papers = db['papers']

p545 = next((p for p in papers if p['id'] == 545), None)
if not p545 or not p545.get('abstract_en'):
    print("ID 545 not found or no abstract")
    exit()

abstract_en = p545['abstract_en']
print(f"Abstract: {abstract_en[:200]}...")

# Translate to CN using Kimi API
api_key = "fecd983b5a08c8402c695bdc248db9bf2b25c7a0d395289e1b9c35d3"

def kimi_chat(messages):
    data = json.dumps({
        "model": "kimi-k2p6",
        "messages": messages,
        "temperature": 0.3
    }).encode('utf-8')
    req = urllib.request.Request(
        "https://api.moonshot.cn/v1/chat/completions",
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error: {e}")
        return None

# Translate
print("Translating...")
translation = kimi_chat([
    {"role": "system", "content": "将以下英文学术论文摘要翻译成简洁准确的中文。保持学术语气。只返回翻译结果，不要解释。"},
    {"role": "user", "content": abstract_en}
])
if translation:
    p545['abstract_cn'] = translation
    print(f"Translation: {translation[:100]}...")

# AI Summary
print("Generating AI summary...")
summary = kimi_chat([
    {"role": "system", "content": "用1-2句话总结这篇论文的核心贡献和创新点。中文，简洁，不超过100字。"},
    {"role": "user", "content": f"Title: {p545['title']}\nAbstract: {abstract_en}"}
])
if summary:
    p545['ai_summary'] = summary
    print(f"Summary: {summary[:100]}...")

with open('database.json', 'w', encoding='utf-8') as f:
    json.dump(db, f, ensure_ascii=False, indent=2)

print("Done!")
