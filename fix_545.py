#!/usr/bin/env python3
import json
import urllib.request
import os

API_KEY = os.environ.get('KIMI_API_KEY', '')
db = json.load(open('database.json'))

p545 = next((p for p in db['papers'] if p['id'] == 545), None)
if not p545 or not p545.get('abstract_en'):
    print("No abstract for 545")
    exit()

print(f"Title: {p545['title']}")
print(f"Abstract: {p545['abstract_en'][:100]}...")

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
        print(f"Error: {e}")
        return None

print("Translating...")
cn = kimi_chat([
    {"role": "system", "content": "将以下英文学术论文摘要翻译成简洁准确的中文。保持学术语气。只返回翻译结果。"},
    {"role": "user", "content": p545['abstract_en']}
])
if cn:
    p545['abstract_cn'] = cn
    print(f"CN: {cn[:80]}...")

print("Summarizing...")
summary = kimi_chat([
    {"role": "system", "content": "用1-2句话总结这篇论文的核心贡献。中文，简洁，不超过100字。"},
    {"role": "user", "content": f"Title: {p545['title']}\nAbstract: {p545['abstract_en']}"}
])
if summary:
    p545['ai_summary'] = summary
    print(f"Summary: {summary[:80]}...")

with open('database.json', 'w', encoding='utf-8') as f:
    json.dump(db, f, ensure_ascii=False, indent=2)

print("Done!")
