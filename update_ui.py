import re

with open('docs/index.html', 'r') as f:
    content = f.read()

# Add CSS before </style>
style_end = content.find('</style>')
if style_end > 0:
    new_css = '''/* AI Summary section */
.ai-summary-box{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:16px 20px;border-radius:12px;margin:0 0 20px}
.ai-summary-label{font-size:11px;opacity:.8;margin-bottom:6px;text-transform:uppercase;letter-spacing:1px}
.ai-summary-text{font-size:14px;line-height:1.7;font-weight:500}

/* Section tabs */
.section-tabs{display:flex;gap:4px;margin:0 15px 12px;background:#fff;padding:4px;border-radius:10px}
.section-tab{flex:1;padding:10px;border:none;border-radius:8px;background:transparent;color:#666;font-size:13px;cursor:pointer;font-weight:500}
.section-tab.on{background:#007AFF;color:#fff}

.section-content{display:none;padding:20px;background:#fff;margin:0 15px 12px;border-radius:14px}
.section-content.on{display:block}
.section-content h4{font-size:13px;color:#007AFF;margin:0 0 10px}
.section-content p{font-size:14px;line-height:1.8;color:#333;margin:0}

'''
    content = content[:style_end] + new_css + content[style_end:]

# Replace detail page structure
old_detail = '<div class="detail" id="det">\n<div class="det-head">\n<div class="det-back" onclick="back()">← 返回列表</div>\n<div style="display:flex;align-items:center;gap:8px">\n<div id="detTitle" class="det-title" style="flex:1"></div>\n<span id="detFav" class="det-fav" onclick="togF(currentDetId)"></span>\n</div>\n<div id="detAuth" class="det-auth"></div>\n<div id="detTags" class="det-tags"></div>\n<div id="detLinks" class="det-links"></div>\n</div>\n<div id="detBody" class="det-body"></div>\n</div>'

new_detail = '<div class="detail" id="det">\n<div class="det-head">\n<div class="det-back" onclick="back()">← 返回列表</div>\n<div style="display:flex;align-items:center;gap:8px">\n<div id="detTitle" class="det-title" style="flex:1"></div>\n<span id="detFav" class="det-fav" onclick="togF(currentDetId)"></span>\n</div>\n<div id="detAuth" class="det-auth"></div>\n<div id="detTags" class="det-tags"></div>\n<div id="detLinks" class="det-links"></div>\n</div>\n<div id="detSummary" class="det-body" style="display:none"></div>\n<div class="section-tabs" id="detTabs" style="display:none">\n<button class="section-tab on" onclick="switchTab(\'abs\')">📄 摘要原文</button>\n<button class="section-tab" onclick="switchTab(\'cn\')">🌐 中文翻译</button>\n</div>\n<div id="tabAbs" class="section-content on" style="display:none"></div>\n<div id="tabCn" class="section-content" style="display:none"></div>\n</div>'

content = content.replace(old_detail, new_detail)

with open('docs/index.html', 'w') as f:
    f.write(content)

print('Updated docs/index.html')
