#!/usr/bin/env python3
import json

db = json.load(open('database.json'))
papers = db['papers']
topics = db['topics']

# 按会议-年份分组
conf_year_papers = {}
for p in papers:
    conf = p.get('conference', 'arXiv')
    year = p.get('year', 2024)
    key = f"{conf}|{year}"
    if key not in conf_year_papers:
        conf_year_papers[key] = []
    conf_year_papers[key].append(p)

# 生成会议-年份目录
conf_years = sorted(set(k.split('|')[0] + '/' + str(k.split('|')[1]) for k in conf_year_papers.keys()))

html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>📚 LLM Serving 论文库</title>
    <style>
        * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
        body { font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif; background: #f5f5f5; margin: 0; padding: 0; padding-bottom: 100px; }
        
        /* 头部 */
        .header {
            background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
            color: white;
            padding: 20px;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .header h1 { margin: 0; font-size: 22px; font-weight: 600; }
        .header .subtitle { opacity: 0.9; font-size: 13px; margin-top: 5px; }
        
        /* 统计 */
        .stats { display: flex; gap: 20px; margin-top: 15px; }
        .stat { text-align: center; }
        .stat .num { font-size: 22px; font-weight: 700; }
        .stat .label { font-size: 11px; opacity: 0.8; }
        
        /* 导航 */
        .nav-bar {
            background: white;
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
            position: sticky;
            top: 70px;
            z-index: 99;
            display: flex;
            gap: 10px;
            overflow-x: auto;
        }
        .nav-bar::-webkit-scrollbar { display: none; }
        .nav-chip {
            padding: 8px 14px;
            background: #f0f0f0;
            border-radius: 18px;
            font-size: 13px;
            white-space: nowrap;
            cursor: pointer;
            flex-shrink: 0;
        }
        .nav-chip.active { background: #007AFF; color: white; }
        .nav-chip.back { background: #e8f4ff; color: #007AFF; }
        
        /* 搜索 */
        .search-box { padding: 15px; background: white; }
        .search-box input {
            width: 100%;
            padding: 12px 16px;
            border: none;
            border-radius: 12px;
            background: #f5f5f5;
            font-size: 15px;
        }
        
        /* 论文列表 */
        .paper-list { padding: 15px; }
        .section-title {
            font-size: 15px;
            font-weight: 600;
            color: #333;
            margin: 20px 0 12px 0;
            padding-left: 5px;
            border-left: 3px solid #007AFF;
        }
        .paper-card {
            background: white;
            border-radius: 14px;
            padding: 16px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .paper-card .title {
            font-size: 15px;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 8px;
            line-height: 1.4;
        }
        .paper-card .authors { font-size: 12px; color: #888; margin-bottom: 10px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .paper-card .meta { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
        .tag { padding: 3px 10px; border-radius: 10px; font-size: 11px; }
        .tag.conf { background: #e8f4ff; color: #007AFF; }
        .tag.topic { background: #f0f0f0; color: #666; }
        .tag.year { background: #fff3e0; color: #FF9500; }
        
        .paper-card .links { display: flex; gap: 12px; margin-top: 10px; padding-top: 10px; border-top: 1px solid #f5f5f5; }
        .paper-card .links a { font-size: 13px; color: #007AFF; text-decoration: none; font-weight: 500; }
        
        /* 摘要展开 */
        .abstract-toggle {
            display: inline-flex;
            gap: 6px;
            padding: 8px 12px;
            background: #f5f5f5;
            border: none;
            border-radius: 8px;
            font-size: 12px;
            color: #666;
            cursor: pointer;
            margin-top: 8px;
        }
        .abstract-content { display: none; margin-top: 12px; }
        .abstract-content.show { display: block; }
        .abstract-content h4 { font-size: 13px; color: #666; margin: 12px 0 6px 0; }
        .abstract-content p { font-size: 13px; line-height: 1.6; color: #333; margin: 0; }
        .abstract-content .cn { color: #666; }
        
        /* 详情页 */
        .detail-page { display: none; }
        .detail-page.show { display: block; }
        .detail-header { background: white; padding: 20px; border-bottom: 1px solid #eee; }
        .detail-header .back-btn { display: inline-flex; align-items: center; gap: 6px; font-size: 14px; color: #007AFF; margin-bottom: 15px; cursor: pointer; }
        .detail-header h2 { font-size: 18px; margin: 0 0 10px 0; line-height: 1.4; }
        .detail-header .authors { font-size: 13px; color: #666; margin-bottom: 12px; }
        .detail-header .meta { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 15px; }
        
        .detail-body { padding: 20px; background: white; margin: 15px; border-radius: 14px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }
        .detail-body h3 { font-size: 15px; color: #333; margin: 20px 0 10px 0; padding-bottom: 8px; border-bottom: 1px solid #eee; }
        .detail-body h3:first-child { margin-top: 0; }
        .detail-body p { font-size: 14px; line-height: 1.8; color: #444; margin: 0 0 15px 0; }
        .detail-body .cn { color: #666; }
        
        .footer { text-align: center; padding: 20px; color: #999; font-size: 12px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📚 LLM Serving 论文库</h1>
        <div class="subtitle">Speculative Decoding · KV Cache · MoE</div>
        <div class="stats">
            <div class="stat"><div class="num" id="paperCount">-</div><div class="label">论文</div></div>
            <div class="stat"><div class="num" id="confCount">-</div><div class="label">会议</div></div>
            <div class="stat"><div class="num" id="topicCount">-</div><div class="label">主题</div></div>
        </div>
    </div>
    
    <!-- 列表页 -->
    <div id="listPage">
        <div class="nav-bar" id="navBar">
            <div class="nav-chip active" data-filter="">全部</div>
        </div>
        <div class="search-box">
            <input type="text" id="searchInput" placeholder="搜索论文标题、作者...">
        </div>
        <div class="paper-list" id="paperList"></div>
    </div>
    
    <!-- 详情页 -->
    <div class="detail-page" id="detailPage">
        <div class="detail-header">
            <div class="back-btn" onclick="showList()">← 返回列表</div>
            <h2 id="detailTitle"></h2>
            <div class="authors" id="detailAuthors"></div>
            <div class="meta" id="detailMeta"></div>
            <div class="links" id="detailLinks"></div>
        </div>
        <div class="detail-body" id="detailBody"></div>
    </div>
    
    <div class="footer">🔄 自动更新 · <span id="totalCount">0</span> 篇论文</div>

    <script>
    const papers = ''' + json.dumps(papers, ensure_ascii=False) + ''';
    const topics = ''' + json.dumps(topics) + ''';
    
    // 按会议-年份分组
    const confYearPapers = {};
    papers.forEach(p => {
        const conf = p.conference || 'arXiv';
        const year = p.year || 2024;
        const key = conf + '|' + year;
        if (!confYearPapers[key]) confYearPapers[key] = [];
        confYearPapers[key].push(p);
    });
    
    const confYears = Object.keys(confYearPapers).sort();
    const conferences = [...new Set(papers.map(p => p.conference).filter(c => c))];
    
    let currentFilter = "";
    let currentSearch = "";
    
    function initNav() {
        const navBar = document.getElementById('navBar');
        
        // 添加主题筛选
        topics.forEach(t => {
            const chip = document.createElement('div');
            chip.className = 'nav-chip';
            chip.dataset.filter = t;
            chip.textContent = t;
            chip.onclick = () => setFilter(t);
            navBar.appendChild(chip);
        });
    }
    
    function setFilter(topic) {
        currentFilter = topic;
        document.querySelectorAll('.nav-chip').forEach(c => {
            c.classList.toggle('active', c.dataset.filter === topic);
        });
        render();
    }
    
    function render() {
        const search = document.getElementById('searchInput').value.toLowerCase();
        currentSearch = search;
        
        let filtered = papers.filter(p => {
            const matchFilter = !currentFilter || p.topic === currentFilter;
            const matchSearch = !search || 
                (p.title && p.title.toLowerCase().includes(search)) || 
                (p.authors && p.authors.toLowerCase().includes(search));
            return matchFilter && matchSearch;
        });
        
        const list = document.getElementById('paperList');
        
        if (filtered.length === 0) {
            list.innerHTML = '<div style="text-align:center;padding:40px;color:#999;">没有找到匹配的论文</div>';
            return;
        }
        
        // 按会议-年份分组显示
        let html = '';
        const grouped = {};
        filtered.forEach(p => {
            const conf = p.conference || 'arXiv';
            const year = p.year || 2024;
            const key = conf + ' / ' + year;
            if (!grouped[key]) grouped[key] = [];
            grouped[key].push(p);
        });
        
        Object.keys(grouped).sort().forEach(key => {
            html += `<div class="section-title">📁 ${key}</div>`;
            grouped[key].forEach(p => {
                html += renderPaperCard(p);
            });
        });
        
        list.innerHTML = html;
        
        document.getElementById('paperCount').textContent = papers.length;
        document.getElementById('confCount').textContent = conferences.length;
        document.getElementById('topicCount').textContent = topics.length;
        document.getElementById('totalCount').textContent = papers.length;
    }
    
    function renderPaperCard(p) {
        return `
        <div class="paper-card" data-id="${p.id}">
            <div class="title">${p.title || 'Untitled'}</div>
            <div class="authors">${p.authors || 'Unknown'}</div>
            <div class="meta">
                <span class="tag conf">${p.conference || 'arXiv'}</span>
                <span class="tag year">${p.year || '2024'}</span>
                <span class="tag topic">${p.topic || 'LLM Serving'}</span>
            </div>
            <div class="links">
                ${p.arxiv_id ? `<a href="https://arxiv.org/abs/${p.arxiv_id}" target="_blank">📄 arXiv</a>` : ''}
                ${p.github_repo ? `<a href="https://github.com/${p.github_repo}" target="_blank">🐙 GitHub</a>` : ''}
                <a href="javascript:void(0)" onclick="showDetail(${p.id})">📖 详情 →</a>
            </div>
            ${(p.abstract_en || p.abstract_cn) ? `
            <button class="abstract-toggle" onclick="toggleAbstract(${p.id}, this)">
                📖 展开摘要
            </button>
            <div class="abstract-content" id="abstract-${p.id}">
                ${p.abstract_en ? `<h4>Abstract</h4><p>${p.abstract_en.substring(0,600)}${p.abstract_en.length>600?'...':''}</p>` : ''}
                ${p.abstract_cn ? `<h4>摘要</h4><p class="cn">${p.abstract_cn.substring(0,400)}${p.abstract_cn.length>400?'...':''}</p>` : ''}
            </div>
            ` : ''}
        </div>`;
    }
    
    function toggleAbstract(id, btn) {
        const content = document.getElementById('abstract-' + id);
        if (content.classList.contains('show')) {
            content.classList.remove('show');
            btn.textContent = '📖 展开摘要';
        } else {
            content.classList.add('show');
            btn.textContent = '📕 收起摘要';
        }
    }
    
    function showDetail(id) {
        const p = papers.find(p => p.id === id);
        if (!p) return;
        
        document.getElementById('listPage').style.display = 'none';
        document.getElementById('detailPage').classList.add('show');
        
        document.getElementById('detailTitle').textContent = p.title;
        document.getElementById('detailAuthors').textContent = p.authors || 'Unknown';
        
        document.getElementById('detailMeta').innerHTML = `
            <span class="tag conf">${p.conference || 'arXiv'}</span>
            <span class="tag year">${p.year || '2024'}</span>
            <span class="tag topic">${p.topic || 'LLM Serving'}</span>
        `;
        
        let links = '';
        if (p.arxiv_id) links += `<a href="https://arxiv.org/abs/${p.arxiv_id}" target="_blank">📄 arXiv</a> `;
        if (p.github_repo) links += `<a href="https://github.com/${p.github_repo}" target="_blank">🐙 GitHub</a> `;
        document.getElementById('detailLinks').innerHTML = links;
        
        let body = '';
        if (p.abstract_en) {
            body += `<h3>Abstract</h3><p>${p.abstract_en}</p>`;
        }
        if (p.abstract_cn) {
            body += `<h3>摘要</h3><p class="cn">${p.abstract_cn}</p>`;
        }
        if (p.intro_en) {
            body += `<h3>1. Introduction</h3><p>${p.intro_en}</p>`;
        }
        if (p.intro_cn) {
            body += `<h3>引言</h3><p class="cn">${p.intro_cn}</p>`;
        }
        
        document.getElementById('detailBody').innerHTML = body || '<p style="color:#999;text-align:center;padding:40px;">暂无详细内容</p>';
        
        window.scrollTo(0, 0);
    }
    
    function showList() {
        document.getElementById('detailPage').classList.remove('show');
        document.getElementById('listPage').style.display = 'block';
    }
    
    document.getElementById('searchInput').addEventListener('input', render);
    
    initNav();
    render();
    </script>
</body>
</html>'''

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Done! Generated {len(papers)} papers")