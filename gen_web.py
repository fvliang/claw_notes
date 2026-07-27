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
    <title>claw notes | LLM Serving Papers</title>
    <style>
        :root {
            --bg: #0f0f12;
            --bg-card: #18181b;
            --bg-elevated: #232329;
            --border: #2a2a32;
            --text: #e4e4e7;
            --text-secondary: #a1a1aa;
            --text-muted: #71717a;
            --accent: #6366f1;
            --accent-soft: #818cf8;
            --accent-glow: rgba(99, 102, 241, 0.15);
            --success: #22c55e;
            --tag-bg: rgba(99, 102, 241, 0.1);
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --shadow: 0 1px 3px rgba(0,0,0,0.3);
            --shadow-lg: 0 8px 30px rgba(0,0,0,0.4);
        }

        * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", Roboto, "Noto Sans SC", sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }

        /* Header */
        .header {
            background: linear-gradient(180deg, var(--bg-card) 0%, var(--bg) 100%);
            padding: 24px 20px 16px;
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 100;
            backdrop-filter: blur(20px);
        }
        .header-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }
        .header h1 {
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, var(--text) 0%, var(--accent-soft) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .header .subtitle {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 2px;
        }
        .header-icon {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--accent) 0%, #8b5cf6 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            box-shadow: 0 2px 12px var(--accent-glow);
        }

        /* Stats */
        .stats {
            display: flex;
            gap: 24px;
        }
        .stat {
            display: flex;
            flex-direction: column;
        }
        .stat .num {
            font-size: 20px;
            font-weight: 700;
            color: var(--text);
            line-height: 1.2;
        }
        .stat .label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Search */
        .search-box {
            padding: 12px 16px;
            background: var(--bg);
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 108px;
            z-index: 99;
        }
        .search-box input {
            width: 100%;
            padding: 10px 16px;
            border: 1px solid var(--border);
            border-radius: 100px;
            background: var(--bg-card);
            color: var(--text);
            font-size: 14px;
            outline: none;
            transition: all 0.2s;
        }
        .search-box input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }
        .search-box input::placeholder { color: var(--text-muted); }

        /* Nav */
        .nav-bar {
            display: flex;
            gap: 8px;
            padding: 12px 16px;
            background: var(--bg);
            border-bottom: 1px solid var(--border);
            overflow-x: auto;
            scrollbar-width: none;
        }
        .nav-bar::-webkit-scrollbar { display: none; }

        .nav-chip, .filter-chip {
            padding: 6px 14px;
            border-radius: 100px;
            font-size: 13px;
            font-weight: 500;
            white-space: nowrap;
            cursor: pointer;
            flex-shrink: 0;
            transition: all 0.15s ease;
            border: 1px solid transparent;
            background: var(--bg-card);
            color: var(--text-secondary);
        }
        .nav-chip:hover, .filter-chip:hover {
            background: var(--bg-elevated);
            color: var(--text);
        }
        .nav-chip.active, .filter-chip.active {
            background: var(--accent);
            color: white;
            border-color: var(--accent);
            box-shadow: 0 2px 12px var(--accent-glow);
        }

        .mode-toggle.active {
            background: linear-gradient(135deg, var(--accent) 0%, #8b5cf6 100%);
        }

        /* Filter group */
        .filter-group {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 10px 16px;
            background: var(--bg);
            border-bottom: 1px solid var(--border);
            min-height: 48px;
        }

        .conf-filter {
            background: rgba(255, 159, 64, 0.1);
            color: #ff9f40;
        }
        .conf-filter:hover { background: rgba(255, 159, 64, 0.2); }
        .conf-filter.active {
            background: #ff9f40;
            color: var(--bg);
            box-shadow: 0 2px 12px rgba(255, 159, 64, 0.2);
        }
        .year-filter {
            font-size: 12px;
            padding: 5px 12px;
            margin-left: 16px;
            background: var(--bg-elevated);
            border-left: 2px solid var(--accent);
        }
        .year-filter.active {
            background: var(--accent);
            color: white;
            border-left-color: white;
        }

        /* Paper list */
        .paper-list { padding: 12px 16px 80px; }

        .section-title {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
            margin: 20px 0 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .section-title::before {
            content: '';
            width: 3px;
            height: 14px;
            background: var(--accent);
            border-radius: 2px;
        }

        .paper-card {
            background: var(--bg-card);
            border-radius: var(--radius-lg);
            padding: 16px;
            margin-bottom: 10px;
            border: 1px solid var(--border);
            transition: all 0.2s ease;
            cursor: pointer;
        }
        .paper-card:hover {
            border-color: rgba(99, 102, 241, 0.3);
            box-shadow: var(--shadow-lg);
            transform: translateY(-1px);
        }

        .paper-card .title {
            font-size: 15px;
            font-weight: 600;
            color: var(--text);
            line-height: 1.5;
            margin-bottom: 6px;
        }
        .paper-card .authors {
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 10px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .paper-card .meta {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            margin-bottom: 10px;
        }
        .tag {
            padding: 3px 10px;
            border-radius: 100px;
            font-size: 11px;
            font-weight: 500;
        }
        .tag.conf { background: var(--tag-bg); color: var(--accent-soft); }
        .tag.topic { background: rgba(255,255,255,0.05); color: var(--text-secondary); }
        .tag.year { background: rgba(255, 159, 64, 0.1); color: #ff9f40; }
        .tag.has-summary { background: rgba(34, 197, 94, 0.1); color: var(--success); }

        .paper-card .links {
            display: flex;
            gap: 16px;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid var(--border);
        }
        .paper-card .links a {
            font-size: 13px;
            color: var(--accent-soft);
            text-decoration: none;
            font-weight: 500;
            transition: opacity 0.15s;
        }
        .paper-card .links a:hover { opacity: 0.7; }

        /* Abstract toggle */
        .abstract-toggle {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            font-size: 12px;
            color: var(--text-secondary);
            cursor: pointer;
            margin-top: 8px;
            transition: all 0.15s;
        }
        .abstract-toggle:hover {
            background: var(--border);
            color: var(--text);
        }

        .abstract-content {
            display: none;
            margin-top: 14px;
            padding-top: 14px;
            border-top: 1px solid var(--border);
            animation: fadeIn 0.2s ease;
        }
        .abstract-content.show { display: block; }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-4px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .abstract-content h4 {
            font-size: 11px;
            color: var(--text-muted);
            margin: 12px 0 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .abstract-content p {
            font-size: 13px;
            line-height: 1.7;
            color: var(--text-secondary);
            margin: 0;
        }
        .abstract-content .cn { color: var(--text-muted); }

        /* AI Summary badge on card */
        .summary-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            font-size: 11px;
            color: var(--success);
            margin-left: auto;
        }

        /* Detail page */
        .detail-page { display: none; }
        .detail-page.show { display: block; animation: fadeIn 0.2s ease; }

        .detail-header {
            background: var(--bg-card);
            padding: 20px;
            border-bottom: 1px solid var(--border);
        }
        .detail-header .back-btn {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 14px;
            color: var(--accent-soft);
            margin-bottom: 16px;
            cursor: pointer;
            font-weight: 500;
        }
        .detail-header h2 {
            font-size: 18px;
            font-weight: 700;
            margin: 0 0 10px;
            line-height: 1.4;
            color: var(--text);
        }
        .detail-header .authors {
            font-size: 13px;
            color: var(--text-secondary);
            margin-bottom: 12px;
        }
        .detail-header .meta {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 12px;
        }
        .detail-header .links {
            display: flex;
            gap: 16px;
            margin-top: 12px;
        }
        .detail-header .links a {
            font-size: 14px;
            color: var(--accent-soft);
            text-decoration: none;
            font-weight: 500;
        }

        .detail-body {
            padding: 0 16px 40px;
            max-width: 800px;
            margin: 0 auto;
        }

        .detail-section {
            background: var(--bg-card);
            border-radius: var(--radius-lg);
            padding: 20px;
            margin: 16px 0;
            border: 1px solid var(--border);
        }

        .detail-section h3 {
            font-size: 14px;
            font-weight: 600;
            color: var(--text);
            margin: 0 0 14px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .detail-section h3:first-child { margin-top: 0; }

        .detail-section p {
            font-size: 14px;
            line-height: 1.8;
            color: var(--text-secondary);
            margin: 0 0 12px;
        }
        .detail-section p:last-child { margin-bottom: 0; }

        .detail-section .lang-label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin: 16px 0 8px;
        }
        .detail-section .lang-label:first-of-type { margin-top: 0; }

        /* AI Summary box */
        .ai-summary-box {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.05) 100%);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: var(--radius-md);
            padding: 16px;
            margin: 12px 0;
        }
        .ai-summary-box .lang-label {
            color: var(--accent-soft);
            font-weight: 500;
        }
        .ai-summary-box p {
            color: var(--text);
        }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-muted);
        }
        .empty-state-icon {
            font-size: 48px;
            margin-bottom: 16px;
            opacity: 0.5;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 24px;
            color: var(--text-muted);
            font-size: 12px;
            border-top: 1px solid var(--border);
        }

        /* Responsive */
        @media (min-width: 768px) {
            .paper-list {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 12px;
            }
            .paper-card { margin-bottom: 0; }
            .section-title { grid-column: 1 / -1; }
        }
        @media (min-width: 1200px) {
            .paper-list { grid-template-columns: repeat(3, 1fr); }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-top">
            <div>
                <h1>claw notes</h1>
                <div class="subtitle">LLM Serving / Speculative Decoding / KV Cache / MoE</div>
            </div>
            <div class="header-icon">&#9889;</div>
        </div>
        <div class="stats">
            <div class="stat"><div class="num" id="paperCount">-</div><div class="label">Papers</div></div>
            <div class="stat"><div class="num" id="confCount">-</div><div class="label">Venues</div></div>
            <div class="stat"><div class="num" id="topicCount">-</div><div class="label">Topics</div></div>
            <div class="stat"><div class="num" id="summaryCount">-</div><div class="label">Summarized</div></div>
        </div>
    </div>

    <!-- Search -->
    <div class="search-box">
        <input type="text" id="searchInput" placeholder="Search papers by title, author, topic...">
    </div>

    <!-- List page -->
    <div id="listPage">
        <div class="nav-bar" id="navBar"></div>
        <div class="filter-group" id="filterGroup"></div>
        <div class="paper-list" id="paperList"></div>
    </div>

    <!-- Detail page -->
    <div class="detail-page" id="detailPage">
        <div class="detail-header">
            <div class="back-btn" onclick="showList()">&#8592; Back to list</div>
            <h2 id="detailTitle"></h2>
            <div class="authors" id="detailAuthors"></div>
            <div class="meta" id="detailMeta"></div>
            <div class="links" id="detailLinks"></div>
        </div>
        <div class="detail-body" id="detailBody"></div>
    </div>

    <div class="footer">
        <span id="totalCount">0</span> papers &middot; Auto-updated daily
    </div>

    <script>
    const papers = ''' + json.dumps(papers, ensure_ascii=False) + ''';
    const topics = ''' + json.dumps(topics) + ''';

    // Group by conf-year
    const confYearPapers = {};
    papers.forEach(p => {
        const conf = p.conference || 'arXiv';
        const year = p.year || 2024;
        const key = conf + '|' + year;
        if (!confYearPapers[key]) confYearPapers[key] = [];
        confYearPapers[key].push(p);
    });

    const confYears = Object.keys(confYearPapers).sort();

    let currentFilter = "";
    let currentSearch = "";
    let filterMode = "topic";
    let expandedConf = null;

    const confYearMap = {};
    confYears.forEach(cy => {
        const [conf, year] = cy.split('|');
        if (!confYearMap[conf]) confYearMap[conf] = [];
        confYearMap[conf].push(year);
    });
    const conferences = Object.keys(confYearMap).sort();

    function initNav() {
        const navBar = document.getElementById('navBar');
        const filterGroup = document.getElementById('filterGroup');

        // Mode toggle
        const modeChip = document.createElement('div');
        modeChip.className = 'nav-chip mode-toggle';
        modeChip.dataset.mode = 'topic';
        modeChip.textContent = 'Topics';
        modeChip.onclick = () => setFilterMode('topic');
        navBar.appendChild(modeChip);

        const confChip = document.createElement('div');
        confChip.className = 'nav-chip mode-toggle';
        confChip.dataset.mode = 'confYear';
        confChip.textContent = 'Venues';
        confChip.onclick = () => setFilterMode('confYear');
        navBar.appendChild(confChip);

        // "All" filter
        const allChip = document.createElement('div');
        allChip.className = 'filter-chip';
        allChip.dataset.filter = '';
        allChip.textContent = 'All';
        allChip.onclick = () => {
            currentFilter = "";
            document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
            render();
        };
        filterGroup.appendChild(allChip);

        // Topic filters
        topics.forEach(t => {
            const chip = document.createElement('div');
            chip.className = 'filter-chip topic-filter';
            chip.dataset.filter = t;
            chip.dataset.mode = 'topic';
            chip.textContent = t;
            chip.onclick = () => setFilter(t);
            chip.style.display = 'none';
            filterGroup.appendChild(chip);
        });

        // Conf filters
        conferences.forEach(conf => {
            const confChip = document.createElement('div');
            confChip.className = 'filter-chip conf-filter';
            confChip.dataset.conf = conf;
            confChip.dataset.mode = 'confYear';
            confChip.textContent = conf;
            confChip.style.display = 'none';
            confChip.onclick = () => toggleConf(conf);
            filterGroup.appendChild(confChip);

            const years = confYearMap[conf].sort((a,b) => b - a);
            years.forEach(year => {
                const yearChip = document.createElement('div');
                yearChip.className = 'filter-chip year-filter';
                yearChip.dataset.filter = conf + '|' + year;
                yearChip.dataset.mode = 'confYear';
                yearChip.dataset.conf = conf;
                yearChip.textContent = year;
                yearChip.style.display = 'none';
                yearChip.onclick = (e) => {
                    e.stopPropagation();
                    setFilter(conf + '|' + year);
                };
                filterGroup.appendChild(yearChip);
            });
        });

        setFilterMode('topic');
    }

    function toggleConf(conf) {
        expandedConf = (expandedConf === conf) ? null : conf;
        updateConfDisplay();
    }

    function updateConfDisplay() {
        document.querySelectorAll('.conf-filter').forEach(c => {
            c.classList.toggle('active', c.dataset.conf === expandedConf);
        });
        document.querySelectorAll('.year-filter').forEach(c => {
            c.style.display = (c.dataset.conf === expandedConf && filterMode === 'confYear') ? '' : 'none';
        });
    }

    function setFilterMode(mode) {
        filterMode = mode;
        expandedConf = null;

        document.querySelectorAll('.mode-toggle').forEach(c => {
            c.classList.toggle('active', c.dataset.mode === mode);
        });

        document.querySelectorAll('.topic-filter').forEach(c => {
            c.style.display = mode === 'topic' ? '' : 'none';
        });
        document.querySelectorAll('.conf-filter').forEach(c => {
            c.style.display = mode === 'confYear' ? '' : 'none';
        });
        document.querySelectorAll('.year-filter').forEach(c => {
            c.style.display = 'none';
        });

        currentFilter = "";
        document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
        render();
    }

    function setFilter(filter) {
        currentFilter = (currentFilter === filter) ? "" : filter;
        document.querySelectorAll('.filter-chip').forEach(c => {
            c.classList.toggle('active', c.dataset.filter === currentFilter);
        });
        render();
    }

    function render() {
        const search = document.getElementById('searchInput').value.toLowerCase();
        currentSearch = search;

        let filtered = papers.filter(p => {
            let matchFilter = true;
            if (currentFilter) {
                if (filterMode === 'topic') {
                    matchFilter = p.topic === currentFilter;
                } else if (filterMode === 'confYear') {
                    const conf = p.conference || 'arXiv';
                    const year = p.year || 2024;
                    matchFilter = (conf + '|' + year) === currentFilter;
                }
            }
            const matchSearch = !search ||
                (p.title && p.title.toLowerCase().includes(search)) ||
                (p.authors && p.authors.toLowerCase().includes(search)) ||
                (p.topic && p.topic.toLowerCase().includes(search)) ||
                (p.abstract_en && p.abstract_en.toLowerCase().includes(search)) ||
                (p.abstract_cn && p.abstract_cn.toLowerCase().includes(search));
            return matchFilter && matchSearch;
        });

        const list = document.getElementById('paperList');

        if (filtered.length === 0) {
            list.innerHTML = `<div class="empty-state">
                <div class="empty-state-icon">&#128269;</div>
                <div>No papers found matching your criteria.</div>
            </div>`;
            return;
        }

        let html = '';
        const grouped = {};
        filtered.forEach(p => {
            const conf = p.conference || 'arXiv';
            const year = p.year || 2024;
            const key = conf + ' / ' + year;
            if (!grouped[key]) grouped[key] = [];
            grouped[key].push(p);
        });

        Object.keys(grouped).sort().reverse().forEach(key => {
            html += `<div class="section-title">${key}</div>`;
            grouped[key].forEach(p => {
                html += renderPaperCard(p);
            });
        });

        list.innerHTML = html;

        // Update stats
        const summaryCount = papers.filter(p => p.ai_summary_en || p.ai_summary_cn || p.ai_summary).length;
        document.getElementById('paperCount').textContent = papers.length;
        document.getElementById('confCount').textContent = Object.keys(confYearMap).length;
        document.getElementById('topicCount').textContent = topics.length;
        document.getElementById('summaryCount').textContent = summaryCount;
        document.getElementById('totalCount').textContent = papers.length;
    }

    function renderPaperCard(p) {
        const hasSummary = p.ai_summary_en || p.ai_summary_cn || p.ai_summary;
        const hasAbstract = p.abstract_en || p.abstract_cn;
        return `
        <div class="paper-card" onclick="showDetail(${p.id})" data-id="${p.id}">
            <div class="title">${p.title || 'Untitled'}</div>
            <div class="authors">${p.authors || 'Unknown'}</div>
            <div class="meta">
                <span class="tag conf">${p.conference || 'arXiv'}</span>
                <span class="tag year">${p.year || '2024'}</span>
                <span class="tag topic">${p.topic || 'LLM Serving'}</span>
                ${hasSummary ? '<span class="tag has-summary">&#10003; AI Summary</span>' : ''}
            </div>
            <div class="links">
                ${p.arxiv_id ? `<a href="https://arxiv.org/abs/${p.arxiv_id}" target="_blank" onclick="event.stopPropagation()">arXiv</a>` : ''}
                ${p.github_repo ? `<a href="https://github.com/${p.github_repo}" target="_blank" onclick="event.stopPropagation()">GitHub</a>` : ''}
                <a href="javascript:void(0)" onclick="event.stopPropagation();showDetail(${p.id})">Read more &rarr;</a>
            </div>
            ${hasAbstract ? `
            <button class="abstract-toggle" onclick="event.stopPropagation();toggleAbstract(${p.id}, this)">
                <span>Abstract</span> <span>&#9662;</span>
            </button>
            <div class="abstract-content" id="abstract-${p.id}">
                ${p.abstract_en ? `<h4>English</h4><p>${escapeHtml(p.abstract_en.substring(0,500))}${p.abstract_en.length>500?'...':''}</p>` : ''}
                ${p.abstract_cn ? `<h4>中文</h4><p class="cn">${escapeHtml(p.abstract_cn.substring(0,350))}${p.abstract_cn.length>350?'...':''}</p>` : ''}
            </div>
            ` : ''}
        </div>`;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function toggleAbstract(id, btn) {
        const content = document.getElementById('abstract-' + id);
        const arrow = btn.querySelector('span:last-child');
        if (content.classList.contains('show')) {
            content.classList.remove('show');
            arrow.innerHTML = '&#9662;';
        } else {
            content.classList.add('show');
            arrow.innerHTML = '&#9652;';
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
        if (p.arxiv_id) links += `<a href="https://arxiv.org/abs/${p.arxiv_id}" target="_blank">&#128196; arXiv</a>`;
        if (p.github_repo) links += `<a href="https://github.com/${p.github_repo}" target="_blank">&#128187; GitHub</a>`;
        document.getElementById('detailLinks').innerHTML = links;

        let body = '';

        // AI Summary section (bilingual)
        const aiEn = p.ai_summary_en || p.ai_summary;
        const aiCn = p.ai_summary_cn;
        if (aiEn || aiCn) {
            body += `<div class="detail-section">`;
            body += `<h3>&#9889; AI Summary</h3>`;
            if (aiEn) {
                body += `<div class="lang-label">English</div>`;
                body += `<div class="ai-summary-box"><p>${escapeHtml(aiEn)}</p></div>`;
            }
            if (aiCn) {
                body += `<div class="lang-label">中文</div>`;
                body += `<div class="ai-summary-box"><p>${escapeHtml(aiCn)}</p></div>`;
            }
            body += `</div>`;
        }

        // Abstract section (bilingual)
        if (p.abstract_en || p.abstract_cn) {
            body += `<div class="detail-section">`;
            body += `<h3>&#128221; Abstract</h3>`;
            if (p.abstract_en) {
                body += `<div class="lang-label">English</div>`;
                body += `<p>${escapeHtml(p.abstract_en)}</p>`;
            }
            if (p.abstract_cn) {
                body += `<div class="lang-label">中文</div>`;
                body += `<p class="cn">${escapeHtml(p.abstract_cn)}</p>`;
            }
            body += `</div>`;
        }

        // Introduction if available
        if (p.intro_en || p.intro_cn) {
            body += `<div class="detail-section">`;
            body += `<h3>&#128214; Introduction</h3>`;
            if (p.intro_en) {
                body += `<div class="lang-label">English</div>`;
                body += `<p>${escapeHtml(p.intro_en)}</p>`;
            }
            if (p.intro_cn) {
                body += `<div class="lang-label">中文</div>`;
                body += `<p class="cn">${escapeHtml(p.intro_cn)}</p>`;
            }
            body += `</div>`;
        }

        if (!body) {
            body = '<div class="detail-section"><p style="color:var(--text-muted);text-align:center;padding:40px;">No detailed content available yet.</p></div>';
        }

        document.getElementById('detailBody').innerHTML = body;
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
