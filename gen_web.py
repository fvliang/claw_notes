#!/usr/bin/env python3
import json

db = json.load(open('database.json'))
papers = db['papers']
topics = db['topics']

# Extract unique years sorted descending
all_years = sorted(set(str(p.get('year', 2024)) for p in papers), reverse=True)

# Conference categories with influence ranking (higher = more influential, earlier in list)
CONF_CATEGORIES = {
    'ML': ['NeurIPS', 'ICML', 'ICLR', 'AAAI', 'AISTATS'],
    'NLP': ['ACL', 'EMNLP', 'NAACL', 'EACL', 'CoNLL', 'TACL', 'Findings'],
    'Systems': ['OSDI', 'SOSP', 'ASPLOS', 'EuroSys', 'ATC', 'PLDI', 'FAST'],
    'Arch': ['ISCA', 'MICRO', 'HPCA'],
    'Database': ['SIGMOD', 'VLDB', 'CIDR', 'ICDE'],
    'Networks': ['SIGCOMM', 'NSDI', 'CoNEXT'],
    'Vision': ['CVPR', 'ICCV', 'ECCV'],
    'Security': ['S&P', 'CCS', 'USENIX Security', 'NDSS'],
}

# Build categorized conf list
all_confs = sorted(set(p.get('conference', 'arXiv') for p in papers))
conf_categories = {}
for cat, confs in CONF_CATEGORIES.items():
    matched = [c for c in confs if c in all_confs]
    if matched:
        conf_categories[cat] = matched

# Unclassified -> Other
classified = set()
for confs in conf_categories.values():
    classified.update(confs)
other_confs = sorted([c for c in all_confs if c not in classified])
if other_confs:
    conf_categories['Other'] = other_confs

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
            --radius-lg: 14px;
            --shadow: 0 1px 3px rgba(0,0,0,0.3);
            --shadow-lg: 0 8px 30px rgba(0,0,0,0.4);
            --star-yellow: #fbbf24;
            --star-yellow-glow: rgba(251, 191, 36, 0.2);
        }

        * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", Roboto, "Noto Sans SC", sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
            max-width: 1400px;
            margin: 0 auto;
        }

        /* Header */
        .header {
            background: linear-gradient(180deg, var(--bg-card) 0%, var(--bg) 100%);
            padding: 20px 16px 12px;
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
            margin-bottom: 12px;
        }
        .header h1 {
            font-size: 18px;
            font-weight: 700;
            letter-spacing: -0.3px;
            background: linear-gradient(135deg, var(--text) 0%, var(--accent-soft) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .header .subtitle {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 1px;
        }

        /* Stats */
        .stats {
            display: flex;
            gap: 16px;
        }
        .stat {
            display: flex;
            flex-direction: column;
        }
        .stat .num {
            font-size: 17px;
            font-weight: 700;
            color: var(--text);
            line-height: 1.2;
        }
        .stat .label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Search */
        .search-box {
            display: flex;
            gap: 8px;
            padding: 10px 12px;
            background: var(--bg);
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 88px;
            z-index: 99;
        }
        .search-box input {
            flex: 1;
            padding: 8px 14px;
            border: 1px solid var(--border);
            border-radius: 100px;
            background: var(--bg-card);
            color: var(--text);
            font-size: 13px;
            outline: none;
            transition: all 0.2s;
        }
        .search-box input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }
        .search-box input::placeholder { color: var(--text-muted); }
        .search-btn {
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 100px;
            padding: 0 16px;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.2s;
            flex-shrink: 0;
        }
        .search-btn:hover {
            background: var(--accent-soft);
        }

        /* Nav */
        .nav-bar {
            display: flex;
            gap: 6px;
            padding: 10px 12px;
            background: var(--bg);
            border-bottom: 1px solid var(--border);
            overflow-x: auto;
            scrollbar-width: none;
        }
        .nav-bar::-webkit-scrollbar { display: none; }

        .nav-chip, .filter-chip {
            padding: 5px 12px;
            border-radius: 100px;
            font-size: 12px;
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
            box-shadow: 0 2px 10px var(--accent-glow);
        }

        .mode-toggle.active {
            background: linear-gradient(135deg, var(--accent) 0%, #8b5cf6 100%);
        }

        /* Filter rows */
        .filter-rows {
            background: var(--bg);
            border-bottom: 1px solid var(--border);
        }

        /* Timeline Sidebar */
        .timeline-sidebar {
            display: none;
            flex-direction: column;
            gap: 2px;
            width: 120px;
            flex-shrink: 0;
            padding: 8px 0;
            position: sticky;
            top: calc(88px + 42px + 1px);
            max-height: calc(100vh - 140px);
            overflow-y: auto;
            scrollbar-width: none;
        }
        .timeline-sidebar::-webkit-scrollbar { display: none; }
        .timeline-sidebar.show {
            display: flex;
        }
        .timeline-sidebar .toc-chip {
            padding: 5px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
            color: var(--text-secondary);
            text-align: left;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .timeline-sidebar .toc-chip:hover {
            background: var(--bg-elevated);
            color: var(--text);
        }
        .timeline-sidebar .toc-chip.active {
            background: var(--accent);
            color: white;
            box-shadow: 0 2px 10px var(--accent-glow);
            font-weight: 700;
        }

        /* Timeline mode layout */
        #listPage.timeline-mode {
            display: grid;
            grid-template-columns: 120px 1fr;
            grid-template-areas:
                "nav nav"
                "filters filters"
                "sidebar content";
            gap: 0;
            padding: 0 12px;
        }
        #listPage.timeline-mode .nav-bar { grid-area: nav; width: 100%; }
        #listPage.timeline-mode .filter-rows { grid-area: filters; width: 100%; }
        #listPage.timeline-mode .timeline-sidebar {
            grid-area: sidebar;
            display: flex;
            flex-direction: column;
            gap: 2px;
            padding: 8px 0;
            position: sticky;
            top: calc(88px + 42px + 1px);
            align-self: start;
            max-height: calc(100vh - 140px);
            overflow-y: auto;
            scrollbar-width: none;
        }
        #listPage.timeline-mode .paper-list {
            grid-area: content;
            padding: 10px 0 80px;
        }
        .filter-row {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            padding: 8px 12px;
            align-items: center;
        }
        .filter-row + .filter-row {
            border-top: 1px solid var(--border);
        }
        .filter-row-label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            flex-shrink: 0;
            padding: 4px 0;
            min-width: 40px;
        }

        .conf-filter {
            background: rgba(255, 159, 64, 0.1);
            color: #ff9f40;
        }
        .conf-filter:hover { background: rgba(255, 159, 64, 0.2); }
        .conf-filter.active {
            background: #ff9f40;
            color: var(--bg);
            box-shadow: 0 2px 10px rgba(255, 159, 64, 0.2);
        }
        .year-filter {
            background: rgba(99, 102, 241, 0.08);
            color: var(--accent-soft);
        }
        .year-filter.active {
            background: var(--accent);
            color: white;
        }

        /* Star button */
        .star-btn {
            position: absolute;
            top: 8px;
            right: 8px;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            border: none;
            background: rgba(15, 15, 18, 0.7);
            backdrop-filter: blur(4px);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            line-height: 1;
            color: var(--text-muted);
            transition: all 0.15s ease;
            z-index: 5;
            padding: 0;
            user-select: none;
        }
        .star-btn:hover {
            background: rgba(251, 191, 36, 0.15);
            transform: scale(1.1);
        }
        .star-btn .star-icon {
            transition: all 0.2s ease;
        }
        .star-btn.favorited {
            color: var(--star-yellow);
        }
        .star-btn.favorited .star-icon {
            text-shadow: 0 0 8px var(--star-yellow-glow);
        }
        .star-btn:active {
            transform: scale(0.9);
        }

        /* Paper card */
        .paper-card {
            position: relative;
            background: var(--bg-card);
            border-radius: var(--radius-md);
            padding: 12px;
            padding-right: 38px;
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
            font-size: 12px;
            font-weight: 600;
            color: var(--text);
            line-height: 1.45;
            margin-bottom: 4px;
            padding-right: 4px;
        }
        .paper-card .authors {
            font-size: 10px;
            color: var(--text-muted);
            margin-bottom: 6px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .paper-card .meta {
            display: flex;
            gap: 4px;
            flex-wrap: wrap;
            margin-bottom: 6px;
        }
        .tag {
            padding: 2px 7px;
            border-radius: 100px;
            font-size: 9px;
            font-weight: 500;
        }
        .tag.conf { background: var(--tag-bg); color: var(--accent-soft); }
        .tag.topic { background: rgba(255,255,255,0.05); color: var(--text-secondary); }
        .tag.year { background: rgba(255, 159, 64, 0.1); color: #ff9f40; }
        .tag.has-summary { background: rgba(34, 197, 94, 0.1); color: var(--success); }

        .paper-card .links {
            display: flex;
            gap: 10px;
            margin-top: 6px;
            padding-top: 6px;
            border-top: 1px solid var(--border);
        }
        .paper-card .links a {
            font-size: 10px;
            color: var(--accent-soft);
            text-decoration: none;
            font-weight: 500;
            transition: opacity 0.15s;
        }
        .paper-card .links a:hover { opacity: 0.7; }

        /* Tooltip preview */
        .paper-tooltip {
            position: fixed;
            z-index: 1000;
            max-width: 420px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 14px 16px;
            box-shadow: 0 12px 40px rgba(0,0,0,0.6);
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.15s ease;
            display: none;
        }
        .paper-tooltip.show {
            display: block;
            opacity: 1;
        }
        .paper-tooltip .tt-title {
            font-size: 12px;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 6px;
            line-height: 1.4;
        }
        .paper-tooltip .tt-label {
            font-size: 10px;
            color: var(--accent-soft);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
            font-weight: 600;
        }
        .paper-tooltip .tt-content {
            font-size: 11px;
            line-height: 1.6;
            color: var(--text-secondary);
        }
        .paper-tooltip .tt-empty {
            font-size: 11px;
            color: var(--text-muted);
            font-style: italic;
        }

        /* Paper list */
        .paper-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 8px;
            padding: 10px 12px 80px;
        }

        .section-title {
            grid-column: 1 / -1;
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            margin: 14px 0 6px;
            text-transform: uppercase;
            letter-spacing: 1px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .section-title::before {
            content: '';
            width: 3px;
            height: 12px;
            background: var(--accent);
            border-radius: 2px;
        }

        /* Detail page */
        .detail-page { display: none; }
        .detail-page.show { display: block; animation: fadeIn 0.2s ease; }

        .detail-header {
            background: var(--bg-card);
            padding: 16px;
            border-bottom: 1px solid var(--border);
        }
        .detail-header .back-btn {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
            color: var(--accent-soft);
            margin-bottom: 12px;
            cursor: pointer;
            font-weight: 500;
        }
        .detail-header h2 {
            font-size: 16px;
            font-weight: 700;
            margin: 0 0 8px;
            line-height: 1.4;
            color: var(--text);
        }
        .detail-header .authors {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 10px;
        }
        .detail-header .meta {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            margin-bottom: 10px;
        }
        .detail-header .links {
            display: flex;
            gap: 14px;
            margin-top: 10px;
        }
        .detail-header .links a {
            font-size: 13px;
            color: var(--accent-soft);
            text-decoration: none;
            font-weight: 500;
        }

        .detail-body {
            padding: 0 12px 40px;
            max-width: 800px;
            margin: 0 auto;
        }

        .detail-section {
            background: var(--bg-card);
            border-radius: var(--radius-md);
            padding: 16px;
            margin: 12px 0;
            border: 1px solid var(--border);
        }

        .detail-section h3 {
            font-size: 13px;
            font-weight: 600;
            color: var(--text);
            margin: 0 0 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .detail-section h3:first-child { margin-top: 0; }

        .detail-section p {
            font-size: 13px;
            line-height: 1.8;
            color: var(--text-secondary);
            margin: 0 0 10px;
        }
        .detail-section p:last-child { margin-bottom: 0; }

        .detail-section .lang-label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin: 12px 0 6px;
        }
        .detail-section .lang-label:first-of-type { margin-top: 0; }

        /* AI Summary box */
        .ai-summary-box {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.05) 100%);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: var(--radius-sm);
            padding: 12px;
            margin: 8px 0;
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
            grid-column: 1 / -1;
            text-align: center;
            padding: 50px 20px;
            color: var(--text-muted);
        }
        .empty-state-icon {
            font-size: 40px;
            margin-bottom: 12px;
            opacity: 0.5;
        }

        /* Favorites banner */
        .fav-banner {
            grid-column: 1 / -1;
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 12px 16px;
            background: linear-gradient(135deg, rgba(251, 191, 36, 0.08) 0%, rgba(251, 191, 36, 0.02) 100%);
            border: 1px solid rgba(251, 191, 36, 0.15);
            border-radius: var(--radius-md);
            margin-bottom: 4px;
        }
        .fav-banner .fav-icon {
            font-size: 18px;
            color: var(--star-yellow);
        }
        .fav-banner .fav-text {
            font-size: 13px;
            color: var(--text);
            font-weight: 600;
        }
        .fav-banner .fav-count {
            font-size: 11px;
            color: var(--text-muted);
            margin-left: auto;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 20px;
            color: var(--text-muted);
            font-size: 11px;
            border-top: 1px solid var(--border);
        }

        @media (min-width: 1100px) {
            .paper-list { gap: 10px; }
            .paper-card { padding: 14px; padding-right: 40px; }
            .paper-card .title { font-size: 13px; }
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
        </div>
        <div class="stats">
            <div class="stat"><div class="num" id="paperCount">-</div><div class="label">Papers</div></div>
            <div class="stat"><div class="num" id="confCount">-</div><div class="label">Venues</div></div>
            <div class="stat"><div class="num" id="topicCount">-</div><div class="label">Topics</div></div>
            <div class="stat"><div class="num" id="favCount">-</div><div class="label">Favorites</div></div>
        </div>
    </div>

    <div class="search-box">
        <input type="text" id="searchInput" placeholder="Search papers by title, author, topic...">
        <button id="searchBtn" class="search-btn">Search</button>
    </div>

    <!-- List page -->
    <div id="listPage">
        <div class="nav-bar" id="navBar"></div>
        <div class="filter-rows" id="filterRows"></div>
        <div class="timeline-sidebar" id="timelineSidebar"></div>
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

    <div class="paper-tooltip" id="paperTooltip">
        <div class="tt-title" id="ttTitle"></div>
        <div class="tt-label">AI Summary (EN)</div>
        <div class="tt-content" id="ttContentEn"></div>
        <div class="tt-label" style="margin-top:8px;">AI Summary (CN)</div>
        <div class="tt-content" id="ttContentCn"></div>
    </div>

    <div class="footer">
        <span id="totalCount">0</span> papers &middot; Auto-updated daily
    </div>

    <script>
    // Load papers data asynchronously
    let papers = [];
    let topics = [];
    let confCategories = {};
    let allYears = [];

    // Favorites stored in localStorage
    const FAV_KEY = 'claw_notes_favorites';
    function getFavorites() {
        try {
            const raw = localStorage.getItem(FAV_KEY);
            return raw ? JSON.parse(raw) : [];
        } catch (e) {
            return [];
        }
    }
    function saveFavorites(favs) {
        localStorage.setItem(FAV_KEY, JSON.stringify(favs));
    }
    function isFavorited(id) {
        return getFavorites().includes(id);
    }
    function toggleFavorite(id) {
        const favs = getFavorites();
        const idx = favs.indexOf(id);
        if (idx >= 0) {
            favs.splice(idx, 1);
        } else {
            favs.push(id);
        }
        saveFavorites(favs);
        updateStarUI(id);
        updateFavCount();
        // If in favorites mode, re-render
        if (filterMode === 'favorites') {
            render();
        }
    }
    function updateStarUI(id) {
        document.querySelectorAll(`.star-btn[data-id="${id}"]`).forEach(btn => {
            const favorited = isFavorited(id);
            btn.classList.toggle('favorited', favorited);
            btn.innerHTML = favorited
                ? '<span class="star-icon">&#9733;</span>'
                : '<span class="star-icon">&#9734;</span>';
        });
    }
    function updateFavCount() {
        document.getElementById('favCount').textContent = getFavorites().length;
    }

    async function loadData() {
        const list = document.getElementById('paperList');
        list.innerHTML = '<div class="empty-state"><div class="empty-state-icon">&#128228;</div><div>Loading papers...</div><div style="margin-top:8px;font-size:12px;color:#666">This may take 5-15 seconds on first visit</div></div>';

        const baseUrl = window.location.href.split('#')[0].split('?')[0];
        const dataUrl = new URL('papers.json', baseUrl + (baseUrl.endsWith('/') ? '' : '/')).href;

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000);
            const resp = await fetch(dataUrl, {signal: controller.signal});
            clearTimeout(timeoutId);

            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            const data = await resp.json();
            papers = data.papers || [];
            topics = data.topics || [];
            confCategories = data.confCategories || {};
            allYears = data.allYears || [];

            allConfs = [...new Set(papers.map(p => p.conference || 'arXiv'))].sort();

            initNav();
            updateFavCount();
            render();
        } catch (e) {
            let msg = e.message || 'Unknown error';
            if (e.name === 'AbortError') msg = 'Request timed out. Please refresh.';
            list.innerHTML = '<div class="empty-state"><div class="empty-state-icon">&#9888;</div><div>Failed to load papers: ' + msg + '</div><div style="margin-top:12px;font-size:13px;color:#888">Try: <a href="' + dataUrl + '" target="_blank" style="color:#818cf8">direct link</a></div></div>';
            console.error(e);
        }
    }

    let allConfs = [];

    let filterMode = "topic";
    let activeTopic = "";
    let activeConf = "";
    let activeYear = "";

    function initNav() {
        const navBar = document.getElementById('navBar');
        navBar.innerHTML = '';

        const modes = [
            {key: 'topic', label: 'Topics'},
            {key: 'confYear', label: 'Venues'},
            {key: 'timeline', label: 'Timeline'},
            {key: 'favorites', label: 'Favorites'},
        ];

        modes.forEach(m => {
            const chip = document.createElement('div');
            chip.className = 'nav-chip mode-toggle';
            chip.dataset.mode = m.key;
            chip.textContent = m.label;
            chip.onclick = () => setFilterMode(m.key);
            navBar.appendChild(chip);
        });

        buildFilterRows();
        setFilterMode('topic');
    }

    function buildFilterRows() {
        const container = document.getElementById('filterRows');
        container.innerHTML = '';

        // Topic filters row
        const topicRow = document.createElement('div');
        topicRow.className = 'filter-row';
        topicRow.id = 'topicRow';

        const allTopic = makeChip('All', '', 'topic', () => { activeTopic = ''; updateFilters(); });
        topicRow.appendChild(allTopic);

        topics.forEach(t => {
            topicRow.appendChild(makeChip(t, t, 'topic', () => { activeTopic = t; updateFilters(); }));
        });
        container.appendChild(topicRow);

        // Venue rows (for confYear mode) - grouped by category
        Object.keys(confCategories).forEach(cat => {
            const catRow = document.createElement('div');
            catRow.className = 'filter-row conf-cat-row';
            catRow.dataset.cat = cat;
            catRow.style.display = 'none';

            const label = document.createElement('span');
            label.className = 'filter-row-label';
            label.textContent = cat;
            catRow.appendChild(label);

            confCategories[cat].forEach(c => {
                const chip = makeChip(c, c, 'conf', () => { activeConf = c; updateFilters(); });
                chip.classList.add('conf-filter');
                catRow.appendChild(chip);
            });
            container.appendChild(catRow);
        });

        const yearRow = document.createElement('div');
        yearRow.className = 'filter-row';
        yearRow.id = 'yearRow';
        yearRow.style.display = 'none';

        const label2 = document.createElement('span');
        label2.className = 'filter-row-label';
        label2.textContent = 'Year';
        yearRow.appendChild(label2);

        const allYear = makeChip('All', '', 'year', () => { activeYear = ''; updateFilters(); });
        allYear.classList.add('year-filter');
        yearRow.appendChild(allYear);

        allYears.forEach(y => {
            const chip = makeChip(y, y, 'year', () => { activeYear = y; updateFilters(); });
            chip.classList.add('year-filter');
            yearRow.appendChild(chip);
        });
        container.appendChild(yearRow);
    }

    function makeChip(text, value, type, onClick) {
        const chip = document.createElement('div');
        chip.className = 'filter-chip';
        chip.dataset.value = value;
        chip.dataset.type = type;
        chip.textContent = text;
        chip.onclick = onClick;
        return chip;
    }

    function updateFilters() {
        document.querySelectorAll('.filter-chip').forEach(c => {
            const t = c.dataset.type;
            const v = c.dataset.value;
            let isActive = false;
            if (t === 'topic') isActive = v === activeTopic;
            if (t === 'conf') isActive = v === activeConf;
            if (t === 'year') isActive = v === activeYear;
            c.classList.toggle('active', isActive);
        });
        render();
    }

    function setFilterMode(mode) {
        filterMode = mode;
        document.querySelectorAll('.mode-toggle').forEach(c => {
            c.classList.toggle('active', c.dataset.mode === mode);
        });

        // Show/hide filter rows based on mode
        document.getElementById('topicRow').style.display = (mode === 'topic') ? 'flex' : 'none';
        document.querySelectorAll('.conf-cat-row').forEach(r => {
            r.style.display = (mode === 'confYear') ? 'flex' : 'none';
        });
        const yr = document.getElementById('yearRow');
        if (yr) yr.style.display = (mode === 'confYear') ? 'flex' : 'none';

        // Show/hide timeline sidebar
        const sidebar = document.getElementById('timelineSidebar');
        if (sidebar) sidebar.classList.toggle('show', mode === 'timeline');

        // Toggle timeline-mode class on listPage
        document.getElementById('listPage').classList.toggle('timeline-mode', mode === 'timeline');

        // Reset filters
        activeTopic = '';
        activeConf = '';
        activeYear = '';
        updateFilters();
    }

    function render() {
        const searchInput = document.getElementById('searchInput');
        const search = searchInput.value.toLowerCase().trim();

        const list = document.getElementById('paperList');

        if (papers.length === 0) {
            list.innerHTML = '<div class="empty-state"><div class="empty-state-icon">&#128228;</div><div>Loading papers...</div></div>';
            return;
        }

        let filtered = papers.filter(p => {
            // Favorites mode: only show favorited
            if (filterMode === 'favorites') {
                return isFavorited(p.id);
            }

            let match = true;
            if (filterMode === 'topic' && activeTopic) {
                match = p.topic === activeTopic;
            }
            if (filterMode === 'confYear') {
                if (activeConf && p.conference !== activeConf) match = false;
                if (activeYear && String(p.year || 2024) !== activeYear) match = false;
            }
            if (!match) return false;

            if (!search) return true;

            const haystack = (
                (p.title || '') + ' ' +
                (p.authors || '') + ' ' +
                (p.topic || '') + ' ' +
                (p.abstract_en || '') + ' ' +
                (p.abstract_cn || '')
            ).toLowerCase();
            return haystack.includes(search);
        });

        if (filtered.length === 0) {
            if (filterMode === 'favorites') {
                list.innerHTML = `<div class="empty-state">
                    <div class="empty-state-icon">&#9734;</div>
                    <div>No favorites yet.</div>
                    <div style="margin-top:8px;font-size:12px;color:#666">Click the star on any paper to add it here.</div>
                </div>`;
            } else {
                list.innerHTML = `<div class="empty-state">
                    <div class="empty-state-icon">&#128269;</div>
                    <div>No papers found matching "${escapeHtml(searchInput.value)}"</div>
                    ${search ? '<div style="margin-top:8px;font-size:12px;color:#666">Try a different keyword or clear filters</div>' : ''}
                </div>`;
            }
            return;
        }

        let html = '';

        // Favorites banner
        if (filterMode === 'favorites') {
            html += `<div class="fav-banner">
                <span class="fav-icon">&#9733;</span>
                <span class="fav-text">Your Favorites</span>
                <span class="fav-count">${filtered.length} paper${filtered.length !== 1 ? 's' : ''}</span>
            </div>`;
        }

        // Grouping logic
        const grouped = {};
        if (filterMode === 'timeline') {
            // Group by YYYY-MM-DD from added_date, or fallback to year
            filtered.forEach(p => {
                let key;
                const date = p.added_date;
                if (date && date.length >= 10) {
                    key = date; // YYYY-MM-DD
                } else if (p.year) {
                    key = String(p.year); // fallback to year only
                } else {
                    key = 'unknown';
                }
                if (!grouped[key]) grouped[key] = [];
                grouped[key].push(p);
            });
        } else {
            filtered.forEach(p => {
                const conf = p.conference || 'arXiv';
                const year = p.year || 2024;
                const key = conf + ' / ' + year;
                if (!grouped[key]) grouped[key] = [];
                grouped[key].push(p);
            });
        }

        Object.keys(grouped).sort().reverse().forEach(key => {
            const slug = 'grp-' + key.replace(/[^a-zA-Z0-9]/g, '-');
            html += `<div class="section-title" id="${slug}">${key}</div>`;
            grouped[key].forEach(p => {
                html += renderPaperCard(p);
            });
        });

        list.innerHTML = html;

        // Build Timeline Sidebar if in timeline mode
        if (filterMode === 'timeline') {
            const sb = document.getElementById('timelineSidebar');
            const keys = Object.keys(grouped).sort().reverse();
            sb.innerHTML = keys.map(k => {
                const slug = 'grp-' + k.replace(/[^a-zA-Z0-9]/g, '-');
                return `<div class="toc-chip" data-slug="${slug}" onclick="scrollToGroup('${slug}')">${k}</div>`;
            }).join('');
            setupTimelineObserver();
        } else {
            document.getElementById('timelineSidebar').innerHTML = '';
            if (window._timelineObserver) {
                window._timelineObserver.disconnect();
                window._timelineObserver = null;
            }
        }

        // Update all star buttons
        document.querySelectorAll('.star-btn').forEach(btn => {
            const id = parseInt(btn.dataset.id);
            updateStarUI(id);
        });

        const summaryCount = papers.filter(p => p.ai_summary_en || p.ai_summary_cn || p.ai_summary).length;
        document.getElementById('paperCount').textContent = papers.length;
        document.getElementById('confCount').textContent = allConfs.length;
        document.getElementById('topicCount').textContent = topics.length;
        document.getElementById('totalCount').textContent = papers.length;
    }

    function renderPaperCard(p) {
        const hasSummary = p.ai_summary_en || p.ai_summary_cn || p.ai_summary;
        const fav = isFavorited(p.id);
        return `
        <div class="paper-card" onclick="showDetail(${p.id})" data-id="${p.id}">
            <button class="star-btn${fav ? ' favorited' : ''}" data-id="${p.id}"
                onclick="event.stopPropagation(); toggleFavorite(${p.id});"
                title="${fav ? 'Remove from favorites' : 'Add to favorites'}">
                <span class="star-icon">${fav ? '&#9733;' : '&#9734;'}</span>
            </button>
            <div class="title">${escapeHtml(p.title) || 'Untitled'}</div>
            <div class="authors">${escapeHtml(p.authors) || 'Unknown'}</div>
            <div class="meta">
                <span class="tag conf">${escapeHtml(p.conference || 'arXiv')}</span>
                <span class="tag year">${p.year || '2024'}</span>
                ${hasSummary ? '<span class="tag has-summary">&#10003; AI</span>' : ''}
            </div>
            <div class="links">
                ${p.arxiv_id ? `<a href="https://arxiv.org/abs/${p.arxiv_id}" target="_blank" onclick="event.stopPropagation()">arXiv</a>` : (p.conference === 'arXiv' ? `<a href="https://www.google.com/search?q=${encodeURIComponent(p.title)}+arxiv" target="_blank" onclick="event.stopPropagation()">Find arXiv</a>` : '')}
                ${p.github_repo ? `<a href="https://github.com/${p.github_repo}" target="_blank" onclick="event.stopPropagation()">GitHub</a>` : ''}
            </div>
        </div>`;
    }

    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Tooltip for AI Summary preview
    const tooltip = document.getElementById('paperTooltip');
    const ttTitle = document.getElementById('ttTitle');
    const ttContentEn = document.getElementById('ttContentEn');
    const ttContentCn = document.getElementById('ttContentCn');

    function showTooltip(p, x, y) {
        ttTitle.textContent = p.title || 'Untitled';

        const aiEn = p.ai_summary_en || p.ai_summary;
        if (aiEn) {
            ttContentEn.textContent = aiEn.substring(0, 240) + (aiEn.length > 240 ? '...' : '');
            ttContentEn.className = 'tt-content';
        } else {
            ttContentEn.textContent = 'No AI summary yet';
            ttContentEn.className = 'tt-empty';
        }

        const aiCn = p.ai_summary_cn;
        if (aiCn) {
            ttContentCn.textContent = aiCn.substring(0, 240) + (aiCn.length > 240 ? '...' : '');
            ttContentCn.className = 'tt-content';
            ttContentCn.style.display = 'block';
        } else {
            ttContentCn.textContent = '';
            ttContentCn.style.display = 'none';
        }

        tooltip.classList.add('show');
        positionTooltip(x, y);
    }

    function positionTooltip(x, y) {
        const rect = tooltip.getBoundingClientRect();
        let left = x + 16;
        let top = y + 16;
        if (left + rect.width > window.innerWidth - 12) {
            left = x - rect.width - 16;
        }
        if (top + rect.height > window.innerHeight - 12) {
            top = y - rect.height - 16;
        }
        tooltip.style.left = left + 'px';
        tooltip.style.top = top + 'px';
    }

    function hideTooltip() {
        tooltip.classList.remove('show');
    }

    // Delegate hover events for paper cards
    document.addEventListener('mouseover', function(e) {
        const card = e.target.closest('.paper-card');
        if (card) {
            const id = parseInt(card.dataset.id);
            const p = papers.find(pp => pp.id === id);
            if (p) showTooltip(p, e.clientX, e.clientY);
        }
    });
    document.addEventListener('mousemove', function(e) {
        if (tooltip.classList.contains('show')) {
            positionTooltip(e.clientX, e.clientY);
        }
    });
    document.addEventListener('mouseout', function(e) {
        const card = e.target.closest('.paper-card');
        if (card) hideTooltip();
    });

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

    function scrollToGroup(slug) {
        const el = document.getElementById(slug);
        if (el) {
            const headerOffset = 140; // approximate sticky header height
            const top = el.getBoundingClientRect().top + window.pageYOffset - headerOffset;
            window.scrollTo({ top: top, behavior: 'smooth' });
        }
    }

    function setupTimelineObserver() {
        if (window._timelineObserver) {
            window._timelineObserver.disconnect();
        }
        const sections = document.querySelectorAll('.section-title[id^="grp-"]');
        if (sections.length === 0) return;

        window._timelineObserver = new IntersectionObserver((entries) => {
            let best = null;
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    // Pick the one closest to top of viewport
                    if (!best || entry.boundingClientRect.top < best.boundingClientRect.top) {
                        best = entry;
                    }
                }
            });
            if (best) {
                const slug = best.target.id;
                document.querySelectorAll('.timeline-sidebar .toc-chip').forEach(chip => {
                    chip.classList.toggle('active', chip.dataset.slug === slug);
                });
            }
        }, {
            rootMargin: '-120px 0px -60% 0px',
            threshold: 0
        });

        sections.forEach(sec => window._timelineObserver.observe(sec));
    }

    document.getElementById('searchInput').addEventListener('input', function(e) {
        clearTimeout(window.searchDebounce);
        window.searchDebounce = setTimeout(render, 200);
    });
    document.getElementById('searchInput').addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            clearTimeout(window.searchDebounce);
            render();
        }
    });
    document.getElementById('searchBtn').addEventListener('click', function() {
        clearTimeout(window.searchDebounce);
        render();
    });

    loadData();
    </script>
</body>
</html>'''

with open('docs/papers.json', 'w', encoding='utf-8') as f:
    json.dump({
        'papers': papers,
        'topics': topics,
        'confCategories': conf_categories,
        'allYears': all_years,
    }, f, ensure_ascii=False)

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Done! Generated {len(papers)} papers")
