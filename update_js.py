import re

with open('docs/index.html', 'r') as f:
    content = f.read()

# Replace openDet function
old_opendet = '''function openDet(id){
const p=P.find(x=>x.id===id);if(!p)return;
currentDetId=id;
document.getElementById('plist').style.display='none';
document.querySelector('.filterbar').style.display='none';
document.querySelector('.search').style.display='none';
document.getElementById('det').classList.add('on');
document.getElementById('detTitle').textContent=p.title||'';
document.getElementById('detFav').textContent=isF(id)?'⭐':'☆';
document.getElementById('detAuth').textContent=p.authors||'';
document.getElementById('detTags').innerHTML='<span class="ptag ptag-c">'+(p.conference||'arXiv')+'</span><span class="ptag ptag-t">'+(p.topic||'')+'</span>'+((isF(id))?'<span class="ptag ptag-f">已收藏</span>':'');
let links='';
if(p.arxiv_id)links+='<a href="https://arxiv.org/abs/'+p.arxiv_id+'" target="_blank">📄 arXiv</a>';
if(p.github_repo)links+='<a href="https://github.com/'+p.github_repo+'" target="_blank">🐙 GitHub</a>';
document.getElementById('detLinks').innerHTML=links;
let body='';
if(p.abstract_en)body+='<h3>Abstract</h3><p>'+p.abstract_en+'</p>';
if(p.abstract_cn)body+='<h3>摘要</h3><p>'+p.abstract_cn+'</p>';
if(p.intro_en)body+='<h3>1. Introduction</h3><p>'+p.intro_en+'</p>';
if(!body)body='<div class="empty">暂无详细内容</div>';
document.getElementById('detBody').innerHTML=body;
if(!isR(id))togR(id);
window.scrollTo(0,0);
}'''

new_opendet = '''function openDet(id){
const p=P.find(x=>x.id===id);if(!p)return;
currentDetId=id;
document.getElementById('plist').style.display='none';
document.querySelector('.filterbar').style.display='none';
document.querySelector('.search').style.display='none';
document.getElementById('det').classList.add('on');
document.getElementById('detTitle').textContent=p.title||'';
document.getElementById('detFav').textContent=isF(id)?'⭐':'☆';
document.getElementById('detAuth').textContent=p.authors||'';
document.getElementById('detTags').innerHTML='<span class="ptag ptag-c">'+(p.conference||'arXiv')+'</span><span class="ptag ptag-t">'+(p.topic||'')+'</span>'+((isF(id))?'<span class="ptag ptag-f">已收藏</span>':'');
let links='';
if(p.arxiv_id)links+='<a href="https://arxiv.org/abs/'+p.arxiv_id+'" target="_blank">📄 arXiv</a>';
if(p.github_repo)links+='<a href="https://github.com/'+p.github_repo+'" target="_blank">🐙 GitHub</a>';
document.getElementById('detLinks').innerHTML=links;

// Show AI summary if available
const summaryEl = document.getElementById('detSummary');
if(p.ai_summary){
  summaryEl.innerHTML = '<div class="ai-summary-box"><div class="ai-summary-label">AI 总结</div><div class="ai-summary-text">'+p.ai_summary+'</div></div>';
  summaryEl.style.display = 'block';
} else {
  summaryEl.style.display = 'none';
}

// Show tabs and content
const hasAbs = p.abstract_en || p.abstract;
const hasCn = p.abstract_cn && p.abstract_cn.indexOf('[中文翻译待补充]') === -1 && p.abstract_cn.indexOf('[自动翻译生成中...]') === -1;

if(hasAbs || hasCn){
  document.getElementById('detTabs').style.display = 'flex';
  
  // Abstract tab
  const absEl = document.getElementById('tabAbs');
  let absText = p.abstract_en || p.abstract || '';
  if(absText){
    absEl.innerHTML = '<h4>Abstract</h4><p>'+absText+'</p>';
    absEl.style.display = 'block';
  } else {
    absEl.style.display = 'none';
  }
  
  // Chinese tab
  const cnEl = document.getElementById('tabCn');
  if(hasCn){
    cnEl.innerHTML = '<h4>中文摘要</h4><p>'+p.abstract_cn+'</p>';
    cnEl.style.display = 'none';
  } else {
    cnEl.innerHTML = '<div class="empty">暂无中文翻译</div>';
    cnEl.style.display = 'none';
  }
  
  // Reset tab states
  document.querySelectorAll('.section-tab').forEach(t => t.classList.remove('on'));
  document.querySelectorAll('.section-content').forEach(t => t.classList.remove('on'));
  document.querySelector('.section-tab').classList.add('on');
  document.getElementById('tabAbs').classList.add('on');
} else {
  document.getElementById('detTabs').style.display = 'none';
  document.getElementById('tabAbs').style.display = 'none';
  document.getElementById('tabCn').style.display = 'none';
}

if(!isR(id))togR(id);
window.scrollTo(0,0);
}

function switchTab(tab){
  document.querySelectorAll('.section-tab').forEach(t => t.classList.remove('on'));
  document.querySelectorAll('.section-content').forEach(t => {t.classList.remove('on'); t.style.display='none';});
  
  if(tab === 'abs'){
    document.querySelectorAll('.section-tab')[0].classList.add('on');
    document.getElementById('tabAbs').classList.add('on');
    document.getElementById('tabAbs').style.display = 'block';
  } else {
    document.querySelectorAll('.section-tab')[1].classList.add('on');
    document.getElementById('tabCn').classList.add('on');
    document.getElementById('tabCn').style.display = 'block';
  }
}'''

content = content.replace(old_opendet, new_opendet)

# Also update back() to hide new elements
old_back = '''function back(){
document.getElementById('det').classList.remove('on');
document.getElementById('plist').style.display='block';
document.querySelector('.filterbar').style.display='flex';
document.querySelector('.search').style.display='block';
currentDetId=null;
}'''

new_back = '''function back(){
document.getElementById('det').classList.remove('on');
document.getElementById('plist').style.display='block';
document.querySelector('.filterbar').style.display='flex';
document.querySelector('.search').style.display='block';
document.getElementById('detSummary').style.display='none';
document.getElementById('detTabs').style.display='none';
document.getElementById('tabAbs').style.display='none';
document.getElementById('tabCn').style.display='none';
currentDetId=null;
}'''

content = content.replace(old_back, new_back)

with open('docs/index.html', 'w') as f:
    f.write(content)

print('Updated JavaScript in docs/index.html')
