const state = {
  view: 'researcher',
  datasetId: null,
  ingest: null,
  train: null,
  drift: null,
  lastUserMessage: '',
  ragCache: null,
  contentSafetyBlocked: false,
};

function $(id){ return document.getElementById(id); }

function setTab(tabId){
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  $(tabId).classList.add('active');
  document.querySelector(`.tab[data-tab="${tabId}"]`).classList.add('active');
}

function openPanel(){ $('side-panel').classList.add('open'); }
function closePanel(){ $('side-panel').classList.remove('open'); }

function getInitialTheme(){
  const saved = localStorage.getItem('theme');
  if (saved === 'dark' || saved === 'light') return saved;
  return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function applyTheme(theme){
  document.documentElement.dataset.theme = theme;
  localStorage.setItem('theme', theme);
  const btn = $('theme-toggle');
  if (btn) btn.textContent = (theme === 'dark') ? 'Light' : 'Dark';
}

function toggleTheme(){
  const cur = document.documentElement.dataset.theme || 'light';
  applyTheme(cur === 'dark' ? 'light' : 'dark');
}

function setSafetyPill(blocked){
  const pill = $('safety-pill');
  if (blocked){
    pill.textContent = 'Safety BLOCK';
    pill.classList.add('off');
  } else {
    pill.textContent = 'Safety ON';
    pill.classList.remove('off');
  }
}

function setView(view){
  state.view = view;
  $('vt-researcher').classList.toggle('active', view === 'researcher');
  $('vt-technical').classList.toggle('active', view === 'technical');

  // Sync all dual-cards tab visuals
  document.querySelectorAll('.dual-card').forEach(card => {
    const tabs = card.querySelectorAll('.dc-tab');
    tabs.forEach(t => t.classList.remove('a'));
    const toActivate = card.querySelector(`.dc-tab[data-view="${view}"]`);
    if (toActivate) toActivate.classList.add('a');
    else tabs[0]?.classList.add('a');
  });

  rerender();
}

function pretty(obj){ return JSON.stringify(obj, null, 2); }

async function api(path, { method='GET', json=null, form=null } = {}){
  const opts = { method, headers: {} };
  if (json){
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(json);
  }
  if (form){
    opts.body = form;
  }

  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok){
    const detail = body?.detail;
    const message = (typeof detail === 'string') ? detail : (detail?.error || detail?.answer || detail?.rationale || `HTTP ${res.status}`);
    const err = new Error(message);
    err.detail = detail;
    err.status = res.status;
    throw err;
  }
  return body;
}

function escapeHtml(s){
  return String(s || '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
}

function updateStepFlow(){
  const ids = ['step-1','step-2','step-3','step-4','step-5','step-6'];
  const els = ids.map($);

  els.forEach((el, idx) => {
    el.classList.remove('sc-done','sc-active','sc-todo');
    el.classList.add('sc-todo');
    el.textContent = String(idx+1);
  });

  if (state.ingest){
    els[0].classList.remove('sc-todo'); els[0].classList.add('sc-done'); els[0].textContent = '✓';
    els[1].classList.remove('sc-todo'); els[1].classList.add('sc-active');
  } else {
    els[0].classList.remove('sc-todo'); els[0].classList.add('sc-active');
  }

  if (state.train){
    els[1].classList.remove('sc-active'); els[1].classList.add('sc-done'); els[1].textContent = '✓';
    els[3].classList.remove('sc-todo'); els[3].classList.add('sc-done'); els[3].textContent = '✓';
    els[4].classList.remove('sc-todo'); els[4].classList.add('sc-active');
  }

  if (state.drift){
    els[5].classList.remove('sc-todo'); els[5].classList.add('sc-active');
  }
}

function updateBadges(){
  const badgeData = $('badge-data');
  const warnings = state.ingest?.quality?.technical?.quarantine_rows || 0;
  if (warnings > 0){
    badgeData.textContent = `${warnings} warnings`;
    badgeData.classList.remove('hidden');
  } else {
    badgeData.classList.add('hidden');
  }

  const badgeDrift = $('badge-drift');
  if (state.drift?.technical?.drift_detected){
    badgeDrift.classList.remove('hidden');
  } else {
    badgeDrift.classList.add('hidden');
  }
}

function renderSidePanelFromRag(technical){
  const sp = $('sp-body');
  sp.innerHTML = '';

  const papers = technical?.papers || [];
  const datasets = technical?.datasets || [];

  if (papers.length === 0 && datasets.length === 0){
    const card = document.createElement('div');
    card.className = 'sp-card';
    card.innerHTML = `
      <div class="sp-card-title">No results</div>
      <div class="sp-card-sub">No papers found for this keyword. Try another query.</div>
      <div class="sp-source">OpenAlex (BioPapers engine)</div>
    `;
    sp.appendChild(card);
    return;
  }

  [...papers, ...datasets].slice(0, 8).forEach((x) => {
    const card = document.createElement('div');
    card.className = 'sp-card';
    const url = x.url ? `<a href="${escapeHtml(x.url)}" target="_blank" rel="noopener">${escapeHtml(x.url)}</a>` : '';
    card.innerHTML = `
      <div class="sp-card-title">${escapeHtml(x.title || x.name || 'Related resource')}</div>
      <div class="sp-card-sub">Source: ${escapeHtml(x.source || '')}</div>
      <div class="sp-source">${url}</div>
    `;
    sp.appendChild(card);
  });
}

function researcherQualityNarrative(ingest){
  const q = ingest?.quality?.technical;
  if (!q) return '<strong>Upload a CSV to start.</strong>';
  const quarant = q.quarantine_rows || 0;
  const total = q.total_rows || 0;
  const curated = q.curated_rows || Math.max(0, total - quarant);
  const reasons = q.quarantine_reasons || [];
  const example = reasons[0];
  const exampleTxt = example ? `Example: <strong>${escapeHtml(example.column)}</strong> out-of-range (rule: ${escapeHtml(example.rule)}).` : '';

  return `
    <strong>Checking your data quality.</strong>
    We reviewed ${total} rows: ${curated} are ready, and ${quarant} were moved to quarantine for review (nothing was silently dropped).
    ${exampleTxt ? `<div class="annotation"><span class="ann-icon">Tip</span>${exampleTxt} Want to inspect the quarantined rows and reasons?</div>` : ''}
  `;
}

function technicalBlock(title, obj){
  return `<strong>${escapeHtml(title)}</strong><br><pre style="white-space:pre-wrap">${escapeHtml(pretty(obj || {}))}</pre>`;
}

function schemaNarrative(schemaInfo){
  if (!schemaInfo?.columns) return '—';
  const cols = Object.entries(schemaInfo.columns).slice(0, 10);
  const lines = cols.map(([k,v]) => `• ${k}: ${v.type} (null_rate=${(v.null_rate ?? 0).toFixed(3)})`).join('<br>');
  return `<strong>Detected columns:</strong><br>${lines}${Object.keys(schemaInfo.columns).length > 10 ? '<br>…' : ''}`;
}

function renderMetrics(){
  if (!state.ingest){
    $('m-total').textContent = '—';
    $('m-quarantine').textContent = '—';
    $('m-valid').textContent = '—';
    return;
  }
  const q = state.ingest.quality.technical;
  const total = q.total_rows || 0;
  const quarant = q.quarantine_rows || 0;
  const curated = q.curated_rows || Math.max(0, total - quarant);
  const validPct = total ? ((curated/total)*100).toFixed(1) + '%' : '—';
  $('m-total').textContent = total.toLocaleString('en-US');
  $('m-quarantine').textContent = quarant.toLocaleString('en-US');
  $('m-valid').textContent = validPct;
}

function resultsMetrics(){
  if (!state.train?.metrics){
    $('r-m1').textContent = '—'; $('r-l1').textContent = 'metric 1';
    $('r-m2').textContent = '—'; $('r-l2').textContent = 'metric 2';
    $('r-m3').textContent = '—'; $('r-l3').textContent = 'metric 3';
    return;
  }

  const m = state.train.metrics;
  if (m.task === 'classification'){
    $('r-m1').textContent = (m.auc ?? 0).toFixed(3);
    $('r-l1').textContent = 'AUC-ROC';
    $('r-m2').textContent = (m.recall ?? 0).toFixed(3);
    $('r-l2').textContent = 'Recall';
    $('r-m3').textContent = `${Math.round(m.latency_p95_ms ?? 0)}ms`;
    $('r-l3').textContent = 'p95 latency';
  } else {
    $('r-m1').textContent = (m.r2 ?? 0).toFixed(3);
    $('r-l1').textContent = 'R²';
    $('r-m2').textContent = `${Math.round(m.latency_p95_ms ?? 0)}ms`;
    $('r-l2').textContent = 'p95 latency';
    $('r-m3').textContent = (m.equalized_odds_difference ?? 0).toFixed(3);
    $('r-l3').textContent = 'fairness (proxy)';
  }
}

function researcherResultsNarrative(train){
  const pass = !!train?.production_gate?.technical?.pass;
  const m = train?.metrics || {};
  const core = pass
    ? 'Your model passed the production gate for a canary rollout.'
    : 'Your model did not pass the production gate. We will not promote it automatically.';

  let metricsLine = '';
  if (m.task === 'classification'){
    metricsLine = `AUC=${(m.auc ?? 0).toFixed(3)}, Recall=${(m.recall ?? 0).toFixed(3)}, p95 latency=${Math.round(m.latency_p95_ms ?? 0)}ms.`;
  } else {
    metricsLine = `R²=${(m.r2 ?? 0).toFixed(3)}, p95 latency=${Math.round(m.latency_p95_ms ?? 0)}ms.`;
  }

  return `<strong>${core}</strong><br>${escapeHtml(metricsLine)}<div class="mini" style="margin-top:8px">Dataset hash: ${escapeHtml(train.dataset_hash || '')}</div>`;
}

function rerender(){
  updateStepFlow();
  updateBadges();
  renderMetrics();
  resultsMetrics();

  if (state.view === 'researcher'){
    $('exp-dual').innerHTML = state.ingest ? researcherQualityNarrative(state.ingest) : $('exp-dual').innerHTML;
    $('data-dual').innerHTML = state.ingest
      ? `<strong>Your data looks usable.</strong> You can inspect quarantined rows before proceeding.\n<br><br><span class="mini">Quarantine is a safety buffer: rows are separated with reasons, not discarded.</span>`
      : '<strong>No dataset loaded yet.</strong> Go to Experiment and upload a CSV.';

    $('schema-dual').innerHTML = state.ingest ? schemaNarrative(state.ingest.schema_info) : '—';

    $('results-dual').innerHTML = state.train
      ? researcherResultsNarrative(state.train)
      : $('results-dual').innerHTML;

    $('monitor-dual').innerHTML = state.drift
      ? `<strong>${state.drift.technical?.drift_detected ? 'Drift detected in your incoming data.' : 'No meaningful drift detected.'}</strong>
         <div class="mini" style="margin-top:6px">Action: ${escapeHtml(state.drift.technical?.monitoring_action || '')}</div>`
      : $('monitor-dual').innerHTML;
  } else {
    $('exp-dual').innerHTML = state.ingest ? technicalBlock('Quality (technical)', state.ingest.quality?.technical) : '<strong>Quality (technical)</strong><br>—';
    $('data-dual').innerHTML = state.ingest
      ? technicalBlock('Ingest (technical)', {dataset_id: state.ingest.dataset_id, schema_info: state.ingest.schema_info})
      : '<strong>Ingest (technical)</strong><br>—';

    $('schema-dual').innerHTML = state.ingest
      ? technicalBlock('Schema (technical)', state.ingest.schema_info)
      : '<strong>Schema (technical)</strong><br>—';

    $('results-dual').innerHTML = state.train
      ? technicalBlock('Train response (technical)', state.train)
      : '<strong>Train response (technical)</strong><br>—';

    $('monitor-dual').innerHTML = state.drift
      ? technicalBlock('Drift report (technical)', state.drift.technical)
      : '<strong>Drift report (technical)</strong><br>—';
  }

  if (state.drift?.technical){
    const detected = !!state.drift.technical.drift_detected;
    $('mon-drift-dot').classList.remove('sd-ok','sd-warn','sd-run');
    $('mon-drift-dot').classList.add(detected ? 'sd-warn' : 'sd-ok');
    $('mon-drift-val').textContent = detected ? 'drift detected' : 'no drift';

    const action = state.drift.technical.monitoring_action;
    $('mon-retrain-dot').classList.remove('sd-ok','sd-warn','sd-run');
    $('mon-retrain-dot').classList.add(action === 'trigger_retraining' ? 'sd-run' : 'sd-ok');
    $('mon-retrain-val').textContent = action === 'trigger_retraining' ? 'recommended' : 'none';
  }
}

function setExpStatus(kind, label, val){
  const row = $('exp-status');
  row.classList.remove('hidden');
  const dot = $('exp-status-dot');
  dot.classList.remove('sd-ok','sd-warn','sd-run');
  dot.classList.add(kind);
  $('exp-status-label').textContent = label;
  $('exp-status-val').textContent = val || '';
}

async function ingestFile(file){
  setExpStatus('sd-run','Ingesting','processing…');
  try {
    const form = new FormData();
    form.append('file', file);
    const body = await api('/api/part1/ingest', { method:'POST', form });
    state.ingest = body;
    state.datasetId = body.dataset_id;

    const cols = Object.keys(body.schema_info?.columns || {});
    const defaultTarget = cols.find(c => !c.toLowerCase().includes('id')) || cols[0] || '';
    $('target-col').value = defaultTarget;

    setExpStatus('sd-ok','Ingest complete', `${file.name} · dataset_id=${body.dataset_id}`);
    rerender();

    if ((body.quality?.technical?.quarantine_rows || 0) > 0){
      setTab('t-data');
    }
  } catch (e){
    setExpStatus('sd-warn','Error', e.message);
  }
}

function appendBubble(kind, html){
  const area = $('chat-area');
  const div = document.createElement('div');
  div.className = 'bubble ' + (kind === 'user' ? 'bubble-user' : 'bubble-ai');
  div.innerHTML = html;
  area.appendChild(div);
  area.scrollTop = area.scrollHeight;
}

async function sendAgentMessage(){
  const inp = $('chat-inp');
  const msg = inp.value.trim();
  if (!msg) return;
  inp.value = '';
  state.lastUserMessage = msg;

  appendBubble('user', escapeHtml(msg));

  try {
    setSafetyPill(false);
    const body = await api('/api/part3/agent/message', {
      method:'POST',
      json: {
        dataset_id: state.datasetId,
        message: msg,
      }
    });

    const rationale = `<div class="rationale-box">${escapeHtml(body.rationale)}</div>`;
    const btn = `<button class="lit-btn" onclick="window.__openPanelAndSearch()">View related research</button>`;
    appendBubble('ai', `${escapeHtml(body.answer)}${rationale}${btn}`);

    if (body.side_panel?.technical?.sources?.length){
      const sources = body.side_panel.technical.sources;
      const technical = {
        papers: sources.filter(x => !((x.source || '').toLowerCase().includes('openml') || (x.source || '').toLowerCase().includes('kaggle'))),
        datasets: sources.filter(x => (x.source || '').toLowerCase().includes('openml') || (x.source || '').toLowerCase().includes('kaggle')),
      };
      state.ragCache = technical;
      renderSidePanelFromRag(technical);
    }
  } catch (e){
    if (e.status === 400){
      setSafetyPill(true);
      state.contentSafetyBlocked = true;
      appendBubble('ai', `I can't help with that request due to safety policy.<div class="rationale-box">${escapeHtml(e.detail?.rationale || 'Blocked by Content Safety (RAI).')}</div>`);
      return;
    }
    appendBubble('ai', `Error: ${escapeHtml(e.message)}`);
  }
}

async function searchRelated(){
  const q = state.lastUserMessage || $('chat-inp').value.trim();
  if (!q) {
    openPanel();
    return;
  }
  const body = await api('/api/part3/rag/search', { method:'POST', json: { query: q, top_k: 5 } });
  state.ragCache = body.technical;
  renderSidePanelFromRag(body.technical);
}

async function trainNow(){
  const target = $('target-col').value.trim();
  if (!state.datasetId){
    appendBubble('ai', 'Please upload a CSV in the Experiment tab to obtain a dataset_id.');
    return;
  }
  if (!target){
    appendBubble('ai', 'Please specify target_column.');
    return;
  }

  appendBubble('ai', `Training with dataset_id=${escapeHtml(state.datasetId)} and target=${escapeHtml(target)}…`);
  try {
    const body = await api('/api/part2/train', {
      method:'POST',
      json: { dataset_id: state.datasetId, target_column: target }
    });
    state.train = body;
    rerender();
    setTab('t-results');

    const pass = body.production_gate?.technical?.pass;
    appendBubble('ai', `${escapeHtml(body.production_gate?.investigator?.summary || (pass ? 'Gate PASS.' : 'Gate FAIL.'))}
      <div class="rationale-box">Gate: ${pass ? 'PASS' : 'FAIL'} · dataset_hash=${escapeHtml(body.dataset_hash)}</div>`);
  } catch (e){
    appendBubble('ai', `Training error: ${escapeHtml(e.message)}`);
  }
}

async function driftNow(){
  const ref = $('drift-ref').value.trim();
  const cur = $('drift-cur').value.trim();
  const cols = $('drift-cols').value.split(',').map(s => s.trim()).filter(Boolean);

  if (!ref || !cur || cols.length === 0){
    alert('Provide reference/current IDs and at least one numeric column.');
    return;
  }

  try {
    const body = await api('/api/part2/drift', { method:'POST', json: { reference_dataset_id: ref, current_dataset_id: cur, numeric_columns: cols } });
    state.drift = body;
    rerender();
    setTab('t-monitor');
  } catch (e){
    alert('Drift error: ' + e.message);
  }
}

function setupUploadZone(){
  const zone = $('upload-zone');
  zone.addEventListener('click', () => $('csvFile').click());

  $('csvFile').addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (file) ingestFile(file);
  });

  ;['dragenter','dragover'].forEach(evt => {
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.style.borderColor = 'var(--color-border-info)';
      zone.style.color = 'var(--color-text-info)';
    });
  });

  ;['dragleave','drop'].forEach(evt => {
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.style.borderColor = '';
      zone.style.color = '';
    });
  });

  zone.addEventListener('drop', (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file) ingestFile(file);
  });
}

function setupEvents(){
  document.querySelectorAll('.tab').forEach((t) => {
    t.addEventListener('click', () => setTab(t.dataset.tab));
  });

  $('vt-researcher').addEventListener('click', () => setView('researcher'));
  $('vt-technical').addEventListener('click', () => setView('technical'));

  document.querySelectorAll('.dc-tab[data-view]').forEach((t) => {
    t.addEventListener('click', () => setView(t.dataset.view));
  });

  $('btn-open-panel').addEventListener('click', async () => {
    openPanel();
    await searchRelated();
  });
  $('sp-close').addEventListener('click', closePanel);

  const themeBtn = $('theme-toggle');
  themeBtn.addEventListener('click', toggleTheme);
  themeBtn.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') toggleTheme();
  });

  $('btn-send').addEventListener('click', sendAgentMessage);
  $('chat-inp').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendAgentMessage();
  });

  $('btn-train').addEventListener('click', trainNow);
  $('btn-drift').addEventListener('click', driftNow);

  window.__openPanelAndSearch = async () => {
    openPanel();
    await searchRelated();
  };
}

async function init(){
  applyTheme(getInitialTheme());

  setupEvents();
  setupUploadZone();
  setView('researcher');
  setSafetyPill(false);

  try { await api('/api/health'); } catch (_) {}

  rerender();
}

window.addEventListener('DOMContentLoaded', init);
