const state = {
  view: 'researcher',
  sessionId: null,
  agentContext: '',
  datasetId: null,
  ingest: null,
  eda: null,
  curatedSample: null,
  quarantineSample: null,
  dataFeedback: null,
  search: null, // { query, technical, investigator, filter }
  train: null,
  drift: null,
  lastUserMessage: '',
  ragCache: null,
  sidePanelFilter: 'papers', // papers | datasets
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
  const v = (s === null || s === undefined) ? '' : String(s);
  return v.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
}

function updateStepFlow(){
  const ids = ['step-1','step-2','step-3','step-4','step-5','step-6'];
  const els = ids.map($);

  els.forEach((el, idx) => {
    el.classList.remove('sc-done','sc-active','sc-todo');
    el.classList.add('sc-todo');
    el.textContent = String(idx+1);
  });

  // Default progression: upload → quality → features → train → evaluate → deploy
  if (!state.ingest){
    els[0].classList.remove('sc-todo'); els[0].classList.add('sc-active');
    return;
  }

  // Upload done
  els[0].classList.remove('sc-todo'); els[0].classList.add('sc-done'); els[0].textContent = '✓';
  // Quality done (ingest includes quality/quarantine)
  els[1].classList.remove('sc-todo'); els[1].classList.add('sc-done'); els[1].textContent = '✓';

  // Azure ML path (preferred when running cloud jobs)
  const aml = state.aml;
  if (aml?.deploy?.status === 'deployed' || aml?.deploy?.status === 'mock'){
    els[2].classList.remove('sc-todo'); els[2].classList.add('sc-done'); els[2].textContent = '✓';
    els[3].classList.remove('sc-todo'); els[3].classList.add('sc-done'); els[3].textContent = '✓';
    els[4].classList.remove('sc-todo'); els[4].classList.add('sc-done'); els[4].textContent = '✓';
    els[5].classList.remove('sc-todo'); els[5].classList.add('sc-done'); els[5].textContent = '✓';
    return;
  }

  if (aml?.results){
    // Features + Train + Evaluate done; next is Deploy
    els[2].classList.remove('sc-todo'); els[2].classList.add('sc-done'); els[2].textContent = '✓';
    els[3].classList.remove('sc-todo'); els[3].classList.add('sc-done'); els[3].textContent = '✓';
    els[4].classList.remove('sc-todo'); els[4].classList.add('sc-done'); els[4].textContent = '✓';
    els[5].classList.remove('sc-todo'); els[5].classList.add('sc-active');
    return;
  }

  if (aml?.job_id){
    els[2].classList.remove('sc-todo'); els[2].classList.add('sc-done'); els[2].textContent = '✓';
    els[3].classList.remove('sc-todo'); els[3].classList.add('sc-active');
    return;
  }

  // If no Azure ML job yet, we're at Features
  els[2].classList.remove('sc-todo'); els[2].classList.add('sc-active');

  // Back-compat: local training sets state.train
  if (state.train){
    els[3].classList.remove('sc-active'); els[3].classList.add('sc-done'); els[3].textContent = '✓';
    els[4].classList.remove('sc-todo'); els[4].classList.add('sc-active');
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

function _setSidePanelTabsActive(){
  const map = {
    papers: $('sp-tab-papers'),
    datasets: $('sp-tab-datasets'),
  };
  Object.entries(map).forEach(([k, el]) => {
    if (!el) return;
    const active = (k === state.sidePanelFilter);
    el.classList.toggle('active', active);
    el.setAttribute('aria-selected', active ? 'true' : 'false');
  });
}

function _renderRagCardsInto(el, technical, { filter='papers', showCountsOnTabs=false, tabs=null } = {}){
  if (!el) return;
  el.innerHTML = '';

  const papers = technical?.papers || [];
  const datasets = technical?.datasets || [];

  const mkBadge = (txt, kind='') => {
    const cls = kind ? `sp-badge ${kind}` : 'sp-badge';
    return `<span class="${cls}">${escapeHtml(txt)}</span>`;
  };

  const renderItem = (x, { showSourceBadge=true, idxLabel='' } = {}) => {
    const title = x.title || x.name || 'Related resource';
    const url = x.url || '';

    const badges = [];
    if (idxLabel) badges.push(mkBadge(idxLabel, 'idx'));
    if (showSourceBadge && x.source) badges.push(mkBadge(x.source, 'src'));
    if (x.year) badges.push(mkBadge(x.year));
    if (x.open_access && String(x.open_access).toLowerCase() === 'y') badges.push(mkBadge('OA', 'oa'));
    if (x.citations) badges.push(mkBadge(`${x.citations} cites`, 'cite'));

    const venue = x.venue ? `<div class="sp-card-sub">${escapeHtml(x.venue)}</div>` : '';
    const snippet = x.snippet ? `<div class="sp-snippet">${escapeHtml(x.snippet)}</div>` : '';

    const titleHtml = url
      ? `<a class="sp-link" href="${escapeHtml(url)}" target="_blank" rel="noopener">${escapeHtml(title)}</a>`
      : `<div class="sp-card-title">${escapeHtml(title)}</div>`;

    return `
      <div class="sp-card">
        <div class="sp-card-title">${titleHtml}</div>
        <div class="sp-badges">${badges.join(' ')}</div>
        ${venue}
        ${snippet}
        ${url ? `<div class="sp-source"><a href="${escapeHtml(url)}" target="_blank" rel="noopener">${escapeHtml(url)}</a></div>` : ''}
      </div>
    `;
  };

  const renderPapers = () => {
    if (!papers.length){
      el.innerHTML = `<div class="sp-card"><div class="sp-card-title">No papers</div><div class="sp-card-sub">Try a more specific query.</div></div>`;
      return;
    }
    el.innerHTML = papers.slice(0, 10).map((x,i) => renderItem(x, { showSourceBadge: false, idxLabel: `P${i+1}` })).join('');
  };

  const renderDatasets = () => {
    if (!datasets.length){
      el.innerHTML = `<div class="sp-card"><div class="sp-card-title">No datasets</div><div class="sp-card-sub">Try another keyword (OpenML uses dataset tags/names).</div></div>`;
      return;
    }
    el.innerHTML = datasets.slice(0, 10).map((x,i) => renderItem(x, { showSourceBadge: true, idxLabel: `D${i+1}` })).join('');
  };

  if (papers.length === 0 && datasets.length === 0){
    el.innerHTML = `
      <div class="sp-card">
        <div class="sp-card-title">No results</div>
        <div class="sp-card-sub">No related papers/datasets found for this query. Try another one.</div>
        <div class="sp-source">Sources: Europe PMC · OpenML</div>
      </div>
    `;
    return;
  }

  if (filter === 'datasets') renderDatasets();
  else renderPapers();
}

function renderSearch(){
  const resEl = $('search-results');
  const insightEl = $('search-insight');
  if (!resEl || !insightEl) return;

  const tech = state.search?.technical;
  const inv = state.search?.investigator;
  const filter = state.search?.filter || 'papers';

  const papers = tech?.papers || [];
  const datasets = tech?.datasets || [];

  if ($('search-tab-papers')){
    $('search-tab-papers').textContent = `Papers (${papers.length})`;
    $('search-tab-papers').classList.toggle('active', filter === 'papers');
    $('search-tab-papers').setAttribute('aria-selected', filter === 'papers' ? 'true' : 'false');
  }
  if ($('search-tab-datasets')){
    $('search-tab-datasets').textContent = `Datasets (${datasets.length})`;
    $('search-tab-datasets').classList.toggle('active', filter === 'datasets');
    $('search-tab-datasets').setAttribute('aria-selected', filter === 'datasets' ? 'true' : 'false');
  }

  if (!tech){
    resEl.innerHTML = '<div class="mini">Run a search to see results.</div>';
    insightEl.textContent = 'Run a search to get an AI comment on how these results can add context and how to enrich your dataset with relevant metadata.';
    return;
  }

  _renderRagCardsInto(resEl, tech, { filter });

  const insight = inv?.insight || '';
  const rationale = inv?.rationale || '';
  if (insight){
    insightEl.innerHTML = `${escapeHtml(insight)}${rationale ? `<div class="rationale-box">${escapeHtml(rationale)}</div>` : ''}`;
  } else {
    insightEl.textContent = 'No AI insight returned.';
  }
}

function renderSidePanelFromRag(technical){
  const sp = $('sp-body');
  sp.innerHTML = '';

  const papers = technical?.papers || [];
  const datasets = technical?.datasets || [];

  // Update tab labels with counts
  if ($('sp-tab-papers')) $('sp-tab-papers').textContent = `Papers (${papers.length})`;
  if ($('sp-tab-datasets')) $('sp-tab-datasets').textContent = `Datasets (${datasets.length})`;

  // Default filter: show papers if available, else datasets
  if (state.sidePanelFilter === 'papers' && papers.length === 0 && datasets.length > 0) state.sidePanelFilter = 'datasets';
  if (state.sidePanelFilter === 'datasets' && datasets.length === 0 && papers.length > 0) state.sidePanelFilter = 'papers';

  _setSidePanelTabsActive();

  _renderRagCardsInto(sp, technical, { filter: state.sidePanelFilter });
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

function quarantineNarrative(ingest){
  const q = ingest?.quality?.technical;
  if (!q) return '<strong>Quarantine is empty or not available.</strong>';
  const n = q.quarantine_rows || 0;
  const reasons = q.quarantine_reasons || [];
  if (n === 0) return '<strong>No rows in quarantine.</strong> Great — no anomalies detected by the current rules.';

  const rows = reasons.map(r => `<tr><td>${escapeHtml(r.column || '—')}</td><td>${escapeHtml(r.rule || '—')}</td><td>${escapeHtml(String(r.affected_rows ?? ''))}</td><td>${escapeHtml(String(r.lower ?? ''))}</td><td>${escapeHtml(String(r.upper ?? ''))}</td></tr>`).join('');
  return `
    <strong>${n} rows were moved to quarantine.</strong>
    <div class="mini" style="margin-top:6px">Rows are separated for review (not deleted). The table shows which rules triggered and how many rows they affected.</div>
    <div style="margin-top:10px">
      <table class="table">
        <thead><tr><th>Column</th><th>Rule</th><th>Affected</th><th>Lower</th><th>Upper</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="5">—</td></tr>'}</tbody>
      </table>
    </div>
  `;
}

function edaDefaultCard(){
  return `
    <strong>Run EDA to see distributions and correlations.</strong>
    <div class="chat-input-row" style="margin-top:10px">
      <button class="send-btn" id="btn-eda">Run EDA</button>
    </div>
    <div class="mini" style="margin-top:6px">EDA runs on the curated dataset and includes missingness, distributions, correlations, and (optionally) outcome balance.</div>
  `;
}

function edaSummaryNarrative(eda){
  const o = eda?.technical?.overview;
  if (!o) return edaDefaultCard();
  const miss = eda?.technical?.missingness || {};
  const topMiss = Object.entries(miss).slice(0,5).map(([k,v]) => `• ${k}: ${(v*100).toFixed(1)}% missing`).join('<br>');
  const pairs = eda?.technical?.correlation?.top_pairs || [];
  const topPairs = pairs.slice(0,5).map(p => `• ${p.a} ↔ ${p.b}: corr=${(p.corr ?? 0).toFixed(3)}`).join('<br>');
  const ta = eda?.technical?.target_analysis;
  const taLine = ta?.task === 'classification'
    ? `Outcome balance: imbalance ratio ≈ ${(ta.imbalance_ratio ?? 1).toFixed(2)} · ${escapeHtml(ta.recommendation || '')}`
    : (ta?.task === 'regression' ? `Outcome: regression-like · ${escapeHtml(ta.recommendation || '')}` : 'Outcome balance: choose a column below (optional).');

  return `
    <strong>EDA overview</strong><br>
    Rows (curated): ${o.rows} · Columns: ${o.columns} · Duplicate rows: ${o.duplicate_rows}<br>
    <div class="mini" style="margin-top:8px"><strong>Missingness (top)</strong><br>${topMiss || '—'}</div>
    <div class="mini" style="margin-top:8px"><strong>Correlations (top)</strong><br>${topPairs || '—'}</div>
    <div class="mini" style="margin-top:8px"><strong>Modeling note</strong><br>${taLine}</div>
    <div class="chat-input-row" style="margin-top:10px">
      <button class="lit-btn" id="btn-eda">Refresh EDA</button>
    </div>
  `;
}

function _table(columns, rows, { maxCols=10 } = {}){
  const cols = (columns || []).slice(0, maxCols);
  const head = cols.map(c => `<th>${escapeHtml(c)}</th>`).join('');
  const body = (rows || []).slice(0, 12).map(r => {
    const tds = cols.map(c => `<td>${escapeHtml(r?.[c])}</td>`).join('');
    return `<tr>${tds}</tr>`;
  }).join('');
  return `<table class="table"><thead><tr>${head}</tr></thead><tbody>${body || `<tr><td colspan="${cols.length || 1}">—</td></tr>`}</tbody></table>`;
}

function previewNarrative(){
  if (!state.datasetId) return '<strong>Preview will appear after upload.</strong>';
  const cur = state.curatedSample?.technical;
  const qua = state.quarantineSample?.technical;

  const curTable = cur?.columns?.length ? _table(cur.columns, cur.rows) : '—';
  const quaTable = qua?.columns?.length ? _table(qua.columns, qua.rows) : '<div class="mini">No quarantined rows to preview.</div>';

  return `
    <strong>Curated rows (ready for analysis)</strong>
    <div style="margin-top:10px">${curTable}</div>
    <div style="margin-top:14px"><strong>Quarantine sample (needs review)</strong></div>
    <div style="margin-top:10px">${quaTable}</div>
    <div class="mini" style="margin-top:10px">Tip: quarantine rows are separated with reasons — you can decide whether to fix, exclude, or adjust rules.</div>
  `;
}

function variablesNarrative(){
  if (!state.ingest) return '<strong>Variables will appear after upload.</strong>';

  const schemaCols = state.ingest.schema_info?.columns || {};
  const cols = Object.keys(schemaCols);

  const features = state.eda?.technical?.features;
  const ids = features?.id_like_columns || [];
  const feats = features?.feature_columns || [];
  const cands = features?.target_candidates || [];

  const idLine = ids.length ? ids.slice(0, 12).map(c => `<span class="chip">${escapeHtml(c)}</span>`).join(' ') : '<span class="mini">None detected.</span>';
  const featLine = feats.length ? feats.slice(0, 18).map(c => `<span class="chip">${escapeHtml(c)}</span>`).join(' ') : '<span class="mini">Run EDA to suggest features (or use Schema above).</span>';
  const candLine = cands.length ? cands.map(x => `<button class="lit-btn" data-set-target="${escapeHtml(x.column)}" style="padding:6px 10px">${escapeHtml(x.column)} (${x.unique})</button>`).join(' ') : '<span class="mini">No obvious candidates found (low-cardinality columns).</span>';

  return `
    <strong>Columns detected:</strong> ${cols.length}<br>
    <div class="mini" style="margin-top:6px">We try to separate ID-like columns (identifiers) from modeling features. You can override any suggestion.</div>

    <div style="margin-top:10px"><strong>Potential ID columns (excluded)</strong></div>
    <div style="margin-top:8px">${idLine}</div>

    <div style="margin-top:12px"><strong>Suggested feature columns</strong></div>
    <div style="margin-top:8px">${featLine}</div>

    <div style="margin-top:12px"><strong>Suggested outcome columns (click to set)</strong></div>
    <div class="chat-input-row" style="margin-top:8px;gap:8px;flex-wrap:wrap">${candLine}</div>
  `;
}

function rerender(){
  updateStepFlow();
  updateBadges();

  if ($('aml-compute')) $('aml-compute').textContent = (state.amlCompute || 'cpu-cluster');
  renderMetrics();
  resultsMetrics();

  // My data extra cards
  if ($('preview-dual')){
    $('preview-dual').innerHTML = (state.view === 'technical')
      ? technicalBlock('Preview (technical)', { curated: state.curatedSample?.technical, quarantine: state.quarantineSample?.technical })
      : previewNarrative();
  }
  if ($('vars-dual')) $('vars-dual').innerHTML = (state.view === 'technical')
    ? technicalBlock('Variables (technical)', state.eda?.technical?.features || state.ingest?.schema_info)
    : variablesNarrative();

  if ($('ai-feedback-dual')){
    $('ai-feedback-dual').innerHTML = (state.view === 'technical')
      ? technicalBlock('AI feedback (technical)', state.dataFeedback?.technical || { note: 'No feedback yet.' })
      : aiFeedbackNarrative();
  }

  if (state.view === 'researcher'){
    $('exp-dual').innerHTML = state.ingest ? researcherQualityNarrative(state.ingest) : $('exp-dual').innerHTML;
    $('data-dual').innerHTML = state.ingest
      ? `<strong>Your data looks usable.</strong> You can inspect quarantined rows before proceeding.\n<br><br><span class="mini">Quarantine is a safety buffer: rows are separated with reasons, not discarded.</span>`
      : '<strong>No dataset loaded yet.</strong> Upload a dataset file here or in Experiment.';

    $('schema-dual').innerHTML = state.ingest ? schemaNarrative(state.ingest.schema_info) : '—';
    $('quarantine-dual').innerHTML = state.ingest ? quarantineNarrative(state.ingest) : '<strong>Quarantine is empty or not available.</strong>';
    $('eda-dual').innerHTML = state.eda ? edaSummaryNarrative(state.eda) : edaDefaultCard();

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

    $('quarantine-dual').innerHTML = state.ingest
      ? technicalBlock('Quarantine (technical)', state.ingest.quality?.technical)
      : '<strong>Quarantine (technical)</strong><br>—';

    $('eda-dual').innerHTML = state.eda
      ? technicalBlock('EDA report (technical)', state.eda.technical)
      : '<strong>EDA report (technical)</strong><br>—';

    $('results-dual').innerHTML = state.train
      ? technicalBlock('Train response (technical)', state.train)
      : '<strong>Train response (technical)</strong><br>—';

    $('monitor-dual').innerHTML = state.drift
      ? technicalBlock('Drift report (technical)', state.drift.technical)
      : '<strong>Drift report (technical)</strong><br>—';
  }

  // keep visuals in sync after view/theme changes
  if (state.eda){
    try {
      plotMissingness();
      plotBalance();
      plotDistribution();
    } catch (_) {}
  }

  try { renderSearch(); } catch (_) {}

  // Azure ML experiment controls
  if (state.aml?.results){
    try { _renderAmlMetrics(state.aml.results); } catch (_) {}
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

async function runEDA({ silent=false } = {}){
  if (!state.datasetId){
    if (!silent) alert('Upload a dataset first.');
    return;
  }

  const chosenTarget = $('target-select')?.value?.trim();
  const targetParam = chosenTarget ? `&target_column=${encodeURIComponent(chosenTarget)}` : '';

  const binsRaw = $('dist-bins')?.value;
  const bins = Number.parseInt(String(binsRaw || '12'), 10);
  const binsParam = Number.isFinite(bins) ? `&bins=${encodeURIComponent(String(bins))}` : '';

  try {
    const body = await api(`/api/part1/eda?dataset_id=${encodeURIComponent(state.datasetId)}${targetParam}${binsParam}`);
    state.eda = body;
    rerender();

    // distributions: populate numeric selector
    const numericCols = Object.keys(body.technical?.numeric || {});
    const sel = $('dist-select');
    if (sel){
      const cur = sel.value;
      sel.innerHTML = '<option value="">Select numeric column</option>';
      numericCols.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c;
        opt.textContent = c;
        sel.appendChild(opt);
      });
      if (cur && numericCols.includes(cur)) sel.value = cur;
      else if (numericCols.length) sel.value = numericCols[0];
    }

    plotMissingness();
    plotBalance();
    plotDistribution();
  } catch (e){
    if (!silent) appendBubble('ai', `EDA error: ${escapeHtml(e.message)}`);
  }
}

function _resizeCanvas(canvas){
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(10, Math.floor(rect.width));
  const height = Math.max(10, Math.floor(rect.height));
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

function _niceTicks(maxVal, steps=4){
  if (!(maxVal > 0)) return [0, 1];
  const rawStep = maxVal / steps;
  const pow = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const n = rawStep / pow;
  const nice = (n <= 1) ? 1 : (n <= 2) ? 2 : (n <= 5) ? 5 : 10;
  const step = nice * pow;
  const top = Math.ceil(maxVal / step) * step;
  const ticks = [];
  for (let v=0; v<=top + 1e-9; v+=step) ticks.push(v);
  return ticks;
}

function _formatTick(v){
  if (Math.abs(v) >= 1000) return String(Math.round(v));
  if (Math.abs(v) >= 10) return v.toFixed(1).replace(/\.0$/,'');
  return v.toFixed(2).replace(/0$/,'').replace(/\.$/,'');
}

function plotDistribution(){
  const col = $('dist-select')?.value?.trim();
  if (!col) {
    if ($('hist-caption')) $('hist-caption').textContent = 'Select a numeric column to plot.';
    if ($('box-caption')) $('box-caption').textContent = '';
    return;
  }
  plotColumn(col);
  plotBoxplot(col);
}

function plotColumn(col){
  const tech = state.eda?.technical;
  const entry = tech?.numeric?.[col];
  const canvas = $('hist-canvas');
  if (!canvas) return;
  const { ctx, width: w, height: h } = _resizeCanvas(canvas);

  const css = getComputedStyle(document.documentElement);
  const bg = css.getPropertyValue('--color-background-secondary').trim();
  const fg = css.getPropertyValue('--color-text-tertiary').trim();
  const axis = css.getPropertyValue('--color-border-secondary').trim();
  const bar = (css.getPropertyValue('--dot-info').trim() || '#2B6CB0');

  const normalize = !!$('dist-normalize')?.checked;
  const logY = !!$('dist-logy')?.checked;

  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = bg;
  ctx.fillRect(0,0,w,h);

  if (!entry?.hist?.counts?.length || !entry?.hist?.bins?.length){
    if ($('hist-caption')) $('hist-caption').textContent = `Run EDA, then choose a numeric column to plot.`;
    return;
  }

  const countsRaw = entry.hist.counts;
  const edges = entry.hist.bins;
  const total = countsRaw.reduce((a,b)=>a+b, 0) || 1;
  const values = normalize ? countsRaw.map(c => (c/total)*100.0) : countsRaw.slice();

  const maxV = Math.max(...values, 1e-9);
  const t = (v) => logY ? Math.log10(1 + v) : v;
  const maxT = t(maxV);

  const padL = 52;
  const padR = 14;
  const padT = 14;
  const padB = 34;
  const innerW = w - padL - padR;
  const innerH = h - padT - padB;

  // axes
  ctx.strokeStyle = axis;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + innerH);
  ctx.lineTo(padL + innerW, padT + innerH);
  ctx.stroke();

  // y ticks + grid
  ctx.fillStyle = fg;
  ctx.font = '11px ui-sans-serif, system-ui';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';

  if (!logY){
    const ticks = _niceTicks(maxV, 4);
    ticks.forEach(v => {
      const y = padT + innerH - (t(v)/maxT)*innerH;
      ctx.strokeStyle = 'rgba(127,127,127,0.22)';
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + innerW, y);
      ctx.stroke();

      ctx.fillStyle = fg;
      ctx.fillText(_formatTick(v), padL - 6, y);
    });
  } else {
    // log ticks at 0, 1, 10, 100, ... up to max
    const maxPow = Math.ceil(Math.log10(maxV + 1));
    const tickVals = [0, ...Array.from({length: Math.max(1, maxPow)}, (_,i) => Math.pow(10, i))].filter(v => v <= maxV);
    tickVals.forEach(v => {
      const y = padT + innerH - (t(v)/maxT)*innerH;
      ctx.strokeStyle = 'rgba(127,127,127,0.22)';
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + innerW, y);
      ctx.stroke();

      ctx.fillStyle = fg;
      ctx.fillText(_formatTick(v), padL - 6, y);
    });
  }

  // bars
  const nBins = values.length;
  const bw = innerW / nBins;
  ctx.fillStyle = bar;
  values.forEach((v, i) => {
    const bh = ((t(v))/maxT) * innerH;
    const x = padL + i*bw;
    const y = padT + innerH - bh;
    ctx.fillRect(x + 1, y, Math.max(1, bw - 3), bh);
  });

  // x labels (min/mid/max)
  const xMin = edges[0];
  const xMax = edges[edges.length - 1];
  const xMid = edges[Math.floor(edges.length/2)];
  ctx.fillStyle = fg;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const yLab = padT + innerH + 8;
  ctx.fillText(_formatTick(xMin), padL, yLab);
  ctx.fillText(_formatTick(xMid), padL + innerW/2, yLab);
  ctx.fillText(_formatTick(xMax), padL + innerW, yLab);

  // axis titles
  ctx.fillStyle = fg;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText(col, padL + innerW/2, h - 2);

  ctx.save();
  ctx.translate(12, padT + innerH/2);
  ctx.rotate(-Math.PI/2);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText(normalize ? '% of rows' : 'count', 0, 0);
  ctx.restore();

  // caption
  const s = entry.summary || {};
  const flags = `${normalize ? 'normalized' : 'count'}${logY ? ', logY' : ''}`;
  if ($('hist-caption')) $('hist-caption').textContent = `${col} (${flags}): n=${s.count ?? 0}, missing=${s.missing ?? 0} (${((s.missing_rate ?? 0)*100).toFixed(1)}%), range=[${(s.min ?? 0).toFixed(3)}, ${(s.max ?? 0).toFixed(3)}]`;
}

function plotBoxplot(col){
  const canvas = $('box-canvas');
  if (!canvas) return;

  const entry = state.eda?.technical?.numeric?.[col];
  const s = entry?.summary;
  const { ctx, width: w, height: h } = _resizeCanvas(canvas);

  const css = getComputedStyle(document.documentElement);
  const bg = css.getPropertyValue('--color-background-secondary').trim();
  const fg = css.getPropertyValue('--color-text-tertiary').trim();
  const axis = css.getPropertyValue('--color-border-secondary').trim();
  const accent = (css.getPropertyValue('--dot-purple').trim() || '#6B46C1');

  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = bg;
  ctx.fillRect(0,0,w,h);

  if (!s || !(Number.isFinite(s.min) && Number.isFinite(s.max) && Number.isFinite(s.p25) && Number.isFinite(s.p75) && Number.isFinite(s.median))){
    if ($('box-caption')) $('box-caption').textContent = 'Boxplot needs numeric summary (run EDA + pick a numeric column).';
    return;
  }

  const min = s.min, q1 = s.p25, med = s.median, q3 = s.p75, max = s.max;
  const span = (max - min) || 1;

  const padL = 52;
  const padR = 14;
  const padT = 16;
  const padB = 24;
  const innerW = w - padL - padR;
  const midY = Math.round((padT + (h - padB)) / 2);

  const x = (v) => padL + ((v - min) / span) * innerW;

  // axis line
  ctx.strokeStyle = axis;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, midY);
  ctx.lineTo(padL + innerW, midY);
  ctx.stroke();

  // whiskers
  ctx.strokeStyle = accent;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x(min), midY);
  ctx.lineTo(x(q1), midY);
  ctx.moveTo(x(q3), midY);
  ctx.lineTo(x(max), midY);
  ctx.stroke();

  // caps
  ctx.beginPath();
  ctx.moveTo(x(min), midY - 12);
  ctx.lineTo(x(min), midY + 12);
  ctx.moveTo(x(max), midY - 12);
  ctx.lineTo(x(max), midY + 12);
  ctx.stroke();

  // box
  const boxH = 28;
  ctx.fillStyle = 'rgba(107,70,193,0.18)';
  ctx.strokeStyle = accent;
  ctx.lineWidth = 2;
  ctx.fillRect(x(q1), midY - boxH/2, Math.max(1, x(q3) - x(q1)), boxH);
  ctx.strokeRect(x(q1), midY - boxH/2, Math.max(1, x(q3) - x(q1)), boxH);

  // median
  ctx.beginPath();
  ctx.moveTo(x(med), midY - boxH/2);
  ctx.lineTo(x(med), midY + boxH/2);
  ctx.stroke();

  // ticks (min, median, max)
  ctx.fillStyle = fg;
  ctx.font = '11px ui-sans-serif, system-ui';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const yLab = midY + boxH/2 + 10;
  ctx.fillText(_formatTick(min), x(min), yLab);
  ctx.fillText(_formatTick(med), x(med), yLab);
  ctx.fillText(_formatTick(max), x(max), yLab);

  // y-axis label (just title)
  ctx.textBaseline = 'bottom';
  ctx.fillText(`${col} (boxplot)`, padL + innerW/2, h - 2);

  if ($('box-caption')) $('box-caption').textContent = `min=${_formatTick(min)} · Q1=${_formatTick(q1)} · median=${_formatTick(med)} · Q3=${_formatTick(q3)} · max=${_formatTick(max)}`;
}

function plotMissingness(){
  const canvas = $('missing-canvas');
  if (!canvas) return;
  const miss = state.eda?.technical?.missingness || {};

  const entries = Object.entries(miss).sort((a,b) => (b[1]||0) - (a[1]||0)).slice(0, 10);
  const { ctx, width: w, height: h } = _resizeCanvas(canvas);

  const css = getComputedStyle(document.documentElement);
  const bg = css.getPropertyValue('--color-background-secondary').trim();
  const fg = css.getPropertyValue('--color-text-tertiary').trim();
  const bar = (css.getPropertyValue('--dot-warn').trim() || '#D97706');

  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = bg;
  ctx.fillRect(0,0,w,h);

  if (!entries.length){
    if ($('missing-caption')) $('missing-caption').textContent = 'Run EDA to visualize missingness.';
    return;
  }

  const padL = 90;
  const padR = 12;
  const padT = 14;
  const padB = 18;
  const innerW = w - padL - padR;
  const innerH = h - padT - padB;
  const rowH = innerH / entries.length;

  // grid
  ctx.strokeStyle = 'rgba(127,127,127,0.22)';
  for (let i=0;i<=4;i++){
    const x = padL + (innerW*i)/4;
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, h - padB);
    ctx.stroke();
  }

  entries.forEach(([col, rate], i) => {
    const y = padT + i*rowH;
    ctx.fillStyle = fg;
    ctx.font = '12px ui-sans-serif, system-ui';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(col).slice(0, 18), 10, y + rowH/2);

    const bw = Math.max(0, Math.min(1, rate)) * innerW;
    ctx.fillStyle = bar;
    ctx.fillRect(padL, y + 4, bw, Math.max(6, rowH - 8));

    ctx.fillStyle = fg;
    ctx.fillText(`${(rate*100).toFixed(1)}%`, padL + bw + 6, y + rowH/2);
  });

  if ($('missing-caption')) $('missing-caption').textContent = 'Top columns by missingness (higher = more missing values).';
}

function plotBalance(){
  const canvas = $('balance-canvas');
  if (!canvas) return;
  const ta = state.eda?.technical?.target_analysis;
  const { ctx, width: w, height: h } = _resizeCanvas(canvas);

  const css = getComputedStyle(document.documentElement);
  const bg = css.getPropertyValue('--color-background-secondary').trim();
  const fg = css.getPropertyValue('--color-text-tertiary').trim();
  const axis = css.getPropertyValue('--color-border-secondary').trim();
  const bar = (css.getPropertyValue('--dot-success').trim() || '#16A34A');

  const asPercent = !!$('balance-percent')?.checked;

  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = bg;
  ctx.fillRect(0,0,w,h);

  if (!ta || ta.task !== 'classification' || !ta.counts?.length){
    const cands = state.eda?.technical?.features?.target_candidates || [];
    const hint = cands.length
      ? `Suggested outcomes: ${cands.map(x => x.column).slice(0,4).join(', ')}`
      : 'Pick an outcome column to analyze balance.';
    if ($('balance-caption')) $('balance-caption').textContent = hint;
    return;
  }

  const entries = ta.counts.slice(0, 10);
  const values = asPercent ? entries.map(x => (x.ratio ?? 0) * 100.0) : entries.map(x => x.count);
  const maxV = Math.max(...values, 1e-9);

  const padL = 52;
  const padR = 14;
  const padT = 14;
  const padB = 34;
  const innerW = w - padL - padR;
  const innerH = h - padT - padB;

  // axes
  ctx.strokeStyle = axis;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + innerH);
  ctx.lineTo(padL + innerW, padT + innerH);
  ctx.stroke();

  // y ticks + grid
  const ticks = _niceTicks(maxV, 4);
  ctx.font = '11px ui-sans-serif, system-ui';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ticks.forEach(v => {
    const y = padT + innerH - (v/maxV)*innerH;
    ctx.strokeStyle = 'rgba(127,127,127,0.22)';
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + innerW, y);
    ctx.stroke();
    ctx.fillStyle = fg;
    ctx.fillText(_formatTick(v) + (asPercent ? '%' : ''), padL - 6, y);
  });

  // bars
  const bw = innerW / entries.length;
  ctx.fillStyle = bar;
  values.forEach((v, i) => {
    const bh = (v/maxV) * innerH;
    const x0 = padL + i*bw;
    const y0 = padT + innerH - bh;
    ctx.fillRect(x0 + 1, y0, Math.max(1, bw - 6), bh);

    ctx.fillStyle = fg;
    ctx.font = '11px ui-sans-serif, system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(String(entries[i].label).slice(0, 8), x0 + (bw/2) - 2, padT + innerH + 8);
    ctx.fillStyle = bar;
  });

  // y-axis title
  ctx.save();
  ctx.fillStyle = fg;
  ctx.translate(12, padT + innerH/2);
  ctx.rotate(-Math.PI/2);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.font = '11px ui-sans-serif, system-ui';
  ctx.fillText(asPercent ? '% of rows' : 'count', 0, 0);
  ctx.restore();

  if ($('balance-caption')) $('balance-caption').textContent = `${ta.target}: imbalance ratio ≈ ${(ta.imbalance_ratio ?? 1).toFixed(2)} · ${ta.recommendation || ''}`;
}

async function _refreshSamples(){
  if (!state.datasetId) return;
  try {
    state.curatedSample = await api(`/api/part1/curated/sample?dataset_id=${encodeURIComponent(state.datasetId)}&limit=12`);
  } catch (_) { state.curatedSample = null; }
  try {
    state.quarantineSample = await api(`/api/part1/quarantine/sample?dataset_id=${encodeURIComponent(state.datasetId)}&limit=12`);
  } catch (_) { state.quarantineSample = null; }
}

function _fillTargetSelect(columns){
  const sel = $('target-select');
  if (!sel) return;
  sel.innerHTML = '<option value="">Select outcome column (optional)</option>';
  (columns || []).forEach(c => {
    const opt = document.createElement('option');
    opt.value = c;
    opt.textContent = c;
    sel.appendChild(opt);
  });
}

async function ingestFile(file){
  setExpStatus('sd-run','Ingesting','processing…');
  try {
    const form = new FormData();
    form.append('file', file);
    const body = await api('/api/part1/ingest', { method:'POST', form });
    state.ingest = body;
    state.datasetId = body.dataset_id;
    state.eda = null;

    const cols = Object.keys(body.schema_info?.columns || {});
    const defaultTarget = cols.find(c => !c.toLowerCase().includes('id')) || cols[0] || '';
    $('target-col').value = defaultTarget;

    _fillTargetSelect(cols);

    // Keep outcome blank by default; user can pick (or click a suggested candidate).
    if ($('target-select')) $('target-select').value = '';

    await _refreshSamples();

    setExpStatus('sd-ok','Ingest complete', `${file.name} · dataset_id=${body.dataset_id}`);
    rerender();

    // Auto-run EDA once for visibility (silent)
    await runEDA({ silent:true });

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
        session_id: state.sessionId,
        user_context: state.agentContext,
        dataset_id: state.datasetId,
        message: msg,
      }
    });

    const rationale = `<div class="rationale-box">${escapeHtml(body.rationale)}</div>`;
    appendBubble('ai', `${escapeHtml(body.answer)}${rationale}<div class="mini" style="margin-top:8px">Tip: use the <strong>Search</strong> tab to find related research and get an AI insight.</div>`);

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

async function runSearch(){
  const q = ($('search-inp')?.value || '').trim();
  if (!q) return;

  // optimistic UI
  $('search-insight').textContent = 'Searching…';
  $('search-results').innerHTML = '<div class="mini">Loading…</div>';

  const body = await api('/api/part3/search', {
    method:'POST',
    json: {
      session_id: state.sessionId,
      user_context: state.agentContext,
      dataset_id: state.datasetId,
      query: q,
      top_k: 10,
    }
  });

  state.search = {
    query: q,
    technical: body.technical,
    investigator: body.investigator,
    filter: state.search?.filter || 'papers'
  };

  renderSearch();
}

function _setAmlStatus(kind, label, val){
  const row = $('aml-status');
  if (!row) return;
  row.classList.remove('hidden');
  const dot = $('aml-status-dot');
  dot.classList.remove('sd-ok','sd-warn','sd-run');
  dot.classList.add(kind);
  $('aml-status-label').textContent = label;
  $('aml-status-val').textContent = val || '';
}

function _renderAmlMetrics(results){
  const el = $('aml-metrics');
  if (!el) return;
  if (!results){
    el.innerHTML = '';
    return;
  }

  const task = results.task || '';
  const best = results.best_model_id || '';
  const dropped = (results.dropped_columns || []).join(', ');
  const models = results.models || {};
  const rows = Object.entries(models).map(([k,v]) => {
    const m = v.metrics || {};
    const cols = task === 'classification'
      ? [`AUC: ${Number(m.auc ?? 0).toFixed(3)}`, `Recall: ${Number(m.recall ?? 0).toFixed(3)}`, `Acc: ${Number(m.accuracy ?? 0).toFixed(3)}`]
      : [`R²: ${Number(m.r2 ?? 0).toFixed(3)}`, `RMSE: ${Number(m.rmse ?? 0).toFixed(3)}`];
    const badge = (k === best) ? '<span class="sp-badge idx">best</span>' : '';
    return `<div class="sp-card"><div class="sp-card-title">${escapeHtml(k)} ${badge}</div><div class="sp-card-sub">${escapeHtml(cols.join(' · '))}</div></div>`;
  }).join('');

  el.innerHTML = `
    <div class="note">
      <div class="note-title">Evaluation summary</div>
      <div class="note-body">
        <div><strong>Task:</strong> ${escapeHtml(task)} · <strong>Best:</strong> ${escapeHtml(best || '—')}</div>
        <div class="mini" style="margin-top:6px"><strong>Dropped:</strong> ${escapeHtml(dropped || 'none')}</div>
      </div>
    </div>
    <div style="margin-top:10px;display:flex;flex-direction:column;gap:10px">${rows}</div>
  `;

  const sel = $('aml-model-select');
  if (sel){
    const cur = sel.value;
    sel.innerHTML = '<option value="">Select model to deploy (after evaluation)</option>';
    Object.keys(models).forEach((k) => {
      const opt = document.createElement('option');
      opt.value = k;
      opt.textContent = k + (k === best ? ' (best)' : '');
      sel.appendChild(opt);
    });
    if (cur && Object.keys(models).includes(cur)) sel.value = cur;
    else if (best) sel.value = best;
  }
}

async function amlTrainNow(){
  const target = ($('exp-target')?.value || '').trim();
  const dropRaw = ($('exp-drop')?.value || '').trim();
  const drop = dropRaw ? dropRaw.split(',').map(s => s.trim()).filter(Boolean) : [];

  if (!state.datasetId){
    alert('Upload a dataset first.');
    return;
  }
  if (!target){
    alert('Target column is required.');
    return;
  }

  state.aml = { job_id: null, status: 'starting', results: null, deploy: null };
  rerender();
  _setAmlStatus('sd-run','Azure ML job','submitting…');

  try {
    const body = await api('/api/part2/aml/train', {
      method:'POST',
      json: { dataset_id: state.datasetId, target_column: target, drop_columns: drop }
    });
    state.aml.job_id = body.job_id;
    state.aml.status = body.status;
    state.aml.studio_url = body.studio_url;
    rerender();

    _setAmlStatus('sd-run','Azure ML job', `${body.job_id} · ${body.status}`);
    await pollAmlJob(body.job_id);
  } catch (e){
    state.aml = null;
    rerender();
    _setAmlStatus('sd-warn','Azure ML job', e.message);
  }
}

async function pollAmlJob(jobId){
  // Poll until completion or error; keep it lightweight.
  for (let i=0; i<120; i++){
    await new Promise(r => setTimeout(r, 1500));
    try {
      const st = await api(`/api/part2/aml/jobs/${encodeURIComponent(jobId)}`);
      state.aml.status = st.status;
      if (st.results){
        state.aml.results = st.results;
        _setAmlStatus('sd-ok','Azure ML job', `${jobId} · ${st.status}`);
        _renderAmlMetrics(st.results);
        rerender();
        return;
      }
      _setAmlStatus('sd-run','Azure ML job', `${jobId} · ${st.status}`);
      rerender();

      const s = String(st.status || '').toLowerCase();
      if (['failed','canceled','cancelled','error'].some(x => s.includes(x))){
        _setAmlStatus('sd-warn','Azure ML job', `${jobId} · ${st.status}`);
        return;
      }
    } catch (_) {
      // keep polling
    }
  }
}

async function amlDeployNow(){
  const jobId = state.aml?.job_id;
  const modelId = ($('aml-model-select')?.value || '').trim();
  const endpointName = ($('aml-endpoint')?.value || '').trim();

  if (!jobId) return alert('Run training first.');
  if (!modelId) return alert('Select a model to deploy.');

  $('aml-deploy').textContent = 'Deploying…';
  try {
    const body = await api('/api/part2/aml/deploy', { method:'POST', json: { job_id: jobId, model_id: modelId, endpoint_name: endpointName || null } });
    state.aml.deploy = body;
    $('aml-deploy').innerHTML = `Deployed to <strong>${escapeHtml(body.endpoint_name)}</strong> (${escapeHtml(body.deployment_name)}). ${body.scoring_uri ? `Scoring URI: <a href="${escapeHtml(body.scoring_uri)}" target="_blank" rel="noopener">${escapeHtml(body.scoring_uri)}</a>` : ''}`;
    rerender();
  } catch (e){
    $('aml-deploy').textContent = 'Deploy error: ' + e.message;
  }
}

async function trainNow(){
  // Existing local training path (kept for fallback)
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

function bindUploadZone(zoneId, inputId){
  const zone = $(zoneId);
  const input = $(inputId);

  zone.addEventListener('click', () => input.click());

  input.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (file) ingestFile(file);
    input.value = '';
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

function setupUploadZones(){
  bindUploadZone('upload-zone', 'csvFile');
  bindUploadZone('upload-zone-data', 'dataFile');
}

function _updateContextCount(){
  const el = $('context-count');
  if (!el) return;
  el.textContent = String((state.agentContext || '').length);
}

function setupEvents(){
  // Agent context persistence
  const ctx = $('agent-context');
  if (ctx){
    ctx.addEventListener('input', () => {
      state.agentContext = String(ctx.value || '');
      localStorage.setItem('agent_context', state.agentContext);
      _updateContextCount();
    });
  }

  $('btn-context-clear')?.addEventListener('click', () => {
    state.agentContext = '';
    localStorage.setItem('agent_context', state.agentContext);
    if ($('agent-context')) $('agent-context').value = '';
    _updateContextCount();
  });

  $('btn-context-template')?.addEventListener('click', () => {
    const template = [
      'Goal: ',
      'Question/Hypothesis: ',
      'Process/Protocol (steps): ',
      'Dataset: source, size, what each row represents: ',
      'Key variables (and units): ',
      'Outcome/target (definition): ',
      'Covariates/confounders: ',
      'Constraints: ',
      'What you want to optimize: ',
    ].join('\n');

    if (!state.agentContext.trim()){
      state.agentContext = template;
    } else {
      state.agentContext = (state.agentContext.trimEnd() + '\n\n' + template);
    }

    localStorage.setItem('agent_context', state.agentContext);
    if ($('agent-context')) $('agent-context').value = state.agentContext;
    _updateContextCount();
  });
  document.querySelectorAll('.tab').forEach((t) => {
    t.addEventListener('click', () => setTab(t.dataset.tab));
  });

  $('vt-researcher').addEventListener('click', () => setView('researcher'));
  $('vt-technical').addEventListener('click', () => setView('technical'));

  document.querySelectorAll('.dc-tab[data-view]').forEach((t) => {
    t.addEventListener('click', () => setView(t.dataset.view));
  });

  $('sp-close').addEventListener('click', closePanel);

  $('btn-search')?.addEventListener('click', runSearch);
  $('search-inp')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') runSearch();
  });

  // Search filter tabs
  $('search-tab-papers')?.addEventListener('click', () => {
    if (!state.search) state.search = { filter: 'papers' };
    state.search.filter = 'papers';
    renderSearch();
  });
  $('search-tab-datasets')?.addEventListener('click', () => {
    if (!state.search) state.search = { filter: 'datasets' };
    state.search.filter = 'datasets';
    renderSearch();
  });

  // Side panel filter tabs
  $('sp-tab-papers')?.addEventListener('click', () => {
    state.sidePanelFilter = 'papers';
    _setSidePanelTabsActive();
    if (state.ragCache) renderSidePanelFromRag(state.ragCache);
  });
  $('sp-tab-datasets')?.addEventListener('click', () => {
    state.sidePanelFilter = 'datasets';
    _setSidePanelTabsActive();
    if (state.ragCache) renderSidePanelFromRag(state.ragCache);
  });


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

  $('btn-aml-train')?.addEventListener('click', amlTrainNow);
  $('btn-aml-deploy')?.addEventListener('click', amlDeployNow);
  $('btn-drift').addEventListener('click', driftNow);

  $('btn-data-feedback')?.addEventListener('click', askDataFeedback);
  $('btn-data-feedback-clear')?.addEventListener('click', () => {
    state.dataFeedback = null;
    rerender();
  });

  // EDA controls are re-rendered; use event delegation.
  document.addEventListener('click', (e) => {
    if (e.target?.id === 'btn-eda') runEDA();
    if (e.target?.id === 'btn-eda-refresh') runEDA();
    if (e.target?.dataset?.setTarget){
      const col = e.target.dataset.setTarget;
      if ($('target-select')) $('target-select').value = col;
      // keep training target synced as a convenience
      if ($('target-col')) $('target-col').value = col;
      runEDA({ silent:true });
    }
    if (e.target?.id === 'btn-plot') plotDistribution();
  });
  $('target-select')?.addEventListener('change', () => runEDA({ silent:true }));

  $('balance-percent')?.addEventListener('change', () => plotBalance());

  $('dist-select')?.addEventListener('change', () => plotDistribution());
  $('dist-normalize')?.addEventListener('change', () => plotDistribution());
  $('dist-logy')?.addEventListener('change', () => plotDistribution());
  $('dist-bins')?.addEventListener('change', () => runEDA({ silent:true }));

  // Side panel is still available for future expansion; Search tab is the primary UI for related research now.

}

function _getOrCreateSessionId(){
  const existing = localStorage.getItem('session_id');
  if (existing) return existing;
  let sid = '';
  try { sid = crypto.randomUUID(); } catch (_) { sid = `sess_${Math.random().toString(16).slice(2)}_${Date.now()}`; }
  localStorage.setItem('session_id', sid);
  return sid;
}

function aiFeedbackNarrative(){
  if (!state.dataFeedback){
    return '<strong>No feedback yet.</strong><div class="mini" style="margin-top:6px">Click “Ask AI for feedback” after uploading a dataset.</div>';
  }
  const inv = state.dataFeedback.investigator || {};
  const summary = inv.summary ? `<strong>${escapeHtml(inv.summary)}</strong>` : '<strong>AI feedback</strong>';
  const bullets = (inv.bullets || []).slice(0, 8).map(x => `<li>${escapeHtml(x)}</li>`).join('');
  const warns = (inv.warnings || []).slice(0, 6).map(x => `<div class="mini" style="color:var(--color-text-warning)">• ${escapeHtml(x)}</div>`).join('');
  return `${summary}${warns ? `<div style="margin-top:8px">${warns}</div>` : ''}${bullets ? `<ul style="margin:10px 0 0 18px">${bullets}</ul>` : ''}`;
}

async function askDataFeedback(){
  if (!state.datasetId){
    alert('Upload a dataset first.');
    return;
  }
  try {
    const body = await api('/api/part3/data/feedback', {
      method:'POST',
      json: {
        session_id: state.sessionId,
        user_context: state.agentContext,
        dataset_id: state.datasetId,
      }
    });
    state.dataFeedback = body;
    rerender();
    setTab('t-data');
  } catch (e){
    alert('AI feedback error: ' + e.message);
  }
}

async function init(){
  applyTheme(getInitialTheme());

  state.sessionId = _getOrCreateSessionId();
  state.agentContext = localStorage.getItem('agent_context') || '';
  if ($('agent-context')) $('agent-context').value = state.agentContext;
  _updateContextCount();

  setupEvents();
  setupUploadZones();
  setView('researcher');
  setSafetyPill(false);

  try {
    const h = await api('/api/health');
    if (h?.azure_ml_compute_name) state.amlCompute = h.azure_ml_compute_name;
  } catch (_) {}

  rerender();
}

window.addEventListener('DOMContentLoaded', init);
