// Headless test for the Settings page (hybrid backend) in web/index.html.
//   node tests/ui_settings_test.js [path/to/index.html]
//
// Three halves: (a) the markup really contains everything the feature needs
// and did not clobber the existing Chat/Agent features, (b) the settings
// script, run against a stub DOM, talks to routes/settings_api.py's endpoints
// and only ever renders masked keys, (c) a non-JSON answer (the bug the user
// reported) is turned into a readable message, not a JSON.parse crash.
'use strict';
const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync(process.argv[2] || 'web/index.html', 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const main = scripts.find(s => s.includes('renderSessions'));
if (!main) { console.error('FAIL: main script not found'); process.exit(1); }

const START = '// ── Settings (hybrid backend';
const startIdx = main.indexOf(START);
if (startIdx < 0) { console.error('FAIL: settings block not found'); process.exit(1); }
const block = main.slice(startIdx);

let failures = 0;
function ok(c, m) { if (!c) { failures++; console.error('FAIL ' + m); } else console.log('ok   ' + m); }
function eq(got, want, msg) {
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g !== w) { failures++; console.error(`FAIL ${msg}\n  got  ${g}\n  want ${w}`); }
  else console.log(`ok   ${msg}`);
}
function has(needle, msg) { ok(html.includes(needle), msg); }

// ────────────────────────── (a) markup ──────────────────────────
console.log('--- markup ---');
ok(html.includes('id="settings-btn"') && html.includes('data-cmd="settings"'),
   'sidebar has a ⚙️ Settings button wired to data-cmd');
// Release 9.1 — the drawer's Commands shortcuts are collapsed by
// default under a toggle heading; Close menu stays outside the fold.
has('id="cmd-toggle"', 'R9.1: Commands heading is the toggle');
has('id="cmd-list" style="display:none"', 'R9.1: command list starts hidden');
ok(html.indexOf('data-cmd="memory"') > html.indexOf('id="cmd-list"') &&
   html.indexOf('data-cmd="memory"') < html.indexOf('data-cmd="close"'),
   'R9.1: the shortcut buttons live inside the fold, Close menu outside');
has('id="settings-modal"', 'settings modal exists');
['set-mode-auto', 'set-mode-local', 'set-mode-cloud', 'set-provider', 'set-enabled',
 'set-model', 'set-model-custom', 'set-base-url', 'set-cost', 'set-fallback',
 'set-key-list', 'set-key-input', 'set-key-label', 'set-key-add', 'set-msg',
 'set-net', 'backend-mode-hint', 'set-local-note', 'set-save', 'set-cancel',
 'settings-close', 'settings-body'].forEach(id => has('id="' + id + '"', 'element #' + id));
ok(/id="set-key-input"[^>]*type="password"|type="password"[^>]*id="set-key-input"/.test(html.replace(/\n/g, '')),
   'API key input is type=password (never echoed on screen)');
has('name="backend-mode"', 'backend mode radios share name="backend-mode"');
ok(html.includes("if(c==='settings'){openSettings();return}"),
   'drawer command dispatch routes settings to openSettings()');

console.log('--- prompt tier markup ---');
has('name="prompt-tier-mode"', 'prompt tier radios share name="prompt-tier-mode"');
['set-prompt-tier', 'set-tier-auto', 'set-tier-cloud-strong', 'set-tier-cloud-fast',
 'set-tier-local', 'prompt-tier-hint', 'prompt-tier-state',
 'prompt-tier-warn'].forEach(id => has('id="' + id + '"', 'element #' + id));
ok(block.includes('renderPromptTier'), 'the settings block renders the tier panel');
ok(block.includes('prompt_tier_mode:tierMode'), 'save sends the chosen tier mode');

console.log('--- existing features ---');
['id="voice-btn"', 'id="attach-btn"', 'id="image-btn"', 'id="add-btn"',
 'id="send-btn"', 'id="sel-bar"', 'id="sel-all"', 'id="session-list"',
 'id="proj-modal"', 'id="drawer"', 'id="viewer"', 'id="image-chip"',
 'id="attach-chip"', 'data-mode="chat"', 'data-mode="agent"', 'data-mode="code"'
].forEach(s => has(s, 'still present: ' + s));

// ─────────────────────── (b) ES5 guard ────────────────────────
console.log('--- ES5 guard (settings block) ---');
const code = block.split('\n').filter(l => !/^\s*\/\//.test(l)).join('\n');
ok(!code.includes('=>'), 'no arrow functions');
ok(!/\bconst\b/.test(code), 'no const');
ok(!/\blet\b/.test(code), 'no let');
ok(!code.includes('new Set('), 'no Set');
ok(!code.includes('`'), 'no template literals');

// ─────────── (b2) manual key selection is wired up ────────────
console.log('--- key selection ---');
ok(code.includes('set-active-key'), 'selection radios share one name');
ok(code.includes('activateKey'), 'activateKey() exists');
ok(block.includes('/activate'), 'the activate endpoint is used');
ok(block.includes('✅ Active'), 'the key in play is badged');
ok(block.includes('refreshCtx'), 'saving repaints the context meter');
ok(block.includes('updateModelInfo'), 'saving repaints the model chip');

// ───────────── (b3) the context meter is the model's own window ────
console.log('--- context meter ---');
ok(html.includes('id="ctx-warn"'), 'the meter has its own warning line');
ok(html.includes('id="ctx-pct"') && html.includes('id="ctx-text"'),
   'meter keeps the token text and the percentage');
has('--orange:#f97316', 'a dark-theme orange band for "trimming soon"');
has('--orange:#c2410c', 'a light-theme orange band too');
ok(main.includes('local_total'), 'the KV buttons keep reading the local total');

const CTX_START = 'function fmtTok(n){';
const CTX_END = 'async function refreshCtx(){';
const ci = main.indexOf(CTX_START), cj = main.indexOf(CTX_END);
ok(ci >= 0 && cj > ci, 'context meter block found in the main script');
const ctxCode = ci >= 0 && cj > ci ? main.slice(ci, cj) : '';
ok(ctxCode.includes('trimming soon'), 'the 90% band warns before a refusal');
ok(ctxCode.includes('context nearly full'), 'the 95% band names the fix');
ok(ctxCode.includes('ctx.model'), 'the tooltip names the model the window belongs to');

// ─────────────────── (c) behaviour on a stub DOM ────────────────
console.log('--- behaviour ---');

function makeEl(tag) {
  const node = {
    tagName: tag || 'div', id: '', className: '', type: '', title: '',
    value: '', textContent: '', checked: false, disabled: false, hidden: false,
    style: {}, attrs: {}, children: [], parentNode: null, listeners: {},
    appendChild(c) { this.children.push(c); c.parentNode = this; return c; },
    setAttribute(k, v) { this.attrs[k] = String(v); },
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); },
    removeEventListener(ev, fn) {
      const l = this.listeners[ev] || [];
      const i = l.indexOf(fn); if (i >= 0) l.splice(i, 1);
    },
    fire(ev, e) {
      const l = (this.listeners[ev] || []).slice();
      const evt = e || { target: this, preventDefault() {}, stopPropagation() {} };
      for (let i = 0; i < l.length; i++) l[i].call(this, evt);
    },
    focus() {},
    click() { this.fire('click'); },
    dispatchEvent(ev) { this.fire(ev && ev.type); return true; },
  };
  let inner = '';
  Object.defineProperty(node, 'innerHTML', {
    get() { return inner; },
    set(v) { inner = v; if (v === '') node.children = []; },
    configurable: true,
  });
  return node;
}

function texts(node) {
  let out = String(node.textContent || '');
  for (const c of node.children || []) out += ' ' + texts(c);
  return out;
}

// The meter, run for real against a stub DOM: four bands, one tooltip.
const ctxReg = {};
['ctx-fill', 'ctx-info', 'ctx-text', 'ctx-warn', 'ctx-pct'].forEach(id => {
  const e = makeEl('div'); e.id = id; ctxReg[id] = e;
});
const ctxSandbox = {
  document: { getElementById(id) { return ctxReg[id] || null; } },
  Math, String, Number, Boolean, JSON,
};
vm.createContext(ctxSandbox);
vm.runInContext(ctxCode, ctxSandbox, { filename: 'ctx-meter.js' });

function meter(pct, used, total) {
  ctxSandbox.updateCtx({ used: used, total: total, pct: pct,
                          model: 'deepseek-flash', source: 'exact' });
  return ctxReg;
}

meter(0, 0, 128000);
eq(ctxReg['ctx-text'].textContent, 'ctx: 0/128K', 'empty meter reads 0/128K');
eq(ctxReg['ctx-fill'].className, 'ctx-fill', 'an empty meter is not coloured');
eq(ctxReg['ctx-info'].className, 'ctx-info', 'below 50% the row stays neutral');
eq(ctxReg['ctx-warn'].textContent, '', 'no nagging at 0%');
eq(ctxReg['ctx-fill'].style.width, '0%', 'the bar is empty at 0%');

meter(54.7, 70000, 128000);
eq(ctxReg['ctx-info'].className, 'ctx-info warn', '50-89% turns amber');
eq(ctxReg['ctx-warn'].textContent, '', 'amber is quiet — no wording');

meter(92.2, 118000, 128000);
eq(ctxReg['ctx-info'].className, 'ctx-info over', '90-94% turns orange');
eq(ctxReg['ctx-fill'].className, 'ctx-fill over', 'the bar matches the row');
eq(ctxReg['ctx-warn'].textContent, 'trimming soon', 'orange warns before a refusal');

meter(96.9, 124000, 128000);
eq(ctxReg['ctx-info'].className, 'ctx-info crit', '>=95% turns red');
eq(ctxReg['ctx-fill'].className, 'ctx-fill crit', 'the bar turns red too');
ok(ctxReg['ctx-warn'].textContent.includes('/clear'),
   'the red band tells the user how to fix it: ' + ctxReg['ctx-warn'].textContent);
ok(ctxReg['ctx-info'].title.includes('deepseek-flash') &&
   ctxReg['ctx-info'].title.includes('exact'),
   'the tooltip names the model and where the limit came from');
eq(ctxReg['ctx-pct'].textContent, '96.9%', 'the percentage is shown to one decimal');

meter(140, 200000, 128000);
eq(ctxReg['ctx-fill'].style.width, '100%', 'the bar never overflows its track');

const registry = {};
const IDS = ['settings-modal', 'settings-close', 'settings-body', 'set-modes',
  'backend-mode-hint', 'set-local-note', 'set-net', 'set-provider', 'set-enabled',
  'set-fallback', 'set-model', 'set-model-custom', 'set-base-url', 'set-cost',
  'set-ctx-limit', 'set-key-list', 'set-key-input', 'set-key-label', 'set-key-add',
  'set-msg', 'set-save', 'set-cancel', 'set-prompt-tier', 'set-tier-auto',
  'set-tier-cloud-strong', 'set-tier-cloud-fast', 'set-tier-local',
  'prompt-tier-hint', 'prompt-tier-state', 'prompt-tier-warn'];
IDS.forEach(id => { const e = makeEl('div'); e.id = id; registry[id] = e; });

const radios = ['auto', 'local', 'cloud'].map(v => {
  const e = makeEl('input');
  e.type = 'radio'; e.value = v; e.checked = (v === 'auto');
  return e;
});

// Same shape as the backend radios: one checked at a time, and the code under
// test only ever reads .value / .checked.
const TIER_MODES = ['auto', 'cloud_strong', 'cloud_fast', 'local'];
const tierRadios = TIER_MODES.map(v => {
  const e = makeEl('input');
  e.type = 'radio'; e.value = v; e.checked = (v === 'auto');
  return e;
});

const RAW_KEY = 'sk-live-raw-secret-key-9876543210';
const PROVIDERS = ['openai', 'gemini', 'openrouter', 'anthropic', 'deepseek'];
const LABELS = { openai: 'OpenAI', gemini: 'Google Gemini', openrouter: 'OpenRouter',
                 anthropic: 'Anthropic', deepseek: 'DeepSeek' };
const MODELS = {
  openai: ['gpt-4o', 'gpt-4o-mini'],
  gemini: ['gemini-2.5-flash', 'gemini-2.5-pro'],
  openrouter: ['google/gemini-2.5-flash', 'openai/gpt-4o-mini'],
  anthropic: ['claude-3-5-haiku-latest'],
  deepseek: ['deepseek-chat', 'deepseek-reasoner'],
};
const REMOTE = {
  ok: true,
  providers: PROVIDERS,
  provider_labels: LABELS,
  modes: ['auto', 'local', 'cloud'],
  prompt_tier_modes: TIER_MODES,
  prompt_tier: { mode: 'auto', resolved: 'cloud_fast', source: 'auto',
                  model: 'gpt-4o-mini', backend: 'cloud', auto: 'cloud_fast',
                  mismatch: false },
  fallback_order: PROVIDERS.slice(),
  local_fallback: true,
  costs: { openai: 0.15, gemini: 0.30, openrouter: 0.00, anthropic: 0.80,
           deepseek: 0.27 },
  models: MODELS,
  config: {
    backend_mode: 'auto', prefer_cloud: true, cost_ack: true,
    fallback: PROVIDERS.slice(),
    providers: {
      openai: { model: 'gpt-4o-mini', base_url: 'https://api.openai.com/v1', enabled: true },
      gemini: { model: 'gemini-2.5-flash', base_url: '', enabled: true },
      openrouter: { model: 'google/gemini-2.5-flash', base_url: '', enabled: true },
      anthropic: { model: 'claude-3-5-haiku-latest', base_url: '', enabled: true },
      deepseek: { model: 'deepseek-chat', base_url: 'https://api.deepseek.com/v1', enabled: true },
    },
    keys: {
      openai: [{ id: 'k_1', label: 'main', key_masked: 'sk-l…3210', status: 'ok',
                 used: 4, active: true },
               { id: 'k_2', label: 'backup', key_masked: 'sk-l…9999', status: 'ok',
                 used: 0, active: false }],
      gemini: [], openrouter: [], anthropic: [], deepseek: [],
    },
  },
  backend: { online: true, state: 'online', local_model: 'model.litertlm',
             providers: { openai: { context_limit: 128000 },
                          deepseek: { context_limit: 64000 } } },
};

const calls = [];
let textMode = false;   // simulate the plain-text 404 that started all this

function jsonRes(obj) { return { status: 200, json: async () => obj, text: async () => JSON.stringify(obj) }; }

async function fetchStub(url, opts) {
  const o = opts || {};
  calls.push({ url, method: o.method || 'GET', body: o.body ? JSON.parse(o.body) : null });
  if (textMode) {
    const plain = 'The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.';
    return { status: 404, text: async () => plain,
             json: async () => { throw new SyntaxError('Unexpected token T'); } };
  }
  if (url === '/api/settings/agent' && (!o.method || o.method === 'GET')) {
    return jsonRes(JSON.parse(JSON.stringify(REMOTE)));
  }
  if (url === '/api/settings/agent' && o.method === 'POST') {
    const merged = JSON.parse(JSON.stringify(REMOTE));
    const body = JSON.parse(o.body);
    if (body.backend_mode) merged.config.backend_mode = body.backend_mode;
    if (typeof body.cost_ack === 'boolean') merged.config.cost_ack = body.cost_ack;
    if (body.prompt_tier_mode) {
      merged.config.prompt_tier_mode = body.prompt_tier_mode;
      // Same arithmetic routes/settings_api.py does: a pinned tier always
      // resolves to itself, and it mismatches when it is not what Auto would
      // have picked for the model that is about to answer.
      const pinned = body.prompt_tier_mode;
      const autoTier = merged.prompt_tier.auto;
      merged.prompt_tier = Object.assign({}, merged.prompt_tier, {
        mode: pinned,
        resolved: pinned === 'auto' ? autoTier : pinned,
        source: pinned === 'auto' ? 'auto' : 'settings',
        mismatch: pinned !== 'auto' && pinned !== autoTier,
      });
    }
    if (body.fallback) { merged.config.fallback = body.fallback; merged.fallback_order = body.fallback; }
    if (body.providers) {
      for (const p of Object.keys(body.providers)) merged.config.providers[p] = body.providers[p];
    }
    return jsonRes({ ok: true, changed: ['backend_mode'], config: merged.config,
                     backend: merged.backend, prompt_tier: merged.prompt_tier });
  }
  if (url === '/api/settings/agent/keys' && o.method === 'POST') {
    const body = JSON.parse(o.body);
    const masked = body.key.slice(0, 4) + '…' + body.key.slice(-4);
    const cfg = JSON.parse(JSON.stringify(REMOTE.config));
    cfg.keys[body.provider] = [{ id: 'k_new', label: body.label, key_masked: masked, status: 'ok' }];
    return jsonRes({ ok: true, key: { id: 'k_new', key_masked: masked }, config: cfg });
  }
  if (o.method === 'DELETE') {
    const cfg = JSON.parse(JSON.stringify(REMOTE.config));
    cfg.keys.openai = [];
    return jsonRes({ ok: true, config: cfg });
  }
  if (url === '/api/settings/agent/keys/test') {
    const body = JSON.parse(o.body);
    if (!body.key_id) return jsonRes({ ok: false, error: 'key is required' });
    return jsonRes({ ok: true, latency_ms: 120, model: 'gpt-4o-mini' });
  }
  if (url === '/api/settings/doctor' && (!o.method || o.method === 'GET')) {
    return jsonRes({ ok: true, config_path: '/data/agent_config.json',
      config_exists: true,
      backend: { online: true, state: 'online', local_model: 'model.litertlm' },
      keys: { openai: { total: 2, healthy: 2, dead: 0, cooling: 0 },
              gemini: { total: 0, healthy: 0, dead: 0, cooling: 0 } },
      disk_free_bytes: 5 * 1073741824,
      storage: [{ label: 'agent data (~/.ai-agent)', bytes: 5242880 },
                { label: 'backup zips', bytes: 1048576 }],
      log_tail: ['2026-10-22 INFO boot ok', '2026-10-22 WARNING demo line'],
      code: { backend: 'opencode', installed: true } });
  }
  if (url === '/api/memory' && (!o.method || o.method === 'GET')) {
    return jsonRes({ text: 'Facts:\n- likes tea' });
  }
  if (url === '/api/remember' && o.method === 'POST') {
    return jsonRes({ ok: true });
  }
  if (url === '/api/memory/forget' && o.method === 'POST') {
    return jsonRes({ ok: true, removed: 2 });
  }
  if (url === '/api/settings/backup' && o.method === 'POST') {
    return jsonRes({ ok: true, file: 'ai-agent-backup-test.zip', files: 12,
                     includes_secrets: true });
  }
  if (url === '/api/settings/agent/reset' && o.method === 'POST') {
    const cfg = JSON.parse(JSON.stringify(REMOTE.config));
    cfg.backend_mode = 'auto';
    return jsonRes({ ok: true, config: cfg, backend: REMOTE.backend,
                     prompt_tier: REMOTE.prompt_tier });
  }
  const act = /^\/api\/settings\/agent\/keys\/([a-z]+)\/([A-Za-z0-9_]+)\/activate$/.exec(url);
  if (act && o.method === 'POST') {
    const cfg = JSON.parse(JSON.stringify(REMOTE.config));
    const list = cfg.keys[act[1]] || [];
    let found = null;
    for (const k of list) { k.active = (k.id === act[2]); if (k.active) found = k; }
    if (!found) {
      const err = JSON.stringify({ error: 'no such key' });
      return { status: 404, json: async () => JSON.parse(err), text: async () => err };
    }
    return jsonRes({ ok: true, key: found, config: cfg });
  }
  return jsonRes({ error: 'unexpected endpoint ' + url });
}

const sandbox = {
  document: {
    getElementById(id) { return registry[id] || (registry[id] = Object.assign(makeEl('div'), { id })); },
    getElementsByName(n) {
      if (n === 'backend-mode') return radios;
      if (n === 'prompt-tier-mode') return tierRadios;
      return [];
    },
    createElement(tag) { return makeEl(tag); },
    querySelectorAll() { return []; },
    documentElement: { getAttribute() { return null; }, setAttribute() {} },
  },
  fetch: fetchStub,
  confirm: () => true,
  console,
  Event: function Event(type, opts) {
    this.type = type;
    this.bubbles = !!(opts && opts.bubbles);
  },
  JSON, Math, Date, Array, Object, String, Number, Boolean, encodeURIComponent,
  // Defined by the main script in the real page; the settings block calls it to
  // print the model's context window, so the stub needs it too.
  fmtTok(n) {
    if (!n) return '0';
    if (n >= 1000000) return (n / 1000000).toFixed(n % 1000000 ? 1 : 0) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(n % 1000 ? 1 : 0) + 'K';
    return '' + n;
  },
  setTimeout: (fn) => { fn(); return 0; },
  clearTimeout() {},
};
vm.createContext(sandbox);
vm.runInContext(block, sandbox, { filename: 'settings-block.js' });

function click(id, ev) { const e = registry[id]; if (e) e.fire(ev || 'click'); }

(async () => {
  // open + load
  sandbox.openSettings();
  eq(registry['settings-modal'].style.display, 'flex', 'openSettings() shows the modal');
  await new Promise(r => setImmediate(r));
  eq(calls[0] && calls[0].url, '/api/settings/agent', 'GET /api/settings/agent on open');

  // backend mode radios
  eq(radios.map(r => r.checked), [true, false, false], 'auto radio is selected');
  ok(texts(registry['backend-mode-hint']).length > 10, 'backend mode hint is filled');
  ok(texts(registry['set-net']).includes('local model'), 'network line shows local model');
  ok(texts(registry['set-local-note']).includes('Local Gemma 4B'),
     'auto mode promises local fallback');

  // ── prompt tier ────────────────────────────────────────────────
  eq(tierRadios.map(r => r.checked), [true, false, false, false],
     'the Auto tier is the one selected on a fresh install');
  ok(texts(registry['prompt-tier-hint']).includes('Auto'),
     'the tier hint explains Auto: ' + texts(registry['prompt-tier-hint']));
  ok(texts(registry['prompt-tier-state']).includes('Resolved: Cloud Fast'),
     'Auto shows the tier it resolved to: ' +
     texts(registry['prompt-tier-state']));
  ok(texts(registry['prompt-tier-state']).includes('gpt-4o-mini'),
     'the resolved line names the model it was decided from');
  eq(registry['prompt-tier-warn'].style.display, 'none',
     'no mismatch warning when Auto agrees with itself');

  // a pinned tier that does not match this turn's model must say so
  sandbox.settingsState.data.prompt_tier = {
    mode: 'cloud_strong', resolved: 'cloud_strong', source: 'settings',
    model: 'gpt-4o-mini', backend: 'cloud', auto: 'cloud_fast', mismatch: true,
  };
  sandbox.renderSettings();
  eq(tierRadios.map(r => r.checked), [false, true, false, false],
     'the pinned tier is the one selected');
  ok(texts(registry['prompt-tier-state']).includes('Pinned: Cloud Strong'),
     'a pinned tier is labelled as pinned: ' +
     texts(registry['prompt-tier-state']));
  ok(registry['prompt-tier-warn'].style.display !== 'none',
     'a pinned tier that fights the model shows the warning badge');
  ok(texts(registry['prompt-tier-warn']).includes('Auto'),
     'the badge says what Auto would have used: ' +
     texts(registry['prompt-tier-warn']));

  // picking a tier is not saving it — say so instead of implying it applied
  tierRadios[3].checked = true;
  tierRadios[3].fire('change');
  ok(texts(registry['prompt-tier-state']).includes('Save'),
     'an unsaved tier says it waits for Save: ' +
     texts(registry['prompt-tier-state']));
  eq(registry['prompt-tier-warn'].style.display, 'none',
     'the warning clears while a new tier is unsaved');
  sandbox.settingsState.data.prompt_tier = {
    mode: 'auto', resolved: 'cloud_fast', source: 'auto', model: 'gpt-4o-mini',
    backend: 'cloud', auto: 'cloud_fast', mismatch: false,
  };
  sandbox.renderSettings();

  // provider dropdown — labels, not ids
  const opts = registry['set-provider'].children.map(o => o.value);
  eq(opts, PROVIDERS, 'all five providers offered');
  eq(registry['set-provider'].children.map(o => o.textContent),
     ['OpenAI', 'Google Gemini', 'OpenRouter', 'Anthropic', 'DeepSeek'],
     'dropdown shows the display labels');
  ok(registry['set-provider'].children[0].selected, 'openai is the selected provider');

  // fallback order
  eq(registry['set-fallback'].children.length, 5, 'fallback order lists every provider');
  eq(texts(registry['set-fallback'].children[0]).trim(), '1. openai ↑ ↓',
     'first fallback row rendered');

  // model presets + cost
  eq(registry['set-model'].children.map(o => o.value), MODELS.openai,
     'model presets for the selected provider');
  eq(registry['set-model-custom'].value, 'gpt-4o-mini', 'model input mirrors the preset');
  ok(texts(registry['set-cost']).includes('$0.15/1K'),
     'cost warning shows the per-1K price');
  ok(texts(registry['set-ctx-limit']).includes('Context window: 128K'),
     'the model\'s own context window is shown: ' + texts(registry['set-ctx-limit']));

  // a provider on a much smaller window gets its own number, never the
  // previous provider's
  registry['set-provider'].value = 'deepseek';
  registry['set-provider'].fire('change');
  ok(texts(registry['set-ctx-limit']).includes('Context window: 64K'),
     'switching provider repaints the window: ' + texts(registry['set-ctx-limit']));
  registry['set-provider'].value = 'openai';
  registry['set-provider'].fire('change');

  // an unknown custom model says so instead of quietly implying a number
  sandbox.settingsState.data.config.providers.openai.model = '';
  sandbox.renderSettings();
  ok(texts(registry['set-ctx-limit']).includes('unknown'),
     'a custom model with no known window says unknown: ' +
     texts(registry['set-ctx-limit']));
  sandbox.settingsState.data.config.providers.openai.model = 'gpt-4o-mini';
  sandbox.renderSettings();

  // keys — masked only
  const keyText = texts(registry['set-key-list']);
  ok(keyText.includes('sk-l…3210'), 'key row shows the masked key');
  ok(keyText.includes('main'), 'key row shows the label');
  ok(keyText.includes('ready'), 'key row shows its status');
  ok(!keyText.includes(RAW_KEY), 'the raw key is never rendered');
  eq(registry['set-key-input'].value, '', 'key input starts empty');

  // manual selection — a radio per key, the badge on the one in play
  const row0 = registry['set-key-list'].children[0];
  const main0 = row0.children[0];
  eq(main0.children[0].tagName, 'label', 'the selector sits in .key-main');
  ok(texts(main0.children[0]).includes('Use this key'),
     'every key offers "Use this key"');
  eq(main0.children[0].children[0].type, 'radio', 'selection is a radio');
  ok(texts(row0).includes('✅ Active'), 'the key in play carries the badge');
  const row1 = registry['set-key-list'].children[1];
  eq(row1.children[0].children[0].children[0].checked, false,
     'only one radio is on');

  // Selecting a key clicks the radio input — from the row:
  // children[0] is .key-main, children[0] the <label>, children[0]
  // the <input>. The change listener lives on the input, so the
  // event must be fired there, not on the label.
  const radio1 = row1.children[0].children[0].children[0];
  radio1.checked = true;
  radio1.fire('change');
  if (typeof radio1.dispatchEvent === 'function') {
    try { radio1.dispatchEvent(new Event('change', { bubbles: true })); } catch (e) {}
  }
  await new Promise(r => setImmediate(r));
  const actCall = calls.find(c => /\/activate$/.test(c.url || ''));
  ok(!!actCall, 'choosing a key POSTs .../activate');
  eq(actCall && actCall.method, 'POST', 'activate is a POST');
  eq(actCall && actCall.url,
     '/api/settings/agent/keys/openai/k_2/activate',
     'activate names the provider and the key id');
  ok(texts(registry['set-msg']).includes('Active key set'),
     'the switch is reported to the user');
  const rows = registry['set-key-list'];
  ok(!texts(rows.children[0]).includes('✅ Active'),
     'the previous key lost the badge');
  ok(texts(rows.children[1]).includes('✅ Active'),
     'the chosen key is now the one marked');
  ok(texts(rows.children[1]).includes('sk-l…9999'),
     'the chosen key still shows only its mask');
  ok(!texts(rows).includes(RAW_KEY), 'no raw key appears after switching');

  // test the seeded key
  registry['set-key-list'].children[0].children[1].children[0].fire('click');
  await new Promise(r => setImmediate(r));
  const testCall = calls.find(c => c.url === '/api/settings/agent/keys/test');
  ok(!!testCall, 'Test POSTs /api/settings/agent/keys/test');
  eq(testCall && testCall.body.key_id, 'k_1', 'Test names the key id (not the key)');
  ok(texts(registry['set-msg']).includes('Key OK'), 'test result is reported');

  // delete the seeded key
  registry['set-key-list'].children[0].children[1].children[1].fire('click');
  await new Promise(r => setImmediate(r));
  const delCall = calls.find(c => c.method === 'DELETE');
  ok(!!delCall, 'Delete sends DELETE');
  eq(delCall && delCall.url, '/api/settings/agent/keys/openai/k_1',
     'Delete URL carries provider and key id');

  // add a key
  const input = registry['set-key-input'];
  input.value = 'sk-new-key-abcdef123456';
  registry['set-key-label'].value = 'second';
  registry['set-key-add'].fire('click');
  await new Promise(r => setImmediate(r));
  const addCall = calls.find(c => c.url === '/api/settings/agent/keys' && c.method === 'POST');
  ok(!!addCall, 'Add POSTs /api/settings/agent/keys');
  eq(addCall && addCall.body.provider, 'openai', 'Add targets the selected provider');
  eq(addCall && addCall.body.key, 'sk-new-key-abcdef123456', 'Add sends the typed key');
  eq(input.value, '', 'key input is cleared after adding');
  const afterAdd = texts(registry['set-key-list']);
  ok(afterAdd.includes('sk-n…3456'), 'new key appears masked');
  ok(!afterAdd.includes('sk-new-key-abcdef123456'), 'new key never rendered raw');

  // fallback reorder
  registry['set-fallback'].children[1].children[3].fire('click'); // ↓ on gemini
  eq(sandbox.settingsState.fallback,
     ['openai', 'openrouter', 'gemini', 'anthropic', 'deepseek'],
     '↓ moves a provider down the fallback order');

  // switch provider
  registry['set-provider'].value = 'gemini';
  registry['set-provider'].fire('change');
  eq(sandbox.settingsState.provider, 'gemini', 'changing the select switches provider');
  eq(registry['set-model'].children.map(o => o.value), MODELS.gemini,
     'model presets follow the provider');
  ok(texts(registry['set-cost']).includes('$0.30'), 'cost follows the provider');
  ok(texts(registry['set-key-list']).includes('မထည့်ရသေးဘူး'),
     'empty key list message for a provider with no keys');

  // DeepSeek is selectable and keeps its own base_url/model
  registry['set-provider'].value = 'deepseek';
  registry['set-provider'].fire('change');
  eq(sandbox.settingsState.provider, 'deepseek', 'DeepSeek is selectable');
  eq(registry['set-model'].children.map(o => o.value), MODELS.deepseek,
     'DeepSeek model presets offered');
  eq(registry['set-base-url'].value, 'https://api.deepseek.com/v1',
     'DeepSeek default base_url is prefilled');
  ok(texts(registry['set-cost']).includes('$0.27'), 'DeepSeek cost shown');

  // back to openai, save as Cloud Only
  registry['set-provider'].value = 'openai';
  registry['set-provider'].fire('change');
  radios[0].checked = false;
  radios[2].checked = true;               // Cloud Only
  radios[2].fire('change');
  ok(texts(registry['backend-mode-hint']).includes('Cloud Only'),
     'selecting Cloud Only updates the hint');
  ok(texts(registry['set-local-note']).includes('off'),
     'Cloud Only turns the local-fallback note off');
  registry['set-model-custom'].value = 'gpt-4o';
  registry['set-enabled'].checked = false;
  tierRadios[2].checked = true;             // Cloud Fast
  tierRadios[2].fire('change');
  registry['set-save'].fire('click');
  await new Promise(r => setImmediate(r));
  const save = calls.filter(c => c.url === '/api/settings/agent' && c.method === 'POST').pop();
  ok(!!save, 'Save POSTs /api/settings/agent');
  eq(save && save.body.backend_mode, 'cloud', 'Save sends backend_mode');
  eq(save && save.body.prompt_tier_mode, 'cloud_fast',
     'Save sends the chosen prompt tier');
  eq(save && save.body.cost_ack, true, 'cloud modes acknowledge the price on save');
  eq(save && save.body.fallback,
     ['openai', 'openrouter', 'gemini', 'anthropic', 'deepseek'],
     'Save sends the reordered fallback');
  eq(save && save.body.providers.openai,
     { model: 'gpt-4o', enabled: false, base_url: 'https://api.openai.com/v1' },
     'Save sends the edited provider settings');

  // Local Only costs are hidden
  radios[2].checked = false;
  radios[1].checked = true;               // Local Only
  radios[1].fire('change');
  ok(texts(registry['backend-mode-hint']).includes('Local Only'),
     'Local Only hint shown');

  // enter key in the key input adds
  registry['set-key-input'].value = 'sk-enter-press-0001112223';
  registry['set-key-input'].fire('keydown', { keyCode: 13, preventDefault() {} });
  await new Promise(r => setImmediate(r));
  ok(calls.filter(c => c.url === '/api/settings/agent/keys' && c.method === 'POST').length >= 2,
     'Enter in the key input adds the key');

  click('settings-close');
  eq(registry['settings-modal'].style.display, 'none', 'close button hides the modal');

  // ── the reported bug: a plain-text 404 must become a readable message ──
  console.log('--- non-JSON response ---');
  textMode = true;
  registry['set-msg'].textContent = '';
  let threw = null;
  try { await sandbox.loadSettings(); } catch (e) { threw = e; }
  textMode = false;
  ok(threw === null, 'loadSettings never throws on a non-JSON body');
  const msg = texts(registry['set-msg']);
  ok(msg.includes('HTTP 404'), 'error message carries the status: ' + JSON.stringify(msg));
  ok(msg.includes('The requested URL'), 'error message carries the body head');
  ok(!msg.includes('Unexpected token'), 'no JSON.parse jargon leaks to the user');

  // one user action on the radio must be one POST, even though the change
  // event was delivered twice (see the fire + dispatchEvent above)
  eq(calls.filter(c => /\/activate$/.test(c.url || '')).length, 1,
     'a key is activated exactly once per user action');

  // ── Release 6: fallback skip reasons, key health detail, System actions ──
  console.log('--- release 6: clarity + system ---');
  await sandbox.loadSettings();
  const fbRows = registry['set-fallback'].children;
  const gemRow = fbRows.find(r => texts(r).includes('gemini'));
  ok(gemRow && texts(gemRow).includes('no keys'),
     'a provider with no keys says so in its fallback row');
  const oaiRow = fbRows.find(r => texts(r).includes('openai'));
  eq(texts(oaiRow).trim(), '1. openai ↑ ↓',
     'a healthy provider row carries no skip tag');
  sandbox.settingsState.data.config.providers.gemini.enabled = false;
  sandbox.renderSettings();
  const gemRow2 = registry['set-fallback'].children.find(r => texts(r).includes('gemini'));
  ok(gemRow2 && texts(gemRow2).includes('off'),
     'a disabled provider is marked off in its fallback row');
  sandbox.settingsState.data.config.providers.gemini.enabled = true;

  // key rows: usage counts, cooldown countdown, error tooltip
  sandbox.settingsState.provider = 'openai';
  const k2 = sandbox.settingsState.data.config.keys.openai[1];
  k2.status = 'rate_limited';
  k2.cooldown_until = Date.now() / 1000 + 600;
  k2.last_error = '429 too many requests';
  k2.used = 7;
  sandbox.renderSettings();
  const keyRows6 = registry['set-key-list'].children;
  ok(texts(keyRows6[0]).includes('used 4'), 'key row shows its usage count');
  ok(texts(keyRows6[1]).includes('used 7'), 'parked key shows its usage count');
  ok(texts(keyRows6[1]).includes('⏳'), 'parked key shows its cooldown countdown');
  const sx1 = keyRows6[1].children[0].children[3];
  eq(sx1 && sx1.title, '429 too many requests',
     'the last error is one hover away on the status badge');
  k2.status = 'ok'; k2.cooldown_until = 0; k2.last_error = null; k2.used = 0;
  sandbox.renderSettings();

  // ── Release 8.1: an expired cooldown reads as ready everywhere ──
  // The backend would use this key on the next turn (status flips to 'ok'
  // only after that call), so neither the badge nor the fallback row may
  // keep calling it parked.
  k2.status = 'rate_limited';
  k2.cooldown_until = Date.now() / 1000 - 60;
  k2.last_error = '429 too many requests';
  sandbox.renderSettings();
  const keyRows81 = registry['set-key-list'].children;
  ok(texts(keyRows81[1]).includes('ready'),
     'expired-cooldown key badges ready');
  ok(!texts(keyRows81[1]).includes('429 cooldown'),
     'expired-cooldown key drops the parked label');
  const oaiRow81 = registry['set-fallback'].children.find(r => texts(r).includes('openai'));
  eq(texts(oaiRow81).trim(), '1. openai ↑ ↓',
     'fallback row treats an expired cooldown as healthy');
  // while every key is still cooling, the row says so plainly
  const k1 = sandbox.settingsState.data.config.keys.openai[0];
  k1.status = 'rate_limited'; k1.cooldown_until = Date.now() / 1000 + 600;
  k2.cooldown_until = Date.now() / 1000 + 600;
  sandbox.renderSettings();
  const oaiRow81b = registry['set-fallback'].children.find(r => texts(r).includes('openai'));
  ok(texts(oaiRow81b).includes('⏳ cooling'),
     'fallback row says cooling while all keys cool down');
  k1.status = 'ok'; k1.cooldown_until = 0;
  k2.status = 'ok'; k2.cooldown_until = 0; k2.last_error = null;
  sandbox.renderSettings();

  // doctor — one tap, plain-language lines
  registry['set-doctor'].fire('click');
  await new Promise(r => setImmediate(r));
  await new Promise(r => setImmediate(r));
  const doc = texts(registry['set-doctor-out']);
  ok(doc.includes('Local model: model.litertlm'), 'doctor reports the local model');
  ok(doc.includes('openai: key 2/2 healthy'), 'doctor reports per-provider key health');
  ok(doc.includes('Storage free: 5.0 GB'), 'doctor reports free storage');
  ok(doc.includes('Usage:'), 'doctor reports the storage breakdown');
  ok(doc.includes('5.0 MB'), 'storage sizes are human readable');
  ok(doc.includes('Recent log'), 'doctor shows the recent log');
  ok(doc.includes('demo line'), 'log lines render');
  ok(doc.includes('opencode'), 'doctor reports the code backend');

  // backup — the zip name comes back, no key material
  registry['set-backup'].fire('click');
  await new Promise(r => setImmediate(r));
  await new Promise(r => setImmediate(r));
  ok(texts(registry['set-msg']).includes('ai-agent-backup-test.zip'),
     'backup reports the created file name');

  // reset — one POST, then settings reload and confirm in the panel
  registry['set-reset'].fire('click');
  await new Promise(r => setImmediate(r));
  await new Promise(r => setImmediate(r));
  await new Promise(r => setImmediate(r));
  ok(calls.some(c => c.url === '/api/settings/agent/reset' && c.method === 'POST'),
     'reset POSTs the reset endpoint');
  ok(texts(registry['set-msg']).includes('reset'), 'reset confirms in the panel');

  console.log('--- memory panel (Release 8) ---');
  registry['set-mem-view'].fire('click');
  await new Promise(r => setImmediate(r));
  await new Promise(r => setImmediate(r));
  ok(texts(registry['set-mem-out']).includes('likes tea'),
     'memory view shows the stored facts');
  registry['set-mem-add'].fire('click');  // primes the lazy input element
  await new Promise(r => setImmediate(r));
  registry['set-mem-input'].value = 'water the plants';
  registry['set-mem-add'].fire('click');
  await new Promise(r => setImmediate(r));
  await new Promise(r => setImmediate(r));
  ok(calls.some(c => c.url === '/api/remember' && c.method === 'POST'),
     'add posts to /api/remember');
  ok(texts(registry['set-mem-out']).includes('မှတ်ထားပြီးပါပြီ'),
     'add confirms in the panel');
  registry['set-mem-input'].value = 'tea';
  registry['set-mem-forget'].fire('click');
  await new Promise(r => setImmediate(r));
  await new Promise(r => setImmediate(r));
  ok(calls.some(c => c.url === '/api/memory/forget' && c.method === 'POST'),
     'forget posts to /api/memory/forget');
  ok(texts(registry['set-mem-out']).includes('2 ခု ဖျက်ပြီးပါပြီ'),
     'forget reports how many were removed');

  console.log('--- settings tabs (Release 7) ---');
  sandbox.showSettingsTab('provider');
  eq(registry['set-pane-backend'].style.display, 'none', 'tab: backend pane hides');
  eq(registry['set-pane-provider'].style.display, '', 'tab: provider pane shows');
  eq(registry['set-pane-system'].style.display, 'none', 'tab: system pane hides');
  sandbox.showSettingsTab('system');
  eq(registry['set-pane-system'].style.display, '', 'tab: system pane shows');
  eq(registry['set-pane-provider'].style.display, 'none', 'tab: provider pane hides again');
  sandbox.showSettingsTab('backend');
  eq(registry['set-pane-backend'].style.display, '', 'tab: backend pane back');

  console.log('--- settings accordion (Release 9.2) ---');
  ok(html.includes('class="set-sec-head"') && html.includes('data-tab="provider"'),
     'R9.2: sections are stacked heading bars, not a tab row');
  ok(!html.includes('id="set-tabs"'), 'R9.2: the old tab row is gone');
  eq(registry['set-arrow-backend'].textContent, '▾', 'R9.2: the open section shows ▾');
  eq(registry['set-arrow-provider'].textContent, '▸', 'R9.2: a closed section shows ▸');
  sandbox.toggleSettingsSection('backend');
  eq(registry['set-pane-backend'].style.display, 'none',
     'R9.2: tapping the open heading closes it');
  eq(registry['set-arrow-backend'].textContent, '▸', 'R9.2: its arrow closes too');
  sandbox.toggleSettingsSection('system');
  eq(registry['set-pane-system'].style.display, '', 'R9.2: tapping a closed heading opens it');
  eq(registry['set-pane-provider'].style.display, 'none',
     'R9.2: opening one section keeps the others closed');
  sandbox.showSettingsTab('backend');

  console.log('--- usage view (Release 13) ---');
  ok(html.includes('data-tab="usage"') && html.includes('id="set-pane-usage"'),
     'R13: Usage is a fourth accordion section');
  ok(html.includes('id="set-usage-list"') && html.includes('id="set-usage-total"'),
     'R13: the usage pane has a list and a totals line');
  ok(block.includes('renderUsage') && block.includes('usageCounts'),
     'R13: the settings block aggregates and renders usage');

  // the accordion treats usage like every other section
  sandbox.showSettingsTab('usage');
  eq(registry['set-pane-usage'].style.display, '', 'R13: usage pane opens');
  eq(registry['set-pane-backend'].style.display, 'none',
     'R13: opening usage closes the others');
  eq(registry['set-arrow-usage'].textContent, '▾', 'R13: the open usage section shows ▾');
  sandbox.toggleSettingsSection('usage');
  eq(registry['set-pane-usage'].style.display, 'none',
     'R13: tapping the usage heading closes it');
  sandbox.showSettingsTab('backend');

  // per-provider aggregation over config.keys, classified by keyUsable()
  const uk = sandbox.settingsState.data.config.keys;
  uk.openai[1].status = 'dead'; uk.openai[1].used = 7;
  uk.openai[1].last_error = '401 unauthorized';
  uk.gemini = [
    { id: 'g_1', label: '', key_masked: 'AIza…1111', status: 'rate_limited',
      cooldown_until: Date.now() / 1000 + 600, used: 2, last_error: '429 rate limited' },
    { id: 'g_2', label: '', key_masked: 'AIza…2222', status: 'rate_limited',
      cooldown_until: Date.now() / 1000 - 60, used: 3, last_error: '429 rate limited' },
  ];
  sandbox.renderSettings();
  const uRows = registry['set-usage-list'].children;
  eq(uRows.length, 5, 'R13: one usage row per provider');
  ok(texts(uRows[0]).includes('OpenAI — 2 keys · used 11'),
     'R13: used totals sum across keys: ' + texts(uRows[0]));
  ok(texts(uRows[0]).includes('ok 1') && texts(uRows[0]).includes('dead 1'),
     'R13: ok/dead counts split the bucket');
  ok(texts(uRows[0]).includes('401 unauthorized'),
     'R13: the last error is shown for the provider');
  ok(texts(uRows[1]).includes('used 5'), 'R13: gemini used sums too');
  ok(texts(uRows[1]).includes('ok 1') && texts(uRows[1]).includes('cooling 1'),
     'R13: an expired cooldown counts as ok, a live one as cooling');
  ok(texts(uRows[2]).includes('မထည့်ရသေးဘူး'),
     'R13: a provider with no keys says so');
  eq(registry['set-usage-total'].textContent, 'Total: 4 keys · used 16',
     'R13: the totals line sums keys and calls');
  ok(!texts(registry['set-usage-list']).includes(RAW_KEY),
     'R13: usage never renders raw key material');
  // restore the shared fixture for the blocks below
  uk.openai[1].status = 'ok'; uk.openai[1].used = 0; uk.openai[1].last_error = null;
  uk.gemini = [];
  sandbox.renderSettings();

  // ---- Release 9: Auto switch toggle + pin state ----
  sandbox.settingsState.data.config.auto_switch = false;
  sandbox.settingsState.data.config.pinned_provider = 'gemini';
  sandbox.renderSettings();
  eq(registry['set-auto-switch'].checked, false,
     'R9: the toggle reflects auto_switch off');
  ok(registry['set-pin-state'].textContent.includes('Lock'),
     'R9: the pin state line names the lock');
  ok(registry['set-pin-state'].textContent.includes('Gemini'),
     'R9: the pin state line names the pinned provider');
  eq(registry['set-pin-clear'].style.display, '',
     'R9: clear-pin shows while a pin is set');
  registry['set-auto-switch'].checked = true;
  const callsBeforeSave9 = calls.length;
  registry['set-save'].fire('click');
  await new Promise(r => setImmediate(r));
  await new Promise(r => setImmediate(r));
  const save9 = calls.slice(callsBeforeSave9)
    .find(c => c.url === '/api/settings/agent' && c.method === 'POST');
  ok(!!save9, 'R9: Save posts the settings');
  eq(save9 && save9.body.auto_switch, true, 'R9: Save carries the toggle state');
  const callsBeforeClear = calls.length;
  registry['set-pin-clear'].fire('click');
  await new Promise(r => setImmediate(r));
  await new Promise(r => setImmediate(r));
  const clearCall = calls.slice(callsBeforeClear)
    .find(c => c.url === '/api/settings/agent' && c.method === 'POST');
  ok(!!clearCall, 'R9: clear-pin posts');
  eq(clearCall && clearCall.body.pinned_provider, null,
     'R9: clear-pin posts a null pin');

  // ---- Release 9.1: the fold actually toggles (IIFE on a stub DOM) ----
  const iifeStart = html.indexOf("var head=document.getElementById('cmd-toggle')");
  ok(iifeStart > 0, 'R9.1: toggle wiring exists in the script');
  const iifeEnd = html.indexOf('})();', iifeStart);
  const iifeSrc = '(function(){' + html.slice(iifeStart, iifeEnd) + '})();';
  const foldEls = {
    'cmd-toggle': makeEl('h3'), 'cmd-list': makeEl('div'),
    'cmd-arrow': makeEl('span'),
  };
  foldEls['cmd-list'].style.display = 'none';
  const foldCtx = vm.createContext({
    document: { getElementById: id => foldEls[id] || null },
    localStorage: {
      _m: {},
      getItem(k) { return this._m[k] || null; },
      setItem(k, v) { this._m[k] = v; },
    },
  });
  vm.runInContext(iifeSrc, foldCtx);
  eq(foldEls['cmd-list'].style.display, 'none', 'R9.1: the fold starts closed');
  foldEls['cmd-toggle'].fire('click');
  eq(foldEls['cmd-list'].style.display, '', 'R9.1: tapping the heading opens the fold');
  eq(foldEls['cmd-arrow'].textContent, '▾', 'R9.1: the arrow follows the fold');
  foldEls['cmd-toggle'].fire('click');
  eq(foldEls['cmd-list'].style.display, 'none', 'R9.1: tapping again closes the fold');

  // every endpoint used must belong to routes/settings_api.py or
  // routes/memory_api.py (the Memory panel)
  const allowed = ['/api/settings/agent', '/api/settings/agent/keys',
                   '/api/settings/agent/keys/test',
                   '/api/settings/doctor', '/api/settings/backup',
                   '/api/memory', '/api/remember', '/api/memory/forget',
                   '/api/settings/agent/reset',
                   /^\/api\/settings\/agent\/keys\/[a-z]+\/[A-Za-z0-9_]+$/,
                   /^\/api\/settings\/agent\/keys\/[a-z]+\/[A-Za-z0-9_]+\/activate$/];
  const bad = calls.filter(c => !allowed.some(a =>
    typeof a === 'string' ? c.url === a : a.test(c.url)));
  eq(bad.map(c => c.url), [], 'only settings_api endpoints are called');

  console.log(failures ? `\n${failures} FAILURES` : '\nALL PASS');
  process.exit(failures ? 1 : 0);
})().catch(e => { console.error('FAIL exception: ' + (e && e.stack || e)); process.exit(1); });
