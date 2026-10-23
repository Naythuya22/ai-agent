// Headless test for the Release 27 live model lists in web/index.html:
// (a) the Settings dropdown fed by /api/settings/models/<provider>
// (search filter, Free-only filter, non-chat tagging, missing warning,
// source line, refresh), and (b) the chat picker's per-provider model
// chips + ⚠️ mark. Both halves run the real script blocks against a
// stub DOM + a fake fetch, like the other ui_* suites.
//
//   node tests/ui_models_test.js [path/to/index.html]
'use strict';
const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync(process.argv[2] || 'web/index.html', 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const main = scripts.find(s => s.includes('renderSessions'));
if (!main) { console.error('FAIL: main script not found'); process.exit(1); }

let failures = 0;
function ok(c, m) { if (!c) { failures++; console.error('FAIL ' + m); } else console.log('ok   ' + m); }
function eq(got, want, msg) {
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g !== w) { failures++; console.error(`FAIL ${msg}\n  got  ${g}\n  want ${w}`); }
  else console.log(`ok   ${msg}`);
}

function makeEl(tag) {
  const node = {
    tagName: String(tag || 'div').toUpperCase(), children: [], listeners: {},
    style: {}, dataset: {}, className: '', textContent: '', value: '',
    checked: false, disabled: false, hidden: false, title: '', type: '',
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); },
    fire(ev, e) {
      const l = (this.listeners[ev] || []).slice();
      const evt = e || { target: this, preventDefault() {}, stopPropagation() {} };
      for (let i = 0; i < l.length; i++) l[i].call(this, evt);
    },
    focus() {},
    click() { this.fire('click'); },
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

// ═══════════════ (a) Settings: the live-fed dropdown ═══════════════
console.log('--- markup ---');
['set-model-search', 'set-model-free', 'set-model-free-wrap',
 'set-model-source', 'set-model-refresh', 'set-model-warn', 'set-model-list',
].forEach(id => ok(html.includes('id="' + id + '"'), 'element #' + id));

const S_START = '// ── Settings (hybrid backend';
const sIdx = main.indexOf(S_START);
if (sIdx < 0) { console.error('FAIL: settings block not found'); process.exit(1); }
const settingsBlock = main.slice(sIdx);

const registry = {};
const IDS = ['settings-modal', 'settings-close', 'settings-body', 'set-modes',
  'backend-mode-hint', 'set-local-note', 'set-net', 'set-provider', 'set-enabled',
  'set-fallback', 'set-model', 'set-model-custom', 'set-model-search',
  'set-model-free', 'set-model-free-wrap', 'set-model-source',
  'set-model-refresh', 'set-model-warn', 'set-model-list', 'set-base-url', 'set-cost',
  'set-ctx-limit', 'set-key-list', 'set-key-input', 'set-key-label',
  'set-key-add', 'set-msg', 'set-save', 'set-cancel', 'set-prompt-tier',
  'set-tier-auto', 'set-tier-cloud-strong', 'set-tier-cloud-fast',
  'set-tier-local', 'prompt-tier-hint', 'prompt-tier-state',
  'prompt-tier-warn', 'set-auto-switch', 'set-pin-state', 'set-pin-clear'];
IDS.forEach(id => { const e = makeEl('div'); e.id = id; registry[id] = e; });

const radios = ['auto', 'local', 'cloud'].map(v => {
  const e = makeEl('input'); e.type = 'radio'; e.value = v; e.checked = (v === 'auto'); return e;
});
const TIER_MODES = ['auto', 'cloud_strong', 'cloud_fast', 'local'];
const tierRadios = TIER_MODES.map(v => {
  const e = makeEl('input'); e.type = 'radio'; e.value = v; e.checked = (v === 'auto'); return e;
});

const REMOTE = {
  ok: true,
  providers: ['openai'],
  provider_labels: { openai: 'OpenAI' },
  modes: ['auto', 'local', 'cloud'],
  prompt_tier_modes: TIER_MODES,
  prompt_tier: { mode: 'auto', resolved: 'cloud_fast', source: 'auto',
                 model: 'gpt-gone-1', backend: 'cloud', auto: 'cloud_fast',
                 mismatch: false },
  fallback_order: ['openai'],
  local_fallback: true,
  costs: { openai: 0.15 },
  models: { openai: ['gpt-4o-mini'] },
  config: {
    backend_mode: 'auto', prefer_cloud: true, cost_ack: true,
    auto_switch: true, pinned_provider: null, fallback: ['openai'],
    providers: {
      openai: { model: 'gpt-gone-1', base_url: '', enabled: true },
    },
    keys: { openai: [{ id: 'k_1', label: 'main', key_masked: 'sk-l…3210',
                       status: 'ok', used: 4, active: true }] },
  },
  backend: { online: true, state: 'online', local_model: '',
             providers: { openai: { context_limit: 128000 } } },
};

const MODELS_PAYLOAD = {
  ok: true, provider: 'openai', source: 'live',
  fetched_at: '2026-10-23T13:00:00', stored_model: 'gpt-gone-1',
  stored_model_status: 'missing',
  models: [
    { id: 'gpt-4o', name: 'GPT-4o', context: 128000, kind: 'chat', free: false },
    { id: 'gpt-4o-mini', name: 'GPT-4o mini', context: 128000, kind: 'chat', free: true },
    { id: 'text-embedding-3-small', name: 'Text Embedding 3 Small', context: 8191, kind: 'embed', free: false },
    { id: 'whisper-1', name: 'Whisper', context: null, kind: 'audio', free: null },
  ],
};

const modelCalls = [];
const settingsPosts = [];
async function fetchStub(url, opts) {
  const o = opts || {};
  const jsonRes = obj => ({ status: 200, json: async () => obj, text: async () => JSON.stringify(obj) });
  if (url === '/api/settings/agent' && o.method === 'POST') {
    // A save lands in REMOTE, so the reload inside saveSettings reads
    // the new state back — the full sequence, not just the paint.
    const body = JSON.parse(o.body);
    settingsPosts.push(body);
    const saved = body.providers && body.providers.openai;
    if (saved && typeof saved.model === 'string') {
      REMOTE.config.providers.openai.model = saved.model;
      MODELS_PAYLOAD.stored_model = saved.model;
      MODELS_PAYLOAD.stored_model_status = 'ok';
    }
    return jsonRes({ ok: true, changed: ['providers.openai'],
                     config: REMOTE.config, backend: REMOTE.backend,
                     prompt_tier: REMOTE.prompt_tier });
  }
  if (url === '/api/settings/agent' && (!o.method || o.method === 'GET')) {
    return jsonRes(JSON.parse(JSON.stringify(REMOTE)));
  }
  if (url.startsWith('/api/settings/models/')) {
    modelCalls.push(url);
    return jsonRes(JSON.parse(JSON.stringify(MODELS_PAYLOAD)));
  }
  return jsonRes({ error: 'unexpected endpoint ' + url });
}

const settingsSandbox = {
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
  Event: function Event(type, opts) { this.type = type; this.bubbles = !!(opts && opts.bubbles); },
  JSON, Math, Date, Array, Object, String, Number, Boolean, encodeURIComponent,
  fmtTok(n) {
    if (!n) return '0';
    if (n >= 1000000) return (n / 1000000).toFixed(n % 1000000 ? 1 : 0) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(n % 1000 ? 1 : 0) + 'K';
    return '' + n;
  },
  setTimeout: (fn) => { fn(); return 0; },
  clearTimeout() {},
};
vm.createContext(settingsSandbox);
vm.runInContext(settingsBlock, settingsSandbox, { filename: 'settings-block.js' });

const tick = () => new Promise(r => setImmediate(r));

// ═══════════════ (b) Picker: model chips under each row ═══════════════
const I_START = '// ── model indicator (status bar): the model that actually ran ──';
const I_END = '// ── end model indicator ──';
const ia = main.indexOf(I_START), ib = main.indexOf(I_END);
if (ia < 0 || ib < 0 || ib < ia) { console.error('FAIL: indicator block not found'); process.exit(1); }
const indicatorBlock = main.slice(ia, ib + I_END.length);

const reg = {};
function regEl(id) { return reg[id] || (reg[id] = makeEl(id)); }
regEl('model-picker').hidden = true;
const mpList = regEl('mp-list');

const PICKER_SETTINGS = {
  ok: true,
  config: {
    backend_mode: 'cloud', auto_switch: true, pinned_provider: null,
    providers: {
      openai: { model: 'gpt-4o-mini', enabled: true },
      gemini: { model: 'gemini-gone', enabled: true },
      deepseek: { model: 'deepseek-chat', enabled: true },
    },
    keys: {
      openai: [{ id: 'k1', status: 'ok' }],
      gemini: [{ id: 'k2', status: 'ok' }],
      deepseek: [{ id: 'k3', status: 'ok' }],
    },
  },
  providers: ['openai', 'gemini', 'deepseek'],
  provider_labels: { openai: 'OpenAI', gemini: 'Gemini', deepseek: 'DeepSeek' },
  fallback_order: ['openai', 'gemini', 'deepseek'],
};
const PICKER_MODELS = {
  openai: { ok: true, provider: 'openai', source: 'live', fetched_at: 'x',
            stored_model: 'gpt-4o-mini', stored_model_status: 'ok',
            models: [
              { id: 'gpt-4o', name: 'GPT-4o', context: 128000, kind: 'chat', free: false },
              { id: 'gpt-4o-mini', name: 'GPT-4o mini', context: 128000, kind: 'chat', free: true },
              { id: 'text-embedding-3-small', name: 'Emb', context: 8191, kind: 'embed', free: false },
            ] },
  gemini: { ok: true, provider: 'gemini', source: 'live', fetched_at: 'x',
            stored_model: 'gemini-gone', stored_model_status: 'missing',
            models: [
              { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash', context: 1048576, kind: 'chat', free: null },
            ] },
};
const postLog = [];
const pickerSandbox = {
  document: {
    getElementById(id) { return regEl(id); },
    createElement() { return makeEl(''); },
  },
  state: { mode: 'agent' },
  fetch: async function (url, opts) {
    if (url === '/api/settings/agent') {
      if (opts && opts.method === 'POST') {
        postLog.push(JSON.parse(opts.body));
        return { status: 200, json: async () => ({ ok: true }) };
      }
      return { status: 200, json: async () => PICKER_SETTINGS };
    }
    const m = /^\/api\/settings\/models\/([a-z]+)$/.exec(url);
    if (m && PICKER_MODELS[m[1]]) {
      return { status: 200, json: async () => PICKER_MODELS[m[1]] };
    }
    return { status: 200, json: async () => ({ ok: true, label: '', last_backend: '', provider: '' }) };
  },
  console,
  JSON, Math, Date, Array, Object, String, Number, Boolean, encodeURIComponent,
  setTimeout: (fn) => { fn(); return 0; },
  clearTimeout() {},
};
vm.createContext(pickerSandbox);
vm.runInContext(indicatorBlock, pickerSandbox, { filename: 'indicator-block.js' });

(async () => {
  console.log('--- settings behaviour ---');
  settingsSandbox.openSettings();
  await tick(); await tick(); await tick();
  const sel = registry['set-model'];
  eq(sel.children.map(o => o.value),
     ['gpt-4o', 'gpt-4o-mini', 'gpt-gone-1', 'text-embedding-3-small', 'whisper-1'],
     'dropdown = chat models, then the stored (custom) value, then non-chat');
  const byVal = {};
  sel.children.forEach(o => { byVal[o.value] = o; });
  eq(byVal['text-embedding-3-small'].disabled, true, 'embed option is disabled');
  ok(byVal['text-embedding-3-small'].textContent.includes('chat မဟုတ်'),
     'embed option is tagged in Burmese: ' + byVal['text-embedding-3-small'].textContent);
  eq(byVal['whisper-1'].disabled, true, 'audio option is disabled');
  eq(byVal['gpt-gone-1'].disabled, false, 'the stored value itself stays selectable');
  ok(byVal['gpt-4o'].textContent.includes('ctx 128K'), 'context size is shown');
  ok(byVal['gpt-4o-mini'].textContent.includes('free'), 'free model is badged');
  ok(!byVal['gpt-4o'].textContent.includes('free'), 'paid model is not badged free');
  eq(registry['set-model-warn'].style.display, '', 'missing warning is visible');
  ok(registry['set-model-warn'].textContent.includes('gpt-gone-1'),
     'warning names the stored model');
  ok(registry['set-model-source'].textContent.includes('Live'),
     'source line says Live: ' + registry['set-model-source'].textContent);
  eq(registry['set-model-free-wrap'].style.display, '', 'Free filter appears when pricing is known');

  // search filter
  const search = registry['set-model-search'];
  search.value = 'embed'; search.fire('input');
  eq(sel.children.map(o => o.value), ['text-embedding-3-small'],
     'search narrows the dropdown by substring');
  search.value = ''; search.fire('input');
  eq(sel.children.length, 5, 'clearing the search restores the list');

  // free-only filter
  const freeBox = registry['set-model-free'];
  freeBox.checked = true; freeBox.fire('change');
  eq(sel.children.map(o => o.value), ['gpt-4o-mini', 'gpt-gone-1'],
     'Free-only keeps free chat models (and the stored value)');
  freeBox.checked = false; freeBox.fire('change');

  // refresh forces ?refresh=1
  const before = modelCalls.length;
  registry['set-model-refresh'].fire('click');
  await tick(); await tick();
  ok(modelCalls.length > before &&
     modelCalls[modelCalls.length - 1].includes('refresh=1'),
     'refresh button calls the endpoint with refresh=1');

  // Release 27.1 — the source line carries the model count, so "the
  // list arrived" is checkable at a glance (an empty arrival reads 0).
  ok(registry['set-model-source'].textContent.includes('4 ခု'),
     'source line shows the model count: ' +
     registry['set-model-source'].textContent);

  // Release 27.1 — tappable in-page rows mirror the dropdown: the
  // selection path that does not depend on the native select popup.
  console.log('--- settings model rows (27.1) ---');
  const listBox = registry['set-model-list'];
  eq(listBox.children.map(r => r.textContent),
     ['gpt-4o · ctx 128K', 'gpt-4o-mini · ctx 128K · free',
      'gpt-gone-1 (custom) ✓',
      'text-embedding-3-small · ctx 8.2K — chat မဟုတ် (embed)',
      'whisper-1 — chat မဟုတ် (audio)'],
     'rows = chat models, stored (custom) row, then tagged non-chat');
  const rowByText = {};
  listBox.children.forEach(r => { rowByText[r.textContent] = r; });
  ok(rowByText['gpt-gone-1 (custom) ✓'].className.includes('current'),
     'the stored row is marked current');
  ok(rowByText['text-embedding-3-small · ctx 8.2K — chat မဟုတ် (embed)']
       .className.includes('off'),
     'non-chat rows are marked off (not tappable)');
  rowByText['gpt-4o-mini · ctx 128K · free'].fire('click');
  eq(registry['set-model-custom'].value, 'gpt-4o-mini',
     'tapping a row fills the custom field (what Save persists)');
  eq(sel.value, 'gpt-4o-mini', 'tapping a row moves the select value');
  ok(rowByText['gpt-4o-mini · ctx 128K · free'].className.includes('current'),
     'the tapped row takes the current highlight');
  ok(!rowByText['gpt-gone-1 (custom) ✓'].className.includes('current'),
     'the stored row loses the highlight after a row tap');

  // The full save sequence: pick in the select, fire its real change
  // handler, click the real Save button, and read the state back after
  // saveSettings' reload — the pick must survive the whole round trip.
  console.log('--- settings save sequence (27.1) ---');
  sel.value = 'gpt-4o'; sel.fire('change');
  eq(registry['set-model-custom'].value, 'gpt-4o',
     'select change mirrors the pick into the custom field');
  registry['set-save'].fire('click');
  for (let t = 0; t < 10; t++) await tick();
  eq(settingsPosts.length, 1, 'Save posts the settings payload once');
  eq(settingsPosts[0].providers.openai.model, 'gpt-4o',
     'the POST carries the picked model');
  const byVal2 = {};
  sel.children.forEach(o => { byVal2[o.value] = o; });
  eq(byVal2['gpt-4o'].selected, true,
     'after the save reload the picked model is the selected option');
  eq(registry['set-model-warn'].style.display, 'none',
     'the missing warning clears once a listed model is saved');
  eq(registry['set-model-custom'].value, 'gpt-4o',
     'the custom field shows the saved model after reload');

  console.log('--- picker behaviour ---');
  await pickerSandbox.openModelPicker();
  await tick(); await tick(); await tick();
  eq(mpList.children.length, 3, 'picker still lists one row per provider');
  const openaiRow = mpList.children[0], geminiRow = mpList.children[1];
  const box = openaiRow.children.find(c => c.className === 'mp-models');
  ok(!!box, 'openai row gained a model chip box');
  eq(box.children.map(c => c.textContent),
     ['gpt-4o', 'gpt-4o-mini ✓ · free'],
     'chips are chat models only, stored one ticked, free badged');
  ok(texts(geminiRow).includes('⚠️'), 'missing stored model is marked ⚠️ in the picker');

  // tapping a chip for the "next" provider saves just the model
  postLog.length = 0;
  box.children[0].fire('click');
  await tick(); await tick();
  eq(postLog.length, 1, 'chip tap posts once for the already-next provider');
  eq(postLog[0], { providers: { openai: { model: 'gpt-4o' } } },
     'chip tap saves the provider model');

  console.log(failures ? `\n${failures} FAILURES` : '\nALL PASS');
  process.exit(failures ? 1 : 0);
})().catch(e => { console.error('FAIL exception: ' + (e && e.stack || e)); process.exit(1); });
