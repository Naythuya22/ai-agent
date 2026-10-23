// Headless test for the model indicator (status bar) in web/index.html.
//
//   node tests/ui_model_info_test.js [path/to/index.html]
//
// The indicator is a small isolated block in the main script, delimited by
// `// ── model indicator` / `// ── end model indicator`, so it can be sliced
// out and run against a stub DOM + a fake fetch. Nothing else in the page is
// executed — that is deliberate: this file must prove the chip updates
// without touching Chat Mode's behaviour.
'use strict';
const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync(process.argv[2] || 'web/index.html', 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const main = scripts.find(s => s.includes('renderSessions'));
if (!main) { console.error('FAIL: main script not found'); process.exit(1); }

const START = '// ── model indicator (status bar): the model that actually ran ──';
const END = '// ── end model indicator ──';
const a = main.indexOf(START), b = main.indexOf(END);
if (a < 0 || b < 0 || b < a) { console.error('FAIL: indicator block not found'); process.exit(1); }
const block = main.slice(a, b + END.length);

let failures = 0;
function ok(c, m) { if (!c) { failures++; console.error('FAIL ' + m); } else console.log('ok   ' + m); }
function eq(got, want, msg) {
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g !== w) { failures++; console.error(`FAIL ${msg}\n  got  ${g}\n  want ${w}`); }
  else console.log(`ok   ${msg}`);
}

// ─────────────────────── (a) markup + CSS ───────────────────────
console.log('--- markup ---');
ok(/<span id="model-info"/.test(html), '#model-info chip exists');
ok(/id="status"[^>]*>[\s\S]*?id="model-info"/.test(html), 'chip lives inside the status bar');
ok(html.includes('.model-info{'), '.model-info CSS rule present');
ok(html.includes('.model-info.cloud{'), 'cloud tint CSS rule present');
ok(html.includes('id="model-picker"'), 'quick provider picker sheet exists');
ok(html.includes('id="mp-list"'), 'picker list element exists');
ok(html.includes('.mp-row{'), 'picker row CSS present');
ok(html.includes('id="status-text"'), 'status text survived the edit');

console.log('--- hooks ---');
const hooks = (main.match(/updateModelInfo\(\)\}catch\(_mi\)\{\}/g) || []).length;
ok(hooks >= 4, 'updateModelInfo is called from ' + hooks + ' places (need >= 4)');
ok(main.includes("window.addEventListener('load', function(){try{updateModelInfo()}catch(_mi){}});"),
   'refreshes on first paint');
ok(main.includes("setStatus('mode: '+state.mode);try{updateModelInfo()}catch(_mi){}"),
   'refreshes on a mode switch');
ok(main.includes('await loadSettings();\n    try{updateModelInfo()}catch(_mi){}'),
   'refreshes after Settings are saved');
ok(main.includes("setStatus(evt.status||'ready');\n          try{updateModelInfo()}catch(_mi){}"),
   'refreshes after a streamed reply');
ok(main.includes('try{applyModelUsage(evt)}catch(_mu){}'),
   'the stream end event paints the chip without a refetch');
// Release 27: the live-discovery route /api/settings/models/<provider>
// shares the old path as a prefix, so this pin now matches the old
// endpoint's exact quoted path only — the chip still never reads it.
ok(!main.includes("'/api/settings/model'") &&
   !main.includes('"/api/settings/model"'),
   'the indicator no longer reads the settings config (the old static source)');

console.log('--- existing features ---');
['id="voice-btn"', 'id="attach-btn"', 'id="image-btn"', 'id="send-btn"',
 'data-mode="code"', 'data-mode="agent"', 'id="settings-btn"', 'id="sel-bar"'
].forEach(s => ok(html.includes(s), 'still present: ' + s));

// ───────────────────── (b) ES5 guard ─────────────────────
console.log('--- ES5 guard (indicator block) ---');
const code = block.split('\n').filter(l => !/^\s*\/\//.test(l)).join('\n');
ok(!code.includes('=>'), 'no arrow functions');
ok(!/\bconst\b/.test(code), 'no const');
ok(!/\blet\b/.test(code), 'no let');
ok(!code.includes('new Set('), 'no Set');
ok(!code.includes('`'), 'no template literals');
ok(code.includes("async function updateModelInfo()"), 'block defines updateModelInfo');

// ─────────────────── (c) behaviour on a stub DOM ────────────────
console.log('--- behaviour ---');

function makeEl(id) {
  const el = {
    id: id || '', textContent: '', className: 'model-info', title: '',
    style: {}, children: [], listeners: {}, hidden: false, type: '', _html: '',
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); },
    fire(ev) { (this.listeners[ev] || []).slice().forEach(fn => fn.call(this, { preventDefault() {} })); },
  };
  Object.defineProperty(el, 'innerHTML', {
    get() { return this._html; },
    set(v) { this._html = v; if (v === '') this.children.length = 0; },
  });
  return el;
}

const reg = {};
function regEl(id) { return reg[id] || (reg[id] = makeEl(id)); }
const chip = regEl('model-info');
let reply = { label: 'Model: deepseek-flash (Cloud)', backend: 'cloud', provider: 'deepseek' };
let failMode = null;           // 'throw' | 'html' | null
let settingsReply = null;      // fixture for GET /api/settings/agent
const postLog = [];            // bodies POSTed to /api/settings/agent
const calls = [];

const sandbox = {
  document: {
    getElementById(id) { return regEl(id); },
    createElement() { return makeEl(''); },
  },
  state: { mode: 'agent' },
  fetch: async function (url, opts) {
    calls.push(url);
    if (url === '/api/settings/agent') {
      if (opts && opts.method === 'POST') {
        postLog.push(JSON.parse(opts.body));
        return { status: 200, json: async () => ({ ok: true }) };
      }
      return { status: 200, json: async () => settingsReply };
    }
    if (failMode === 'throw') throw new Error('Network down');
    if (failMode === 'html') {
      return { status: 404, json: async () => { throw new SyntaxError('Unexpected token T'); } };
    }
    return { status: 200, json: async () => reply };
  },
  console,
  JSON, Math, Date, Array, Object, String, Number, Boolean,
  setTimeout: (fn) => { fn(); return 0; },
  clearTimeout() {},
};
vm.createContext(sandbox);
vm.runInContext(block, sandbox, { filename: 'indicator-block.js' });

(async () => {
  // cloud chip — reported by the status endpoint after a real cloud turn
  chip.textContent = '';
  reply = { ok: true, label: 'Model: deepseek-chat (Cloud, last used)',
            last_backend: 'cloud', provider: 'deepseek' };
  await sandbox.updateModelInfo();
  eq(calls[0], '/api/agent/status', 'asks the status endpoint, not the config');
  eq(chip.textContent, 'Model: deepseek-chat (Cloud, last used)', 'cloud label rendered');
  eq(chip.className, 'model-info cloud', 'chip tinted as cloud');
  ok(chip.title.includes('deepseek'), 'provider shown in the title: ' + chip.title);

  // local fallback chip
  reply = { ok: true, label: 'Model: Gemma 4 (Local, fallback)',
            last_backend: 'local', provider: '' };
  await sandbox.updateModelInfo();
  eq(chip.textContent, 'Model: Gemma 4 (Local, fallback)', 'local label rendered');
  eq(chip.className, 'model-info', 'chip back to the quiet colour');

  // no turn yet in this process: an honest blank, never a config-invented name
  reply = { ok: true, label: '', last_backend: '', provider: '' };
  await sandbox.updateModelInfo();
  eq(chip.textContent, '', 'no label means no model has run');

  // non-agent mode: no indicator at all (only Agent Mode has a backend)
  sandbox.state.mode = 'code';
  chip.textContent = 'Model: Gemma 4 (Local, fallback)';
  await sandbox.updateModelInfo();
  eq(chip.textContent, '', 'no chip in Code Mode');
  eq(calls.length, 3, 'no request made for Code Mode');

  // back to agent: an error answer must not blank the chip wrongly
  sandbox.state.mode = 'agent';
  reply = { ok: false, error: 'backend unavailable' };
  await sandbox.updateModelInfo();
  eq(chip.textContent, '', 'an error answer renders nothing');

  // the stream `end` event paints the chip without another round trip
  const before = calls.length;
  sandbox.applyModelUsage({ model_label: 'Model: Gemma 4 (Local, fallback)',
                            last_backend: 'local', provider: '' });
  eq(chip.textContent, 'Model: Gemma 4 (Local, fallback)', 'end event painted the chip');
  eq(chip.className, 'model-info', 'end-event local label is untinted');
  eq(calls.length, before, 'no fetch for the end event');
  sandbox.applyModelUsage({ type: 'end', text: 'hi' });
  eq(chip.textContent, 'Model: Gemma 4 (Local, fallback)',
     'events carrying no model label are ignored');

  // network failure / non-JSON body must not throw, and must not blank the chip
  reply = { ok: true, label: 'Model: x (Local)', last_backend: 'local' };
  await sandbox.updateModelInfo();
  eq(chip.textContent, 'Model: x (Local)', 'label established before the failure');
  failMode = 'throw';
  let threw = null;
  try { await sandbox.updateModelInfo(); } catch (e) { threw = e; }
  ok(threw === null, 'a fetch error does not escape the function');
  eq(chip.textContent, 'Model: x (Local)', 'previous label kept after a failure');

  failMode = 'html';
  threw = null;
  try { await sandbox.updateModelInfo(); } catch (e) { threw = e; }
  ok(threw === null, 'a non-JSON body does not escape the function');

  // ── quick provider picker (Release 7) ──
  failMode = null;
  reply = { ok: true, label: '', last_backend: '', provider: '' };
  // Mirrors the REAL GET /api/settings/agent shape (routes/settings_api.py):
  // per-provider config + keys live under `config`; the top level carries
  // the provider id list, labels and the effective fallback order. (The
  // first picker build read the top level directly and showed "no keys"
  // for everyone on a real device — this fixture pins the contract.)
  settingsReply = {
    ok: true,
    config: {
      backend_mode: 'cloud',
      providers: {
        openai: { model: 'gpt-4o-mini', enabled: true },
        gemini: { model: 'gemini-2.5-flash', enabled: true },
        deepseek: { model: 'deepseek-chat', enabled: true },
      },
      keys: {
        openai: [{ id: 'k1', status: 'ok' }],
        gemini: [{ id: 'k2', status: 'ok' }],
        deepseek: [],
      },
    },
    providers: ['openai', 'gemini', 'deepseek'],
    provider_labels: { openai: 'OpenAI', gemini: 'Gemini', deepseek: 'DeepSeek' },
    fallback_order: ['openai', 'gemini', 'deepseek'],
  };
  const picker = regEl('model-picker'), mpList = regEl('mp-list');
  chip.fire('click');
  await new Promise(r => setImmediate(r));
  await new Promise(r => setImmediate(r));
  eq(picker.hidden, false, 'tapping the chip opens the picker');
  eq(mpList.children.length, 3, 'picker lists every provider in fallback order');
  eq(mpList.children[0].children[0].textContent, 'OpenAI', 'row 1 is the first provider');
  eq(mpList.children[0].children[1].textContent, 'gpt-4o-mini', 'row 1 shows its configured model');
  eq(mpList.children[0].children[2].textContent, '✓ next', 'first usable provider is marked next');
  eq(mpList.children[2].children[2].textContent, 'no keys', 'keyless provider explains itself');
  ok(!(mpList.children[2].listeners.click || []).length, 'unusable row cannot be picked');
  mpList.children[1].fire('click');
  await new Promise(r => setImmediate(r));
  await new Promise(r => setImmediate(r));
  eq(postLog.length, 1, 'picking posts exactly once');
  eq(postLog[0].fallback, ['gemini', 'openai', 'deepseek'],
     'pick moves the provider to the front, order otherwise kept');
  eq(picker.hidden, true, 'picker closes after a pick');

  // ── Release 8.1: the picker health rule matches the backend ──
  // KeyManager.healthy(): usable = not dead AND cooldown expired. A
  // rate_limited key whose cooldown is over is usable again — its stored
  // status only flips back to 'ok' after a successful call, which may
  // never come if an earlier provider keeps answering first. (The first
  // picker demanded status==='ok' and greyed such providers out.)
  settingsReply.config.keys.openai = [{ id: 'k1', status: 'rate_limited', cooldown_until: Date.now() / 1000 - 60 }];
  settingsReply.config.keys.gemini = [{ id: 'k2', status: 'rate_limited', cooldown_until: Date.now() / 1000 + 600 }];
  settingsReply.config.keys.deepseek = [{ id: 'k3', status: 'dead' }];
  await sandbox.openModelPicker();
  eq(mpList.children.length, 3, 'picker reopens for the 8.1 health cases');
  eq(mpList.children[0].children[2].textContent, '✓ next',
     'expired-cooldown provider is usable again and marked next');
  eq(mpList.children[1].children[2].textContent, '⏳ cooling',
     'still-cooling provider says cooling, not no healthy key');
  ok(!(mpList.children[1].listeners.click || []).length, 'cooling provider cannot be picked');
  eq(mpList.children[2].children[2].textContent, 'key invalid',
     'all-dead provider says key invalid');
  ok(!(mpList.children[2].listeners.click || []).length, 'dead-key provider cannot be picked');
  // a usable provider that is not next stays pickable
  settingsReply.config.keys.openai = [{ id: 'k1', status: 'ok' }];
  settingsReply.config.keys.gemini = [{ id: 'k2', status: 'rate_limited', cooldown_until: Date.now() / 1000 - 60 }];
  await sandbox.openModelPicker();
  eq(mpList.children[1].children.length, 2, 'usable non-next row carries no block tag');
  mpList.children[1].fire('click');
  await new Promise(r => setImmediate(r));
  await new Promise(r => setImmediate(r));
  eq(postLog.length, 2, 'picking the expired-cooldown provider posts once more');
  eq(postLog[1].fallback, ['gemini', 'openai', 'deepseek'],
     'expired-cooldown pick moves it to the front');
  eq(picker.hidden, true, 'picker closes after the 8.1 pick');

  // ---- Release 9: locked picker (auto_switch off + a pin) ----
  settingsReply.config.auto_switch = false;
  settingsReply.config.pinned_provider = 'gemini';
  settingsReply.config.keys.openai = [{ id: 'k1', status: 'ok' }];
  settingsReply.config.keys.gemini = [{ id: 'k2', status: 'ok' }];
  settingsReply.config.keys.deepseek = [{ id: 'k3', status: 'dead', last_error: 'HTTP 402: insufficient credits' }];
  await sandbox.openModelPicker();
  eq(mpList.children[1].children[0].textContent, '● Gemini',
     'locked: the pinned row carries the dot');
  eq(mpList.children[1].children[2].textContent, '🔒 locked',
     'locked: the pinned row is tagged locked');
  eq(mpList.children[0].children.length, 2, 'locked: no ✓ next is offered');
  eq(mpList.children[2].children[2].textContent, 'no balance',
     'a key killed by a 402 says no balance, not key invalid');
  ok(!(mpList.children[1].listeners.click || []).length,
     'the pinned row itself is inert');
  ok(regEl('mp-note').textContent.includes('●'),
     'the locked note explains the dot');
  const postsBeforePin = postLog.length;
  mpList.children[0].fire('click');
  await new Promise(r => setImmediate(r));
  await new Promise(r => setImmediate(r));
  eq(postLog.length, postsBeforePin + 1, 'tapping another provider posts the pin move');
  eq(postLog[postLog.length - 1].pinned_provider, 'openai',
     'the tap posts pinned_provider');
  eq(postLog[postLog.length - 1].fallback, undefined,
     'a pin move never reorders the fallback chain');
  eq(picker.hidden, true, 'picker closes after pinning');

  sandbox.state.mode = 'code';
  const callsBefore = calls.length;
  await sandbox.openModelPicker();
  eq(calls.length, callsBefore, 'picker never opens in Code Mode');
  sandbox.state.mode = 'agent';

  // ── no-key banner (Release 25) ──
  // A fresh install can answer nothing: no provider has a usable key
  // and there is no local model. The banner must say so up front —
  // and must retire the moment either side can take a turn.
  ok(html.includes('id="nokey-banner"'), 'no-key banner markup exists');
  ok(html.includes('#nokey-banner.show{'), 'no-key banner CSS present');
  ok(html.includes('id="nokey-settings-btn"'), 'banner carries a Settings button');
  const nkBanner = regEl('nokey-banner');
  settingsReply = { ok: true, backend: { usable: [], local_model: '' } };
  await sandbox.checkNoKeyBanner();
  eq(nkBanner.className, 'show', 'banner shows when nothing can answer');
  settingsReply = { ok: true, backend: { usable: ['gemini'], local_model: '' } };
  await sandbox.checkNoKeyBanner();
  eq(nkBanner.className, '', 'banner hides once a provider is usable');
  settingsReply = { ok: true, backend: { usable: [], local_model: 'gemma-4-e4b' } };
  await sandbox.checkNoKeyBanner();
  eq(nkBanner.className, '', 'banner hides when a local model exists');
  settingsReply = { error: 'backend unavailable' };
  threw = null;
  try { await sandbox.checkNoKeyBanner(); } catch (e) { threw = e; }
  ok(threw === null, 'an error settings answer does not escape the banner check');
  eq(nkBanner.className, '', 'an error settings answer keeps the banner hidden');

  console.log(failures ? `\n${failures} FAILURES` : '\nALL PASS');
  process.exit(failures ? 1 : 0);
})().catch(e => { console.error('FAIL exception: ' + (e && e.stack || e)); process.exit(1); });
