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
ok(!main.includes('/api/settings/model'),
   'the indicator no longer reads the settings config (the old static source)');

console.log('--- existing features ---');
['id="voice-btn"', 'id="attach-btn"', 'id="image-btn"', 'id="send-btn"',
 'data-mode="chat"', 'data-mode="agent"', 'id="settings-btn"', 'id="sel-bar"'
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
  return {
    id: id, textContent: '', className: 'model-info', title: '',
    style: {},
  };
}

const chip = makeEl('model-info');
let reply = { label: 'Model: deepseek-flash (Cloud)', backend: 'cloud', provider: 'deepseek' };
let failMode = null;           // 'throw' | 'html' | null
const calls = [];

const sandbox = {
  document: { getElementById(id) { return id === 'model-info' ? chip : makeEl(id); } },
  state: { mode: 'agent' },
  fetch: async function (url) {
    calls.push(url);
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

  // chat mode: no indicator at all (Chat Mode must stay untouched)
  sandbox.state.mode = 'chat';
  chip.textContent = 'Model: Gemma 4 (Local, fallback)';
  await sandbox.updateModelInfo();
  eq(chip.textContent, '', 'no chip in Chat Mode');
  eq(calls.length, 3, 'no request made for Chat Mode');

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

  console.log(failures ? `\n${failures} FAILURES` : '\nALL PASS');
  process.exit(failures ? 1 : 0);
})().catch(e => { console.error('FAIL exception: ' + (e && e.stack || e)); process.exit(1); });
