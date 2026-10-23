// Headless smoke test for the Agent-mode file-attach helpers in web/index.html.
//   node tests/ui_file_test.js [path/to/index.html]
'use strict';
const fs = require('fs');

const html = fs.readFileSync(process.argv[2] || 'web/index.html', 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const main = scripts.find(s => s.includes('attachFile'));
if (!main) { console.error('FAIL: script with attachFile not found'); process.exit(1); }

// Same slice shape as ui_voice_test.js: voice decls + everything up to sendStream
// (the file-upload block lives just before sendStream).
const dStart = main.indexOf('// ── Voice input (Agent mode only)');
const dEnd = main.indexOf('var recog=null;', dStart);
if (dStart < 0 || dEnd < 0) { console.error('FAIL: decl bounds'); process.exit(1); }
const decls = main.slice(dStart, dEnd + 'var recog=null;'.length);

const iStart = main.indexOf('// ── Voice input — Web Speech API');
const iEnd = main.indexOf('async function sendStream', iStart);
if (iStart < 0 || iEnd < 0 || iEnd < iStart) { console.error('FAIL: impl bounds'); process.exit(1); }
const impl = main.slice(iStart, iEnd);

// The send path must actually consume the attachment.
const sendStart = main.indexOf('async function sendStream');
const sendEnd = main.indexOf('async function runCreateSkill', sendStart);
const sendSrc = main.slice(sendStart, sendEnd < 0 ? undefined : sendEnd);

let failures = 0;
function eq(got, want, msg) {
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g !== w) { failures++; console.error(`FAIL ${msg}\n  got  ${g}\n  want ${w}`); }
  else console.log(`ok   ${msg}`);
}
function ok(c, m) { if (!c) { failures++; console.error('FAIL ' + m); } else console.log('ok   ' + m); }

function el() {
  const classes = new Set();
  return {
    hidden: false, disabled: false, value: '', files: null, attrs: {}, listeners: {},
    textContent: '',
    classList: {
      add(c) { classes.add(c); }, remove(c) { classes.delete(c); },
      toggle(c, on) { if (on === undefined) { classes.has(c) ? classes.delete(c) : classes.add(c); } else if (on) classes.add(c); else classes.delete(c); },
      contains(c) { return classes.has(c); },
    },
    setAttribute(k, v) { this.attrs[k] = v; },
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    removeAttribute(k) { delete this.attrs[k]; },
    addEventListener(ev, fn) { this.listeners[ev] = fn; },
    click(...args) { if (this.listeners.click) this.listeners.click(...args); },
    focus() {},
    classes,
  };
}

function build(opts) {
  opts = opts || {};
  const cache = {};
  const documentStub = { getElementById: (id) => (cache[id] || (cache[id] = el())) };
  const state = { mode: opts.mode || 'code', busy: false, abort: null, listening: false };
  const inputEl = el();
  const status = { text: '' };
  const setStatus = (t) => { status.text = t || ''; };
  const windowStub = { SpeechRecognition: function () {}, };
  const fetchStub = opts.fetch || (() => Promise.resolve({
    json: () => Promise.resolve({ used: 0, total: 8192 }),
  }));
  const FRS = opts.FileReader || function FileReaderStub() {};
  const api = new Function(
    'document', 'window', 'state', 'inputEl', 'setStatus', 'fetch', 'FileReader',
    decls + '\n' + impl + `
    return { syncAgentTools, startListening, stopListening, insertTranscript,
             attachFile, clearAttachment, buildFilePrompt, fileMessageFor,
             truncateToTokens, estTokens, renderAttachChip, attachBudget,
             get pendingFile() { return pendingFile; } };`
  )(documentStub, windowStub, state, inputEl, setStatus, fetchStub, FRS);
  return { api, els: cache, state, inputEl, status };
}

function readerFor(text) {
  const calls = [];
  function FR() {}
  FR.prototype.readAsText = function (f, enc) {
    calls.push({ file: f, encoding: enc });
    this.result = (f && typeof f.__text === 'string') ? f.__text : '';
    if (this.onload) this.onload();
  };
  FR.calls = calls;
  return FR;
}

// ── Markup: the chip exists next to the chat box ──────────────────────
ok(/id="attach-btn"/.test(html), 'attach button markup present');
ok(/id="file-input"[^>]*accept="\.txt,\.py,\.md,\.json"/.test(html) ||
   /accept="\.txt,\.py,\.md,\.json"[^>]*id="file-input"/.test(html),
   'file input only accepts .txt .py .md .json');
ok(/id="attach-chip"/.test(html) && /id="attach-label"/.test(html),
   'attachment chip markup present');

// ── Mode sync: visible in Agent only ──────────────────────────────────
{
  const t = build();
  t.api.syncAgentTools();
  eq(t.els['add-btn'].hidden, true, '“+” hidden in code mode');
  eq(t.els['attach-btn'].hidden, true, 'attach button hidden in code mode');
  t.state.mode = 'agent';
  t.api.syncAgentTools();
  eq(t.els['add-btn'].hidden, false, '“+” shown in agent mode');
  eq(t.els['attach-btn'].hidden, true, 'attach button collapsed by default in agent mode');
  t.els['add-btn'].click();
  eq(t.els['attach-btn'].hidden, false, 'pressing “+” reveals the attach button');
  eq(t.els['attach-btn'].classes.has('tool-pop'), true, 'pop-in animation class applied');
  eq(t.els['add-btn'].attrs['aria-expanded'], 'true', 'aria-expanded tracks the open state');
  t.els['add-btn'].click();
  eq(t.els['attach-btn'].hidden, true, 'pressing “+” again hides it');
  t.els['add-btn'].click();
  t.api.syncAgentTools();
  eq(t.els['attach-chip'].hidden, true, 'chip hidden while nothing is attached');
}

// ── Extension whitelist ───────────────────────────────────────────────
async function rejects(name, size, text, why) {
  const FR = readerFor(text);
  const t = build({ mode: 'agent', FileReader: FR });
  await t.api.attachFile({ name, size, __text: text });
  eq(t.api.pendingFile, null, name + ' → ' + why);
  ok(FR.calls.length === 0, name + ' → not even read');
  return t.status.text;
}
(async () => {
  ok((await rejects('payload.exe', 10, 'x', 'rejected')).indexOf('unsupported') >= 0,
     '.exe rejected with an explanation');
  await rejects('photo.png', 10, 'x', 'rejected');
  await rejects('notes', 10, 'x', 'rejected (no extension)');

  // ── Size limit: 100KB = 102400 bytes ────────────────────────────────
  {
    const FR = readerFor('ok');
    const t = build({ mode: 'agent', FileReader: FR });
    await t.api.attachFile({ name: 'limit.txt', size: 102400, __text: 'ok' });
    ok(t.api.pendingFile !== null, 'exactly 102400 bytes accepted');
    eq(FR.calls[0].encoding, 'UTF-8', 'readAsText uses UTF-8');
  }
  {
    const t = build({ mode: 'agent', FileReader: readerFor('x') });
    await t.api.attachFile({ name: 'big.txt', size: 102401, __text: 'x' });
    eq(t.api.pendingFile, null, '102401 bytes rejected');
    ok(t.status.text.indexOf('100KB') >= 0, 'oversize error names the limit');
  }

  // ── Chip: filename · size, and removal ──────────────────────────────
  {
    const t = build({ mode: 'agent', FileReader: readerFor('hello world') });
    await t.api.attachFile({ name: 'notes.py', size: 2048, __text: 'hello world' });
    eq(t.api.pendingFile.name, 'notes.py', 'pendingFile keeps the name');
    t.api.renderAttachChip();
    eq(t.els['attach-chip'].hidden, false, 'chip visible in agent mode');
    eq(t.els['attach-label'].textContent, 'notes.py · 2.0 KB',
       'chip reads "filename · size"');
    t.state.mode = 'code';
    t.api.syncAgentTools();
    eq(t.els['attach-chip'].hidden, true, 'chip hidden in code mode');
    ok(t.api.pendingFile !== null, 'attachment survives a mode switch');
    t.state.mode = 'agent';
    t.api.syncAgentTools();
    eq(t.els['attach-chip'].hidden, false, 'chip returns with agent mode');
    t.els['attach-remove'].click();
    eq(t.api.pendingFile, null, '× clears the attachment');
    eq(t.els['attach-chip'].hidden, true, 'chip hides after removal');
  }

  // ── Prompt format: [File: filename]\n(content)\n\n<user text> ────────
  {
    const t = build({ mode: 'agent', FileReader: readerFor('print(1)') });
    await t.api.attachFile({ name: 'app.py', size: 8, __text: 'print(1)' });
    eq(t.api.buildFilePrompt('run it'),
       '[File: app.py]\nprint(1)\n\nrun it',
       'prompt format is [File: name]\\n(content)\\n\\n<user text>');
  }

  // ── 8K context: token budget + truncation marker ────────────────────
  {
    const t = build({ mode: 'agent', FileReader: readerFor('b'.repeat(40000)) });
    await t.api.attachFile({ name: 'long.txt', size: 40000, __text: 'b'.repeat(40000) });
    const pf = t.api.pendingFile;
    ok(pf, 'oversized-in-tokens file still attaches (truncated)');
    eq(pf.omitted, 40000 - 12288, 'content cut to the 3072-token budget');
    eq(pf.content.length, 12288, 'truncated content length matches the budget');
    const prompt = t.api.buildFilePrompt('summarize');
    ok(prompt.indexOf('\n[truncated: 27712 chars omitted]\n\nsummarize') > 0,
       'truncation marker appended before the user text');
    ok(t.status.text.indexOf('truncated') >= 0, 'truncation reported in the status line');
    eq(t.els['attach-label'].textContent.indexOf('truncated') >= 0, true,
       'chip shows that the file was truncated');
  }
  {
    // Burmese-heavy text must cost more tokens per char (1.5 chars/token).
    eq(t_api_est(), true, 'estTokens mirrors core/llm.py (Burmese ~1.5 chars/token)');
  }

  // ── Context full → refuse instead of overflowing ────────────────────
  {
    const t = build({
      mode: 'agent',
      FileReader: readerFor('x'),
      fetch: () => Promise.resolve({ json: () => Promise.resolve({ used: 8000, total: 8192 }) }),
    });
    await t.api.attachFile({ name: 'late.txt', size: 1, __text: 'x' });
    eq(t.api.pendingFile, null, 'full context refuses the attachment');
    ok(t.status.text.indexOf('context full') >= 0, 'context refusal is explained');
  }
  {
    const t = build({
      mode: 'agent',
      FileReader: readerFor('fallback content'),
      fetch: () => Promise.reject(new Error('offline')),
    });
    await t.api.attachFile({ name: 'fb.md', size: 15, __text: 'fallback content' });
    ok(t.api.pendingFile, 'ctx probe failure falls back to the 8K default');
  }

  // ── Send path: file payload for agent, untouched for chat ───────────
  {
    const t = build({ mode: 'agent', FileReader: readerFor('secret') });
    await t.api.attachFile({ name: 's.txt', size: 6, __text: 'secret' });
    eq(t.api.fileMessageFor('hello', 'code'), 'hello', 'code mode never gets the file');
    ok(t.api.pendingFile, 'code send keeps the attachment for later');
    eq(t.api.fileMessageFor('hello', 'agent'),
       '[File: s.txt]\nsecret\n\nhello', 'agent send builds the file prompt');
    eq(t.api.pendingFile, null, 'agent send clears the chip');
    eq(t.els['attach-chip'].hidden, true, 'chip hidden right after send');
    eq(t.api.fileMessageFor('bare', 'agent'), 'bare', 'no attachment → plain text');
  }
  ok(/fileMessageFor\(text,mode\)/.test(sendSrc), 'sendStream calls fileMessageFor');
  ok(/var payload=\{message:message\}/.test(sendSrc),
     'sendStream builds the payload from the assembled message');
  ok(/body:JSON\.stringify\(payload\)/.test(sendSrc),
     'sendStream posts the assembled message');

  // ── Full-screen viewer for the attached text file ──────────────────
  ok(/id="attach-thumb"/.test(html), 'file preview tile present');
  ok(/id="viewer"/.test(html) && /id="viewer-pre"/.test(html), 'viewer markup present');
  {
    const t = build({ mode: 'agent', FileReader: readerFor('def hello(): pass') });
    await t.api.attachFile({ name: 'app.py', size: 18, __text: 'def hello(): pass' });
    ok(t.api.pendingFile, 'file attached for preview');
    t.els['attach-thumb'].click();
    eq(t.els['viewer'].hidden, false, 'clicking the file tile opens the viewer');
    eq(t.els['viewer-name'].textContent, 'app.py', 'viewer names the file');
    eq(t.els['viewer-pre'].textContent, 'def hello(): pass', 'viewer shows the file text');
    t.els['viewer-close'].click();
    eq(t.els['viewer'].hidden, true, 'viewer closes');
  }

  console.log(failures ? `\n${failures} FAILED` : '\nALL PASS');
  process.exit(failures ? 1 : 0);
})().catch((e) => { console.error('FAIL (threw): ' + e.stack); process.exit(1); });

function t_api_est() {
  const t = build();
  return t.api.estTokens('a'.repeat(4)) === 1 && t.api.estTokens('က'.repeat(15)) === 10;
}
