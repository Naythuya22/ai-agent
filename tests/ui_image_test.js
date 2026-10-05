// Headless smoke test for the Agent-mode image-upload helpers in web/index.html.
//   node tests/ui_image_test.js [path/to/index.html]
'use strict';
const fs = require('fs');

const html = fs.readFileSync(process.argv[2] || 'web/index.html', 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const main = scripts.find(s => s.includes('attachImage'));
if (!main) { console.error('FAIL: script with attachImage not found'); process.exit(1); }

const dStart = main.indexOf('// ── Voice input (Agent mode only)');
const dEnd = main.indexOf('var recog=null;', dStart);
if (dStart < 0 || dEnd < 0) { console.error('FAIL: decl bounds'); process.exit(1); }
const decls = main.slice(dStart, dEnd + 'var recog=null;'.length);

const iStart = main.indexOf('// ── Voice input — Web Speech API');
const sendStart = main.indexOf('async function sendStream');
const sendEnd = main.indexOf('async function runCreateSkill', sendStart);
if (iStart < 0 || sendStart < 0 || sendStart < iStart) { console.error('FAIL: impl bounds'); process.exit(1); }
// voice impl + everything up to sendStream, plus sendStream itself (payload check)
const impl = main.slice(iStart, sendStart) + main.slice(sendStart, sendEnd < 0 ? undefined : sendEnd);
const sendSrc = impl.slice(impl.indexOf('async function sendStream'));

let failures = 0;
function eq(got, want, msg) {
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g !== w) { failures++; console.error(`FAIL ${msg}\n  got  ${g}\n  want ${w}`); }
  else console.log(`ok   ${msg}`);
}
function ok(c, m) { if (!c) { failures++; console.error('FAIL ' + m); } else console.log('ok   ' + m); }

function el() {
  const classes = new Set();
  const node = {
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
  // Real DOM: setting `img.src` reflects into the attribute.
  Object.defineProperty(node, 'src', {
    get() { return 'src' in this.attrs ? this.attrs.src : ''; },
    set(v) { this.attrs.src = v; },
    configurable: true,
  });
  return node;
}

function FormDataStub() { this.fields = []; }
FormDataStub.prototype.append = function (n, v, f) { this.fields.push({ name: n, value: v, filename: f }); };
FormDataStub.prototype.get = function (n) {
  const f = this.fields.find((x) => x.name === n);
  return f ? f.value : null;
};

function build(opts) {
  opts = opts || {};
  const cache = {};
  const docListeners = {};
  const documentStub = {
    getElementById: (id) => (cache[id] || (cache[id] = el())),
    addEventListener: (ev, fn) => { docListeners[ev] = fn; },
  };
  const state = { mode: opts.mode || 'chat', busy: false, abort: null, listening: false };
  const inputEl = el();
  const status = { text: '' };
  const setStatus = (t) => { status.text = t || ''; };
  const calls = [];
  const fetchStub = function (url, init) {
    calls.push({ url, init });
    // fetchResult: object, array of objects (per call), or function.
    let r = opts.fetchResult;
    if (Array.isArray(r)) r = r[Math.min(calls.length - 1, r.length - 1)];
    else if (typeof r === 'function') r = r(url, init, calls.length - 1);
    if (!r) r = { ok: true, status: 200, body: { id: 'a1b2.png', name: 'pic.png', size: 1024, vision: true } };
    return Promise.resolve({
      ok: r.ok, status: r.status,
      json: () => Promise.resolve(r.body || {}),
      text: () => Promise.stringify(r.body || {}),
    });
  };
  const api = new Function(
    'document', 'window', 'state', 'inputEl', 'setStatus', 'fetch', 'FileReader',
    'FormData', 'AbortController', 'appendMsg', 'setBusy', 'appendTyping',
    'setActivity', 'killTyping', 'clearActivity', 'activityFor',
    'renderMarkdown', 'scrollBottom', 'updateCtx',
    decls + '\n' + impl + `
    return { syncAgentTools, attachImage, renderImageChip, clearImage, attachFile,
             sendStream, fileMessageFor,
             get pendingImage() { return pendingImage; },
             get IMAGE_MAX_BYTES() { return IMAGE_MAX_BYTES; } };`
  )(documentStub, {}, state, inputEl, setStatus, fetchStub, function FR() {},
    FormDataStub, function () { this.abort = function () {}; },
    function () {}, function () {},
    function () { return { remove() {} }; },
    function () {}, function () {}, function () {},
    function () {}, function () {}, function () {}, function () {});
  return { api, els: cache, state, status, calls, docListeners };
}

const pngFile = { name: 'pic.png', type: 'image/png', size: 1024 };

(async () => {
  // ── Markup ─────────────────────────────────────────────────────────
  ok(/id="image-btn"/.test(html), 'image button markup present');
  ok(/id="image-input"[^>]*accept="image\/png,image\/jpeg,image\/webp,image\/gif"/.test(html),
     'image input accepts only png/jpeg/webp/gif');
  ok(/id="image-chip"/.test(html) && /id="image-label"/.test(html), 'image chip markup present');

  // ── Agent-only visibility ──────────────────────────────────────────
  {
    const t = build();
    t.api.syncAgentTools();
    eq(t.els['add-btn'].hidden, true, '“+” hidden in chat mode');
    eq(t.els['image-btn'].hidden, true, 'image button hidden in chat mode');
    eq(t.els['image-chip'].hidden, true, 'image chip hidden with nothing attached');
    t.state.mode = 'agent';
    t.api.syncAgentTools();
    eq(t.els['add-btn'].hidden, false, '“+” shown in agent mode');
    eq(t.els['image-btn'].hidden, true, 'image button collapsed by default in agent mode');
    t.els['add-btn'].click();
    eq(t.els['image-btn'].hidden, false, 'pressing “+” reveals the image button');
    eq(t.els['image-btn'].classes.has('tool-pop'), true, 'pop-in animation applied');
    t.els['add-btn'].click();
    eq(t.els['image-btn'].hidden, true, 'pressing “+” hides it again');
    t.els['add-btn'].click();
    await t.api.attachImage(pngFile);
    ok(t.api.pendingImage, 'valid png attaches');
    t.api.syncAgentTools();
    eq(t.els['image-chip'].hidden, false, 'image chip visible in agent mode');
    t.state.mode = 'chat';
    t.api.syncAgentTools();
    eq(t.els['image-chip'].hidden, true, 'image chip hidden in chat mode');
    t.state.mode = 'agent';
    t.api.syncAgentTools();
    t.els['image-remove'].click();
    eq(t.api.pendingImage, null, '× clears the image');
    eq(t.els['image-chip'].hidden, true, 'chip hides after removal');
  }

  // ── Client-side guards ─────────────────────────────────────────────
  {
    const t = build({ mode: 'agent' });
    await t.api.attachImage({ name: 'doc.pdf', type: 'application/pdf', size: 10 });
    eq(t.api.pendingImage, null, 'non-image mime rejected client-side');
    ok(t.status.text.indexOf('unsupported image type') >= 0, 'bad mime reported in the status line');
    eq(t.calls.length, 0, 'bad mime never hits the upload endpoint');
  }
  {
    const t = build({ mode: 'agent' });
    await t.api.attachImage({ name: 'huge.png', type: 'image/png', size: 5 * 1024 * 1024 + 1 });
    eq(t.api.pendingImage, null, '5MB+1 rejected client-side');
    ok(t.status.text.indexOf('5MB') >= 0, 'oversize reported in the status line');
    eq(t.calls.length, 0, 'oversize never hits the upload endpoint');
  }
  eq(build({ mode: 'agent' }).api.IMAGE_MAX_BYTES, 5242880, 'IMAGE_MAX_BYTES is exactly 5MB');

  // ── Upload request + chip ──────────────────────────────────────────
  {
    const t = build({ mode: 'agent' });
    await t.api.attachImage(pngFile);
    eq(t.calls.length, 1, 'one upload request issued');
    eq(t.calls[0].url, '/api/agent/upload', 'POSTs to /api/agent/upload');
    eq(t.calls[0].init.method, 'POST', 'upload uses POST');
    const sent = t.calls[0].init.body;
    ok(sent && typeof sent.get === 'function', 'body is FormData');
    eq(sent.get('file'), pngFile, 'FormData carries the file');
    eq(t.api.pendingImage.id, 'a1b2.png', 'pendingImage keeps the returned id');
    eq(t.els['image-label'].textContent, 'pic.png · 1.0 KB',
       'chip reads "filename · size"');
    eq(t.status.text.indexOf('image attached') >= 0, true, 'status confirms the attachment');
  }

  // ── Server rejection surfaces reason + OCR fallback (req 6) ────────
  {
    const t = build({
      mode: 'agent',
      fetchResult: {
        ok: false, status: 400,
        body: {
          error: 'vision unavailable — model reports vision=False (text-only model)',
          suggestion: 'OCR အစားထိုးနည်း: ပုံထဲက စာသားကို OCR tool ဖြင့် ဖတ်ပြီး ကူးထည့်ပါ',
        },
      },
    });
    await t.api.attachImage(pngFile);
    eq(t.api.pendingImage, null, '400 leaves no pending image');
    ok(t.status.text.indexOf('vision unavailable') >= 0, 'reason shown in status');
    ok(t.status.text.indexOf('OCR') >= 0, 'OCR fallback suggested in status');
  }

  // ── Send path: image_id only in agent mode ─────────────────────────
  ok(/payload\.image_id=pendingImage\.id/.test(sendSrc), 'sendStream forwards image_id');
  ok(/if\(mode==='agent'&&pendingImage\)/.test(sendSrc), 'image_id is agent-mode only');
  ok(/clearImage\(\)/.test(sendSrc), 'sendStream clears the image chip');
  ok(/body:JSON\.stringify\(payload\)/.test(sendSrc), 'sendStream posts the payload');

  // ── Live payload: message + image_id must both travel ──────────────
  const streamCall = (t) => t.calls.find((c) => /\/stream$/.test(c.url));
  // call #1 = image upload (must succeed), call #2 = the turn (must fail)
  const uploadThenFail = [
    { ok: true, status: 200, body: { id: 'a1b2.png', name: 'pic.png', size: 1024, vision: true } },
    { ok: false, status: 500, body: { error: 'boom' } },
  ];
  {
    const t = build({ mode: 'agent', fetchResult: uploadThenFail });
    await t.api.attachImage(pngFile);
    await t.api.sendStream('hello agent', 'agent');
    eq(t.calls.length, 2, 'upload + send = two requests');
    eq(t.calls[1].url, '/api/agent/stream', 'agent turn goes to the agent stream');
    const body = JSON.parse(t.calls[1].init.body);
    eq(Object.prototype.hasOwnProperty.call(body, 'message'), true, 'payload has message');
    eq(body.message, 'hello agent', 'payload message is the typed text');
    eq(body.image_id, 'a1b2.png', 'payload carries image_id');
    eq(t.api.pendingImage, null, 'image cleared after send');
  }
  {
    // An image staged in agent mode must never leak into a chat turn.
    const t = build({
      mode: 'agent',
      fetchResult: uploadThenFail,
    });
    await t.api.attachImage(pngFile);
    t.state.mode = 'chat';
    await t.api.sendStream('plain chat text', 'chat');
    const call = streamCall(t);
    eq(call.url, '/api/chat/stream', 'chat turn still hits the chat stream');
    const body = JSON.parse(call.init.body);
    eq(Object.prototype.hasOwnProperty.call(body, 'image_id'), false,
       'chat payload carries no image_id');
    eq(body.message, 'plain chat text', 'chat message is untouched');
    eq(t.api.pendingImage.id, 'a1b2.png', 'image survives a chat send');
  }
  {
    // Image + typed text together: message and image_id side by side.
    const t = build({
      mode: 'agent',
      fetchResult: uploadThenFail,
    });
    await t.api.attachImage(pngFile);
    await t.api.sendStream('hi', 'agent');
    const body = JSON.parse(streamCall(t).init.body);
    eq(body.message, 'hi', 'message present alongside image_id');
    eq(body.image_id, 'a1b2.png', 'image_id present alongside message');
  }

  // ── ChatGPT-style preview thumbnail + full-screen viewer ───────────
  ok(/id="image-thumb"/.test(html), 'image thumbnail element present');
  ok(/id="viewer"/.test(html) && /id="viewer-img"/.test(html), 'full-screen viewer markup present');
  ok(/id="viewer-close"/.test(html) && /id="viewer-pre"/.test(html), 'viewer has close button + text pane');

  {
    const blob = new Blob([new Uint8Array(1024)], { type: 'image/png' });
    blob.name = 'shot.png';
    const t = build({ mode: 'agent' });
    await t.api.attachImage(blob);
    ok(t.api.pendingImage, 'Blob-based image attaches');
    const thumb = t.els['image-thumb'];
    eq(thumb.hidden, false, 'thumbnail visible after selecting an image');
    ok(String(thumb.getAttribute('src') || '').indexOf('blob:') === 0,
       'thumbnail uses an object URL');
    eq(t.els['image-chip'].hidden, false, 'chip shows the thumbnail');
    thumb.click();
    eq(t.els['viewer'].hidden, false, 'clicking the thumbnail opens the viewer');
    eq(t.els['viewer-img'].hidden, false, 'viewer shows the image');
    eq(t.els['viewer-img'].getAttribute('src'), thumb.getAttribute('src'),
       'viewer shows the same image');
    t.els['viewer-close'].click();
    eq(t.els['viewer'].hidden, true, 'close button hides the viewer');
    thumb.click();
    t.docListeners.keydown({ key: 'Escape' });
    eq(t.els['viewer'].hidden, true, 'Escape closes the viewer');
    thumb.click();
    t.els['viewer'].click({ target: t.els['viewer'] });
    eq(t.els['viewer'].hidden, true, 'clicking the backdrop closes the viewer');
    t.els['image-remove'].click();
    eq(t.els['image-thumb'].hidden, true, 'removing the image hides the thumbnail');
    ok(!t.els['image-thumb'].getAttribute('src'), 'object URL dropped with the image');
    eq(t.els['image-icon'].hidden, false, '🖼️ icon returns as the fallback');
  }

  // ── Send button prominence (Agent only) ────────────────────────────
  {
    const t = build({ mode: 'chat' });
    t.api.syncAgentTools();
    eq(t.els['send-btn'].classes.has('send-accent'), false, 'send button plain in chat mode');
    t.state.mode = 'agent';
    t.api.syncAgentTools();
    eq(t.els['send-btn'].classes.has('send-accent'), true, 'send button prominent in agent mode');
    t.state.mode = 'code';
    t.api.syncAgentTools();
    eq(t.els['send-btn'].classes.has('send-accent'), false, 'send button plain in code mode');
  }

  console.log(failures ? `\n${failures} FAILED` : '\nALL PASS');
  process.exit(failures ? 1 : 0);
})().catch((e) => { console.error('FAIL (threw): ' + e.stack); process.exit(1); });
