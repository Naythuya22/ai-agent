// Headless test for sidebar session multi-select (bulk delete) in web/index.html.
//   node tests/ui_session_select_test.js [path/to/index.html]
'use strict';
const fs = require('fs');

const html = fs.readFileSync(process.argv[2] || 'web/index.html', 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const main = scripts.find(s => s.includes('renderSessions') && s.includes('deleteSelected'));
if (!main) { console.error('FAIL: script with renderSessions/deleteSelected not found'); process.exit(1); }

const block = main.slice(main.indexOf('// ── Session multi-select'),
                          main.indexOf('async function activateSession'));
if (!block.includes('function deleteSelected') || !block.includes('function renderSessions')) {
  console.error('FAIL: multi-select block bounds wrong');
  process.exit(1);
}

let failures = 0;
function ok(c, m) { if (!c) { failures++; console.error('FAIL ' + m); } else console.log('ok   ' + m); }
function eq(got, want, msg) {
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g !== w) { failures++; console.error(`FAIL ${msg}\n  got  ${g}\n  want ${w}`); }
  else console.log(`ok   ${msg}`);
}

function classListOf(node) {
  const get = () => node.className.split(/\s+/).filter(Boolean);
  const set = a => { node.className = a.join(' '); };
  return {
    add(c) { const a = get(); if (a.indexOf(c) < 0) { a.push(c); set(a); } },
    remove(c) { set(get().filter(x => x !== c)); },
    contains(c) { return get().indexOf(c) >= 0; },
    toggle(c, force) {
      const has = this.contains(c);
      const on = force === undefined ? !has : !!force;
      if (on && !has) this.add(c);
      if (!on && has) this.remove(c);
      return on;
    },
  };
}

function el(tag) {
  const node = {
    tagName: tag || 'div', className: '', type: '', textContent: '', checked: false,
    hidden: false, disabled: false, value: '', attrs: {}, listeners: {}, children: [],
    appendChild(c) { this.children.push(c); c.parentNode = this; return c; },
    setAttribute(k, v) { this.attrs[k] = String(v); },
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    removeAttribute(k) { delete this.attrs[k]; },
    addEventListener(ev, fn) { this.listeners[ev] = fn; },
    removeEventListener(ev) { delete this.listeners[ev]; },
    click() { if (this.onclick) this.onclick({ stopPropagation() {} }); },
    focus() {},
  };
  let _html = '';
  Object.defineProperty(node, 'innerHTML', {
    get() { return _html; },
    set(v) { _html = v; if (v === '') node.children = []; },
    configurable: true,
  });
  node.classList = classListOf(node);
  return node;
}

function makeDocument(cache) {
  const docListeners = {};
  function matches(node, cls) { return node.className.split(/\s+/).indexOf(cls) >= 0; }
  function collect(root, cls, out) {
    (root.children || []).forEach(c => { if (matches(c, cls)) out.push(c); collect(c, cls, out); });
    return out;
  }
  return {
    doc: {
      getElementById: id => (cache[id] || (cache[id] = el('div'))),
      createElement: tag => el(tag),
      addEventListener: (ev, fn) => { docListeners[ev] = fn; },
      querySelectorAll(sel) {
        const root = cache['session-list'];
        if (!root) return [];
        return collect(root, sel.indexOf('.check') >= 0 ? 'check' : 'session-item', []);
      },
    },
    docListeners,
  };
}

function makeTimers() {
  const timers = [];
  return {
    set: (fn, ms) => { timers.push({ fn, ms, cancelled: false, done: false }); return timers.length; },
    clear: id => { if (timers[id - 1]) timers[id - 1].cancelled = true; },
    run: () => timers.forEach(t => { if (!t.cancelled && !t.done) { t.done = true; t.fn(); } }),
  };
}

// The block under test needs these names in scope; each build() supplies
// its own spies so tests stay independent.
function build(opts) {
  opts = opts || {};
  const cache = {};
  // pre-create the ids the app looks up, so a test can inspect any of them
  ['session-list', 'sel-bar', 'sel-count', 'sel-del', 'sel-all'].forEach(id => { cache[id] = el('div'); });
  const made = makeDocument(cache);
  const timers = makeTimers();
  const state = { mode: opts.mode || 'chat' };
  const chatEl = el('div');
  const inputEl = el('div');
  const status = { text: '' };
  const calls = [];
  const spies = { activate: [], history: 0, sessions: 0 };
  let confirmText = null;

  const api = new Function(
    'document', 'navigator', 'state', 'chatEl', 'inputEl', 'setStatus',
    'fetch', 'confirm', 'loadHistory', 'loadSessions', 'activateSession',
    'deleteSession', 'setTimeout', 'clearTimeout',
    block + `
    return { renderSessions, toggleSelect, selectAll, cancelSel, deleteSelected,
             setSelMode, syncSel, bmyNum, fireLongPress,
             get selMode(){return selMode}, get selIds(){return selIds},
             get renderedSessions(){return renderedSessions} };`
  )(
    made.doc, {}, state, chatEl, inputEl, t => { status.text = t || ''; },
    function (url, init) {
      calls.push({ url, init });
      const r = (typeof opts.fetchResult === 'function')
        ? opts.fetchResult(url, init, calls.length - 1) : (opts.fetchResult || {});
      return Promise.resolve({
        ok: r.ok !== false, status: r.status || 200,
        json: () => Promise.resolve(r.body || { ok: true, deleted: [], missing: [] }),
        text: () => JSON.stringify(r.body || {}),
      });
    },
    function (msg) { confirmText = msg; return opts.confirmAnswer !== false; },
    () => { spies.history++; return Promise.resolve(); },
    () => { spies.sessions++; return Promise.resolve(); },
    id => { spies.activate.push(id); },
    () => {}, timers.set, timers.clear
  );

  return { api, els: cache, state, status, calls, spies, chatEl, inputEl,
           runTimers: timers.run, docListeners: made.docListeners,
           get confirmText() { return confirmText; } };
}

const ROWS = [
  { id: 'chat-1', mode: 'chat', title: 'one', count: 2, active: true },
  { id: 'chat-2', mode: 'chat', title: 'two', count: 4, active: false },
  { id: 'chat-3', mode: 'chat', title: 'three', count: 0, active: false },
];

(async () => {
  // ── Markup + styles ────────────────────────────────────────────────
  ok(/id="sel-bar"/.test(html), 'selection bar markup present');
  ok(/id="sel-all"/.test(html) && /id="sel-del"/.test(html) && /id="sel-cancel"/.test(html),
     'Select All / Delete Selected / Cancel buttons present');
  ok(/#sel-bar\.on\{display:flex\}/.test(html), 'bar stays hidden until selection mode');
  ok(/\.session-item \.check\{/.test(html), 'per-session checkbox styled');
  ok(/document\.getElementById\('sel-all'\)\.addEventListener/.test(main),
     'bar buttons are wired up');

  // ── One checkbox per session ───────────────────────────────────────
  const t = build();
  t.api.renderSessions(ROWS);
  const list = t.els['session-list'];
  eq(list.children.length, 3, 'one row per session');
  eq(list.children.map(r => r.getAttribute('data-id')), ['chat-1', 'chat-2', 'chat-3'],
     'rows carry their session id');
  eq(list.children.map(r => r.children[0].className), ['check', 'check', 'check'],
     'checkbox is the first child of every row');
  eq(list.children.map(r => r.children[0].type), ['checkbox', 'checkbox', 'checkbox'],
     'first child is an input of type checkbox');
  eq(t.els['sel-bar'].classList.contains('on'), false, 'bar starts hidden');

  // ── Ticking a checkbox enters selection mode ───────────────────────
  list.children[0].children[0].click();
  eq(t.api.selMode, true, 'ticking a checkbox enters selection mode');
  eq(t.els['sel-bar'].classList.contains('on'), true, 'bar appears in selection mode');
  eq(t.els['sel-del'].disabled, false, 'Delete Selected enabled once something is chosen');
  eq(list.children[0].classList.contains('selected'), true, 'row is highlighted');
  eq(t.els['sel-count'].textContent, '၁ / ၃ ရွေးထားသည်', 'counter uses Burmese digits');
  eq(t.spies.activate.length, 0, 'ticking does not activate the session');

  // ── Row click inside selection mode toggles, never activates ───────
  list.children[1].onclick();
  eq(Object.keys(t.api.selIds).length, 2, 'row click selects a second session');
  eq(t.spies.activate.length, 0, 'no activation while selecting');
  list.children[1].onclick();
  eq(Object.keys(t.api.selIds).length, 1, 'row click again deselects');
  list.children[1].onclick();

  // ── Row click outside selection mode activates (unchanged) ─────────
  t.api.cancelSel();
  eq(t.api.selMode, false, 'Cancel leaves selection mode');
  eq(Object.keys(t.api.selIds).length, 0, 'Cancel clears the selection');
  list.children[2].onclick();
  eq(t.spies.activate, ['chat-3'], 'normal click still activates the session');

  // ── Select All / Deselect All ──────────────────────────────────────
  t.api.selectAll();
  eq(Object.keys(t.api.selIds).length, 3, 'Select All picks every listed session');
  eq(t.els['sel-all'].textContent, 'Deselect All', 'button flips to Deselect All');
  t.api.selectAll();
  eq(t.api.selMode, false, 'Deselect All leaves selection mode');

  // ── Long-press enters selection mode ───────────────────────────────
  const lp = build();
  lp.api.renderSessions(ROWS);
  const row = lp.els['session-list'].children[1];
  row.listeners.pointerdown({ target: row, clientX: 10, clientY: 10 });
  lp.runTimers();
  eq(lp.api.selMode, true, 'long-press enters selection mode');
  eq(Object.keys(lp.api.selIds), ['chat-2'], 'long-press selects the pressed session');
  row.onclick();   // the click that follows a long-press must not undo it
  eq(Object.keys(lp.api.selIds), ['chat-2'], 'the click after a long-press does not deselect');

  // ── Long-press cancelled by a 10px+ move (scroll) ──────────────────
  const mv = build();
  mv.api.renderSessions(ROWS);
  const row2 = mv.els['session-list'].children[0];
  row2.listeners.pointerdown({ target: row2, clientX: 0, clientY: 0 });
  row2.listeners.pointermove({ clientX: 40, clientY: 0 });
  mv.runTimers();
  eq(mv.api.selMode, false, 'moving the pointer cancels the long-press');

  // ── Long-press never starts on the delete button or checkbox ───────
  const skip = build();
  skip.api.renderSessions(ROWS);
  const skipRow = skip.els['session-list'].children[0];
  skipRow.listeners.pointerdown({ target: skipRow.children[3], clientX: 0, clientY: 0 });
  skip.runTimers();
  eq(skip.api.selMode, false, 'long-press on the × button does not enter selection');

  // ── Delete Selected: confirm text, payload, refresh ────────────────
  const del = build({
    fetchResult: {
      ok: true, status: 200,
      body: { ok: true, deleted: ['chat-1', 'chat-2', 'chat-3'], missing: [] },
    },
  });
  del.api.renderSessions(ROWS);
  del.api.selectAll();
  await del.api.deleteSelected();
  eq(del.confirmText, 'Session ၃ ခုကို ဖျက်မှာ သေချာလား?',
     'confirmation names the count in Burmese digits');
  eq(del.calls.length, 1, 'one request only (no per-id DELETE loop)');
  eq(del.calls[0].url, '/api/sessions/bulk_delete', 'bulk endpoint is used');
  eq(del.calls[0].init.method, 'POST', 'bulk delete is a POST');
  eq(JSON.parse(del.calls[0].init.body).ids, ['chat-1', 'chat-2', 'chat-3'],
     'payload carries every selected id');
  eq(del.api.selMode, false, 'selection mode closed after a successful delete');
  eq(del.spies.sessions >= 1, true, 'session list refreshed after delete');
  eq(del.spies.history, 1, 'transcript rebuilt because the active session went too');

  // ── Declining the confirmation must send nothing ───────────────────
  const no = build({ confirmAnswer: false });
  no.api.renderSessions(ROWS);
  no.api.selectAll();
  await no.api.deleteSelected();
  eq(no.calls.length, 0, 'no request when the user backs out');
  eq(no.api.selMode, true, 'selection kept when the user backs out');

  // ── Deleting a non-active selection leaves the transcript alone ────
  const other = build();
  other.state.mode = 'agent';
  other.api.renderSessions(ROWS);
  other.api.selectAll();
  await other.api.deleteSelected();
  eq(other.spies.history, 0, 'history not rebuilt when the active thread survives');
  eq(other.spies.sessions >= 1, true, 'list still refreshed');

  // ── Empty selection never fires a request ──────────────────────────
  const empty = build();
  empty.api.renderSessions(ROWS);
  await empty.api.deleteSelected();
  eq(empty.calls.length, 0, 'nothing sent with nothing selected');

  // ── Re-render keeps a selection only for rows still listed ─────────
  const keep = build();
  keep.api.renderSessions(ROWS);
  keep.api.selectAll();
  keep.api.renderSessions([ROWS[1]]);
  eq(Object.keys(keep.api.selIds), ['chat-2'], 'selection pruned to the visible rows');
  keep.api.renderSessions([]);
  eq(keep.api.selMode, false, 'selection mode exits when nothing is listed');

  // ── Helper ─────────────────────────────────────────────────────────
  eq(t.api.bmyNum(15), '၁၅', 'bmyNum renders Burmese digits');

  console.log(failures ? `\n${failures} FAILURE(S)` : '\nALL PASS');
  process.exit(failures ? 1 : 0);
})().catch(e => { console.error('FAIL exception: ' + (e && e.stack || e)); process.exit(1); });
