// Headless smoke test for the activity indicator (Thinking/Reading/Running)
// helpers embedded in web/index.html.
//   node act_test.js <path/to/index.html>
'use strict';
const fs = require('fs');

const html = fs.readFileSync(process.argv[2] || 'web/index.html', 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const main = scripts.find(s => s.includes('setActivity'));
if (!main) { console.error('FAIL: script with setActivity not found'); process.exit(1); }
const start = main.indexOf("var statusTextEl=document.getElementById('status-text');");
const end = main.indexOf("document.querySelectorAll('.mode-btn')", start);
if (start < 0 || end < 0 || end < start) { console.error('FAIL: slice bounds'); process.exit(1); }
const slice = main.slice(start, end);

let failures = 0;
function eq(got, want, msg) {
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g !== w) { failures++; console.error(`FAIL ${msg}\n  got  ${g}\n  want ${w}`); }
  else console.log(`ok   ${msg}`);
}
function ok(c, m) { if (!c) { failures++; console.error('FAIL ' + m); } else console.log('ok   ' + m); }

function el() {
  return { textContent: '', className: '', hidden: false, disabled: false, classList: { add() {}, remove() {} }, style: {} };
}
const statusTextEl = el();
const activityEl = el();
const actLabelEl = el();
const sendBtn = el();
const inputEl = el();
const documentStub = { getElementById: (id) => ({ 'status-text': statusTextEl, 'activity': activityEl, 'act-label': actLabelEl }[id] || el()) };
const state = { busy: false, abort: null };

const api = new Function(
  'document', 'state', 'sendBtn', 'inputEl',
  slice + `
  return { setStatus, clearActivity, setActivity, activityFor, setBusy, killTyping,
           setTyping: (e) => { typingEl = e; },
           setActPhase: (p) => { actPhase = p; } };`
)(documentStub, state, sendBtn, inputEl);

// setStatus writes to the span, not the whole bar (the pill lives inside #status)
api.setStatus('ready');
eq(statusTextEl.textContent, 'ready', 'setStatus targets #status-text');

// thinking → running → thinking cycle
api.setActivity('thinking', 'Thinking…');
eq(activityEl.hidden, false, 'pill shown');
eq(activityEl.className, 'activity thinking', 'thinking phase class');
eq(actLabelEl.textContent, 'Thinking…', 'thinking label');

api.setActivity('running', 'Running run_command');
eq(activityEl.className, 'activity running', 'running phase class');
eq(actLabelEl.textContent, 'Running run_command', 'running label');

api.setActivity('thinking', 'Thinking…');
eq(activityEl.className, 'activity thinking', 'back to thinking');

api.setActivity('reading', 'Reading config.py');
eq(activityEl.className, 'activity reading', 'reading phase class');

api.clearActivity();
eq(activityEl.hidden, true, 'pill hidden when idle');

// the typing bubble mirrors the label
const bubbleChild = { className: 'typing', textContent: 'thinking' };
const bubble = { firstChild: bubbleChild, removed: false, remove() { this.removed = true; } };
api.setTyping(bubble);
api.setActivity('thinking', 'Reading file…');
eq(bubbleChild.textContent, 'Reading file…', 'typing bubble follows the phase');
api.setTyping(null);

// killTyping is safe when there is no bubble
api.killTyping();
ok(true, 'killTyping with no bubble does not throw');

// tool → phase mapping
eq(api.activityFor('read_file', { path: '/sdcard/a.py' }),
   ['reading', 'Reading a.py'], 'read_file → Reading <file>');
eq(api.activityFor('list_files', { directory: '/sdcard/x' }),
   ['reading', 'Listing x'], 'list_files → Listing <dir>');
eq(api.activityFor('write_file', { path: '/sdcard/out.py' }),
   ['running', 'Writing out.py'], 'write_file → Writing <file>');
eq(api.activityFor('run_command', { command: 'ls -la /sdcard' }),
   ['running', 'Running ls -la /sdcard'], 'run_command → Running <cmd>');
eq(api.activityFor('run_command', {}),
   ['running', 'Running shell'], 'run_command without args → shell');
eq(api.activityFor('web_search', { query: 'burmese' }),
   ['running', 'Searching…'], 'web_search → Searching');
eq(api.activityFor('smart_search', {}),
   ['running', 'Searching…'], 'smart_search → Searching');
eq(api.activityFor('web_fetch', { url: 'https://example.com/x' }),
   ['running', 'Fetching https://example.com/x'], 'web_fetch → Fetching <url>');
eq(api.activityFor('set_alarm', { hour: 7 }),
   ['running', 'Running set_alarm'], 'unknown-ish tool falls back to Running <tool>');
eq(api.activityFor('nonexistent_tool', null),
   ['running', 'Running nonexistent_tool'], 'null args tolerated');

// setBusy(false) is the single choke point that clears the pill
api.setActivity('thinking', 'Thinking…');
state.busy = true;
api.setBusy(true);
ok(!activityEl.hidden, 'setBusy(true) leaves the pill alone');
api.setBusy(false);
eq(activityEl.hidden, true, 'setBusy(false) clears the pill');
eq(sendBtn.textContent, '➤', 'idle button glyph restored');

console.log(failures ? `\n${failures} FAILED` : '\nALL PASS');
process.exit(failures ? 1 : 0);
