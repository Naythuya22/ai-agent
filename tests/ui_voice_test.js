// Headless smoke test for the Agent-mode voice input helpers in web/index.html.
//   node tests/ui_voice_test.js [path/to/index.html]
'use strict';
const fs = require('fs');

const html = fs.readFileSync(process.argv[2] || 'web/index.html', 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const main = scripts.find(s => s.includes('startListening'));
if (!main) { console.error('FAIL: script with startListening not found'); process.exit(1); }

// Element/voice declarations (voiceBtn, SRImpl, ...) — normally above the slice.
const dStart = main.indexOf('// ── Voice input (Agent mode only)');
const dEnd = main.indexOf('var recog=null;', dStart);
if (dStart < 0 || dEnd < 0) { console.error('FAIL: voice declaration bounds'); process.exit(1); }
const decls = main.slice(dStart, dEnd + 'var recog=null;'.length);

// Implementation block — must stay clear of the slice ui_activity_test.js evals.
const iStart = main.indexOf('// ── Voice input — Web Speech API');
const iEnd = main.indexOf('async function sendStream', iStart);
if (iStart < 0 || iEnd < 0 || iEnd < iStart) { console.error('FAIL: voice impl bounds'); process.exit(1); }
const impl = main.slice(iStart, iEnd);
if (!/if\(voiceBtn\)\{/.test(impl)) { console.error('FAIL: voice click handler missing'); process.exit(1); }

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
    hidden: false, disabled: false, value: '', attrs: {}, listeners: {},
    classList: {
      add(c) { classes.add(c); },
      remove(c) { classes.delete(c); },
      toggle(c, on) {
        if (on === undefined) { classes.has(c) ? classes.delete(c) : classes.add(c); }
        else if (on) { classes.add(c); } else { classes.delete(c); }
      },
      contains(c) { return classes.has(c); },
    },
    setAttribute(k, v) { this.attrs[k] = v; },
    addEventListener(ev, fn) { this.listeners[ev] = fn; },
    focus() { this.focused = true; },
    classes,
  };
}

// Fake SpeechRecognition: records config, fires the handlers the app installed.
function FakeSR() { FakeSR.last = this; this.started = false; this.stopped = false; }
FakeSR.prototype.start = function () { this.started = true; if (this.onstart) this.onstart(); };
FakeSR.prototype.stop = function () { this.stopped = true; if (this.onend) this.onend(); };
FakeSR.prototype.abort = function () { this.stopped = true; if (this.onend) this.onend(); };

function build(windowStub) {
  FakeSR.last = null;
  const voiceBtn = el();
  const inputEl = el();
  const status = { text: '' };
  const state = { mode: 'chat', busy: false, abort: null, listening: false };
  const documentStub = { getElementById: (id) => (id === 'voice-btn' ? voiceBtn : el()) };
  const setStatus = (t) => { status.text = t || ''; };
  const api = new Function(
    'document', 'window', 'state', 'inputEl', 'setStatus',
    decls + '\n' + impl + `
    return { syncAgentTools, startListening, stopListening, insertTranscript,
             setListeningUI, setToolsOpen,
             get recog() { return recog; },
             get supported() { return voiceSupported; } };`
  )(documentStub, windowStub, state, inputEl, setStatus);
  return { api, voiceBtn, inputEl, state, status };
}

// ── Mode sync: the button exists for Agent only ────────────────────────
{
  const t = build({ SpeechRecognition: FakeSR });
  t.api.syncAgentTools();   // what the page runs on init (mode is 'chat')
  eq(t.voiceBtn.hidden, true, 'voice button hidden by default (chat mode)');
  t.state.mode = 'agent';
  t.api.syncAgentTools();
  eq(t.voiceBtn.hidden, true, 'voice button collapsed by default in agent mode');
  t.api.setToolsOpen(true);
  eq(t.voiceBtn.hidden, false, '“+” toggle reveals the voice button');
  t.api.setToolsOpen(false);
  eq(t.voiceBtn.hidden, true, 'toggling again hides the voice button');
  t.state.mode = 'code';
  t.api.syncAgentTools();
  eq(t.voiceBtn.hidden, true, 'voice button hidden again in code mode');
}

// Leaving agent mode while listening must stop the microphone.
{
  const t = build({ SpeechRecognition: FakeSR });
  t.state.mode = 'agent';
  t.api.syncAgentTools();
  t.api.startListening();
  eq(t.state.listening, true, 'listening flag set after start');
  eq(FakeSR.last.lang, 'my-MM', 'recognition language is my-MM');
  eq(FakeSR.last.interimResults, true, 'interim results enabled');
  eq(t.voiceBtn.classes.has('listening'), true, 'red listening indicator on');
  t.state.mode = 'chat';
  t.api.syncAgentTools();
  eq(t.state.listening, false, 'leaving agent mode stops listening');
  eq(t.voiceBtn.classes.has('listening'), false, 'listening indicator off after mode switch');
}

// ── Transcript insertion ───────────────────────────────────────────────
{
  const t = build({ SpeechRecognition: FakeSR });
  t.api.insertTranscript('မင်္ဂလာပါ');
  eq(t.inputEl.value, 'မင်္ဂလာပါ', 'first transcript lands in the chat box');
  t.api.insertTranscript('ကြိုဆိုပါတယ်');
  eq(t.inputEl.value, 'မင်္ဂလာပါကြိုဆိုပါတယ်',
     'Burmese segments concatenate without a spurious space');
  t.api.insertTranscript('hello');
  t.api.insertTranscript('world');
  eq(t.inputEl.value, 'မင်္ဂလာပါကြိုဆိုပါတယ်hello world',
     'two Latin runs stay word-separated');
  t.api.insertTranscript('   ');
  eq(t.inputEl.value, 'မင်္ဂလာပါကြိုဆိုပါတယ်hello world', 'blank transcript is ignored');
}

// ── Result handling: final text goes to the box, interim only to status ─
{
  const t = build({ SpeechRecognition: FakeSR });
  t.state.mode = 'agent';
  t.api.syncAgentTools();
  t.api.startListening();
  const rec = FakeSR.last;
  rec.onresult({ resultIndex: 0, results: [{ isFinal: false, 0: { transcript: 'စမ်း' } }] });
  eq(t.inputEl.value, '', 'interim text is not committed to the chat box');
  ok(t.status.text.indexOf('စမ်း') >= 0, 'interim text shown in the status line');
  // Real events carry a cumulative list — only the tail segment is final.
  rec.onresult({
    resultIndex: 1,
    results: [
      { isFinal: false, 0: { transcript: 'စမ်း' } },
      { isFinal: true, 0: { transcript: 'စမ်းသပ်မှု' } },
    ],
  });
  eq(t.inputEl.value, 'စမ်းသပ်မှု', 'final transcript committed');
  rec.onend();
  eq(t.state.listening, false, 'recognition end clears the listening flag');
  eq(t.voiceBtn.classes.has('listening'), false, 'recognition end clears the indicator');
  eq(t.status.text, 'ready', 'status returns to ready after a clean end');
}

// ── Guards ─────────────────────────────────────────────────────────────
{
  const t = build({ SpeechRecognition: FakeSR });
  t.state.busy = true;
  t.api.startListening();
  eq(FakeSR.last, null, 'busy state refuses to start the microphone');
  ok(t.status.text.indexOf('busy') >= 0, 'busy refusal is reported in the status line');
}
{
  const t = build({});
  eq(t.api.supported, false, 'no Web Speech API → reported unsupported');
  t.voiceBtn.listeners.click();
  ok(t.status.text.indexOf('voice unavailable') >= 0, 'unsupported click explains why');
}
{
  const t = build({ SpeechRecognition: FakeSR });
  t.state.mode = 'agent';
  t.api.syncAgentTools();
  t.api.startListening();
  const rec = FakeSR.last;
  const heard = t.status.text;
  t.api.stopListening(true);
  eq(t.state.listening, false, 'stopListening clears the flag');
  eq(rec.stopped, true, 'stopListening aborts the recognition');
  eq(t.status.text, heard, 'silent stop leaves the status line alone');
}

console.log(failures ? `\n${failures} FAILED` : '\nALL PASS');
process.exit(failures ? 1 : 0);
