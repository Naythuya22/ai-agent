// Headless smoke test for the slash-command autocomplete in web/index.html.
// No browser here, so the autocomplete slice of the page script is executed
// against a minimal DOM stub.
//   node ac_test.js <path/to/index.html>
'use strict';
const fs = require('fs');

const html = fs.readFileSync(process.argv[2] || 'web/index.html', 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const main = scripts.find(s => s.includes('Slash-command autocomplete'));
if (!main) { console.error('FAIL: script block with autocomplete not found'); process.exit(1); }
const start = main.indexOf('// ── Slash-command autocomplete');
const end = main.indexOf("sendBtn.addEventListener('click'");
const slice = main.slice(start, end);

let failures = 0;
function eq(got, want, msg) {
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g !== w) { failures++; console.error(`FAIL ${msg}\n  got  ${g}\n  want ${w}`); }
  else console.log(`ok   ${msg}`);
}
function ok(cond, msg) { if (!cond) { failures++; console.error('FAIL ' + msg); } else console.log('ok   ' + msg); }

function el(tag) {
  const e = {
    tag, children: [], className: '', textContent: '', hidden: false, value: '',
    _html: '', listeners: {},
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); },
    classList: { toggle() {}, add() {}, remove() {}, contains() { return false; } },
    focus() {}, scrollIntoView() {}, remove() {}, firstChild: null,
    set innerHTML(v) { this._html = v; if (v === '') this.children = []; },
    get innerHTML() { return this._html; },
  };
  return e;
}

const acMenu = el('div');
const inputEl = el('input');
const documentStub = { getElementById: (id) => (id === 'ac-menu' ? acMenu : el('div')), createElement: el };
const skills = { list: ['summarize', 'quiz-mm', 'translate-en-mm'], fail: false };
const fetchStub = (url) => Promise.resolve({
  json: () => Promise.resolve(skills.fail ? {} : { skills: skills.list.map(n => ({ name: n, args: 'text', description: n })) }),
});

const factory = new Function(
  'document', 'fetch', 'inputEl', 'setTimeout', 'Date',
  slice + `
  return {
    acEntries, acUpdate, acAccept, acMove, acHide, COMMANDS, acMenu,
    get acIndex(){return acIndex}, get acItems(){return acItems},
    setSkillNames(v){skillNames=v}
  };`
);
const api = factory(documentStub, fetchStub, inputEl, setTimeout, Date);

// ── filtering ──────────────────────────────────────────────
eq(api.acEntries('hello'), null, 'no menu for plain text');
eq(api.acEntries(''), null, 'no menu for empty input');
ok(api.acEntries('/').length === api.COMMANDS.length, 'bare "/" lists every command');

let r = api.acEntries('/cl');
eq(r && r.map(x => x.cmd), ['/clear'], '"/cl" narrows to /clear');

r = api.acEntries('/C');
ok(r && r.length > 0 && r.every(x => x.cmd.toLowerCase().startsWith('/c')), 'case-insensitive prefix');

r = api.acEntries('/memory');
eq(r && r[0].insert, '/memory', 'no-arg command inserts without trailing space');
r = api.acEntries('/code');
eq(r && r[0].insert, '/code ', 'command with args inserts trailing space');
r = api.acEntries('/create-skill');
eq(r && r[0].insert, '/create-skill ', 'placeholder is not pasted into the input');
r = api.acEntries('/skill');
eq(r && r[0].cmd, '/skill <name> <args>', '"/skill" outranks "/skills"');

eq(api.acEntries('/code task here'), null, 'after choosing a command the menu closes');
eq(api.acEntries('/nonsense'), null, 'unknown command → no menu');

// ── skill name completion (async fetch) ────────────────────
ok(api.acEntries('/skill ') === null, 'skills not cached yet → no menu, triggers fetch');

// ── render / move / accept ─────────────────────────────────
api.acUpdate = api.acUpdate; // (kept for clarity)
inputEl.value = '/me';
api.acUpdate();
ok(!acMenu.hidden, 'menu opens for "/me"');
eq(acMenu.children.map(c => c.children[0].textContent), ['/memory'], '"/me" matches only /memory');
eq(acMenu.children[0].className, 'ac-item active', 'first item is active');
eq(acMenu.children[0].children.map(c => c.className), ['ac-cmd', 'ac-desc', 'ac-key'], 'item structure');
eq(acMenu.children[0].children[0].textContent, '/memory', 'command text set via textContent (no HTML injection)');

inputEl.value = '/c';
api.acUpdate();
const nChoices = acMenu.children.length;
ok(nChoices > 1, '"/c" offers several commands (' + nChoices + ')');
api.acMove(1);
eq(api.acIndex, 1, 'ArrowDown moves selection');
api.acMove(-1);
eq(api.acIndex, 0, 'ArrowUp moves back');
api.acMove(-1);
eq(api.acIndex, nChoices - 1, 'ArrowUp wraps to the last item');

inputEl.value = '/clear';
api.acUpdate();
api.acAccept(api.acIndex);
eq(inputEl.value, '/clear', 'accept fills the input');
ok(acMenu.hidden, 'menu closes after accepting a complete command');

inputEl.value = '/code';
api.acUpdate();
api.acAccept(api.acIndex);
eq(inputEl.value, '/code ', 'arg command accepts with a space, caret ready');
ok(acMenu.hidden, '"/code " has no matching command → menu closed');

inputEl.value = '/skill';
api.setSkillNames(['summarize', 'quiz-mm', 'translate-en-mm']);
api.acUpdate();
api.acAccept(api.acIndex);
eq(inputEl.value, '/skill ', '"/skill" accepts straight to the skill slot');
ok(!acMenu.hidden, 'skill list opens right after accepting "/skill"');
eq(acMenu.children.map(c => c.children[0].textContent),
   ['/skill summarize', '/skill quiz-mm', '/skill translate-en-mm'], 'skill names offered immediately');

inputEl.value = '/ne';
api.acUpdate();
ok(!acMenu.hidden, 'menu open again for "/ne"');
inputEl.value = '';
api.acUpdate();
ok(acMenu.hidden, 'clearing the input closes the menu');

// keyboard listener wiring
ok((inputEl.listeners.keydown || []).length === 1, 'keydown handler attached');
ok((inputEl.listeners.input || []).length === 1, 'input handler attached');
ok((inputEl.listeners.blur || []).length === 1, 'blur handler attached');

const keydown = inputEl.listeners.keydown[0];
inputEl.value = '/hi';
api.acUpdate();
let prevented = false;
keydown({ key: 'ArrowDown', preventDefault() { prevented = true; } });
ok(prevented, 'ArrowDown prevents default caret move');
prevented = false;
keydown({ key: 'Escape', preventDefault() { prevented = true; } });
ok(acMenu.hidden, 'Escape closes the menu');

inputEl.value = '/sum';
api.acUpdate();
prevented = false;
keydown({ key: 'Tab', preventDefault() { prevented = true; } });
ok(prevented, 'Tab is captured by the menu');
eq(inputEl.value, '/summary', 'Tab accepted the active command (no stray space)');

// Enter: accepts while incomplete, submits when exact
inputEl.value = '/hist';
api.acUpdate();
prevented = false;
keydown({ key: 'Enter', preventDefault() { prevented = true; } });
ok(prevented, 'Enter accepts an incomplete command instead of sending');
eq(inputEl.value, '/history', 'Enter completed to the full command');
api.acUpdate();                 // menu shows the exact match
prevented = false;
keydown({ key: 'Enter', preventDefault() { prevented = true; } });
ok(!prevented && acMenu.hidden, 'Enter on an exact match lets the form submit');

// ── skill names after the fetch resolves ──────────────────
setTimeout(() => {
  inputEl.value = '/skill ';
  api.acUpdate();
  ok(!acMenu.hidden, 'cached skills open a menu after "/skill "');
  eq(acMenu.children.map(c => c.children[0].textContent),
     ['/skill summarize', '/skill quiz-mm', '/skill translate-en-mm'], 'skill names listed');
  inputEl.value = '/skill quiz';
  api.acUpdate();
  eq(acMenu.children.map(c => c.children[0].textContent), ['/skill quiz-mm'], 'skill name filtered');
  api.acAccept(api.acIndex);
  eq(inputEl.value, '/skill quiz-mm ', 'skill accepted with trailing space');
  ok(acMenu.hidden, 'no skill matches the empty arg slot → menu closed');

  console.log(failures ? `\n${failures} FAILED` : '\nALL PASS');
  process.exit(failures ? 1 : 0);
}, 20);
