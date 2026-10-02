'use strict';

// ---- State ----
const state = {
  mode: 'chat',
  busy: false,
};

// ---- DOM ----
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const formEl = document.getElementById('input-form');
const sendBtn = document.getElementById('send-btn');
const statusEl = document.getElementById('status');
const menuBtn = document.getElementById('menu-btn');
const drawer = document.getElementById('drawer');
const modeBtns = document.querySelectorAll('.mode-btn');
const cmdBtns = document.querySelectorAll('.cmd');

// ---- Utilities ----
function clearWelcome() {
  const w = chatEl.querySelector('.welcome');
  if (w) w.remove();
}

function scrollBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function appendMsg(text, kind) {
  clearWelcome();
  const div = document.createElement('div');
  div.className = `msg ${kind}`;
  div.textContent = text;
  chatEl.appendChild(div);
  scrollBottom();
  return div;
}

function appendTyping() {
  clearWelcome();
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.innerHTML = '<span class="typing">thinking</span>';
  chatEl.appendChild(div);
  scrollBottom();
  return div;
}

function setStatus(text) {
  statusEl.textContent = text || '';
}

function setBusy(b) {
  state.busy = b;
  sendBtn.disabled = b;
  inputEl.disabled = b;
}

// ---- Mode ----
function setMode(mode) {
  state.mode = mode;
  modeBtns.forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });
  setStatus(`mode: ${mode}`);
}

modeBtns.forEach(btn => {
  btn.addEventListener('click', () => setMode(btn.dataset.mode));
});

// ---- Drawer ----
menuBtn.addEventListener('click', () => drawer.classList.add('open'));
drawer.addEventListener('click', (e) => {
  if (e.target === drawer) drawer.classList.remove('open');
});

cmdBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const cmd = btn.dataset.cmd;
    drawer.classList.remove('open');
    if (cmd === 'exit') return;
    runCommand(cmd);
  });
});

// ---- Command handling ----
async function runCommand(cmd) {
  setBusy(true);
  try {
    if (cmd === 'memory') {
      const r = await fetch('/api/memory');
      const j = await r.json();
      appendMsg(j.text || '(empty)', 'system');
    } else if (cmd === 'profile') {
      const r = await fetch('/api/profile');
      const j = await r.json();
      const lines = Object.entries(j.profile || {})
        .map(([k, v]) => `${k}: ${v}`)
        .join('\n');
      appendMsg(lines || '(profile empty)', 'system');
    } else if (cmd === 'clear') {
      chatEl.innerHTML = '';
      await fetch('/api/clear', { method: 'POST' });
      appendMsg('Conversation cleared.', 'system');
    } else if (cmd === 'mode') {
      const r = await fetch('/api/mode');
      const j = await r.json();
      appendMsg(`mode: ${j.mode}\ncontext: ${j.context}`, 'system');
    }
  } catch (e) {
    appendMsg(`Error: ${e.message}`, 'error');
  } finally {
    setBusy(false);
    inputEl.focus();
  }
}

// ---- Send message ----
async function send(text) {
  appendMsg(text, 'user');
  const typing = appendTyping();
  setBusy(true);
  setStatus('thinking…');

  try {
    const endpoint = state.mode === 'agent' ? '/api/agent' : '/api/chat';
    const r = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    const j = await r.json();
    typing.remove();

    if (!r.ok) {
      appendMsg(j.error || `HTTP ${r.status}`, 'error');
      setStatus('error');
      return;
    }

    appendMsg(j.reply || '(no reply)', 'assistant');

    if (j.remembered && j.remembered.length) {
      appendMsg(`(remembered: ${j.remembered.join(', ')})`, 'system');
    }
    setStatus(j.status || 'ready');
  } catch (e) {
    typing.remove();
    appendMsg(`Network error: ${e.message}`, 'error');
    setStatus('offline');
  } finally {
    setBusy(false);
    inputEl.focus();
  }
}

// ---- Form ----
formEl.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text || state.busy) return;
  inputEl.value = '';

  if (text.startsWith('/')) {
    const cmd = text.slice(1).trim();
    runCommand(cmd);
    return;
  }
  send(text);
});

// ---- Init ----
inputEl.focus();
setStatus('ready');
