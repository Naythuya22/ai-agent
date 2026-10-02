'use strict';

const CACHE = 'ai-agent-v1';
const SHELL = [
  '/',
  '/manifest.json',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE)
      .then((c) => c.addAll(SHELL))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) =>
        Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET' && req.method !== 'POST') return;

  let url;
  try { url = new URL(req.url); } catch (_) { return; }

  // Same-origin only
  if (url.origin !== self.location.origin) return;

  const path = url.pathname;

  // ── API + OpenAI bridge → network, never cache ──
  if (path.startsWith('/api/') || path.startsWith('/v1/')) {
    e.respondWith(
      fetch(req).catch((err) =>
        new Response(
          JSON.stringify({ error: 'offline', detail: String(err) }),
          { status: 503, headers: { 'Content-Type': 'application/json' } }
        )
      )
    );
    return;
  }

  // ── Static shell → cache-first ──
  e.respondWith(
    caches.match(req).then((cached) => {
      if (cached) return cached;
      return fetch(req).then((resp) => {
        if (resp && resp.status === 200 && resp.type === 'basic') {
          const copy = resp.clone();
          caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
        }
        return resp;
      }).catch(() => caches.match('/'));
    })
  );
});

// Allow page to force-refresh cache
self.addEventListener('message', (e) => {
  if (e.data && e.data.type === 'SKIP_WAITING') self.skipWaiting();
});
