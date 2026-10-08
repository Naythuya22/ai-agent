'use strict';

// __CACHE_VERSION__ is replaced server-side with sha256(index.html)[:8]
// (routes/pwa.py). A deploy changes it, so this file changes byte-for-byte
// and the browser installs the new worker instead of serving stale HTML.
const CACHE = 'ai-agent-__CACHE_VERSION__';
const CACHE_PREFIX = 'ai-agent-';
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
        Promise.all(
          keys
            .filter((k) => k.startsWith(CACHE_PREFIX) && k !== CACHE)
            .map((k) => caches.delete(k))
        )
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

  // /terminal/* is a live reverse-proxy for ttyd. Never cache,
  // never serve from cache — let the browser handle it directly.
  if (path.startsWith('/terminal/')) return;

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

  // ── HTML documents → network-first (B1) ──
  // index.html must never come from the previous deploy's cache while the
  // device is online; the cache is only the offline fallback.
  const isDocument = req.mode === 'navigate' || path === '/' || path.endsWith('.html');
  if (isDocument) {
    e.respondWith(
      fetch(req)
        .then((resp) => {
          if (resp && resp.status === 200 && resp.type === 'basic') {
            const copy = resp.clone();
            caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
          }
          return resp;
        })
        .catch(() =>
          caches.match(req).then((cached) =>
            cached ||
            caches.match('/').then((shell) =>
              shell ||
              new Response('<h1>offline</h1>', {
                status: 503,
                headers: { 'Content-Type': 'text/html; charset=utf-8' },
              })
            )
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
