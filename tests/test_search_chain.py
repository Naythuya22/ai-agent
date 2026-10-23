"""Release 28 — the multi-source search chain (tools/web_search.py).

Chain: DuckDuckGo Lite → Bing → Brave → Wikipedia. The first source
that answers wins; bot-blocks (HTTP 202 class), error statuses,
timeouts and zero-parseable-result pages all fail over to the next
independent source; a failed source is remembered as down for a
cooldown so it cannot tax every later search. No raw HTTP status
code may appear in any tool output.

Fixtures mirror the live markup probed from this network on
2026-10-23 (Bing b_algo blocks, Brave data-type="web" snippets).

Run:  sh tests/run.sh -p "test_search_chain.py"
"""
from __future__ import annotations

import unittest

from _harness import bootstrap

bootstrap()

from tools import web_search as ws  # noqa: E402

_BING_HTML = """
<html><body><ol id="b_results">
<li class="b_algo" data-id iid="SERP.1">
  <div class="b_algoheader"><a href="https://example.com/yangon-weather"
     h="ID=SERP,1"><h2><strong>Yangon</strong> Weather Forecast</h2></a></div>
  <div class="b_caption"><p class="b_lineclamp3">Hourly weather forecast
  in Yangon with current conditions, radar and more detail here.</p></div>
</li>
<li class="b_algo" data-id iid="SERP.2">
  <div class="b_algoheader"><a href="https://example.org/second"
     h="ID=SERP,2"><h2>Second Result Title</h2></a></div>
  <div class="b_caption"><p class="b_lineclamp3">Second snippet text that
  is comfortably longer than twenty chars.</p></div>
</li>
</ol></body></html>
"""

_BRAVE_HTML = """
<html><body>
<div class="snippet svelte-a" data-pos="1" data-type="web" data-keynav="true">
  <div class="result-content svelte-b">
    <a href="https://example.net/brave-hit" target="_self" class="svelte-c l1">
      <div class="site-name-content"><div>ExampleNet</div></div>
      <div class="title search-snippet-title line-clamp-2 svelte-c"
           title="Brave Result Title">Brave Result Title</div>
    </a>
    <div class="generic-snippet svelte-d">
      <div class="content desktop-default-regular svelte-d">Brave snippet
      body text for the first result.</div>
    </div>
  </div>
</div>
<div class="snippet svelte-a" data-pos="2" data-type="web">
  <div class="result-content svelte-b">
    <a href="https://example.net/second-hit" class="l1">
      <div class="title search-snippet-title"
           title="Brave Second Title">Brave Second Title</div>
    </a>
    <div class="generic-snippet svelte-d">
      <div class="content svelte-d">Second brave snippet.</div>
    </div>
  </div>
</div>
</body></html>
"""


class BingParserCase(unittest.TestCase):
    def test_parses_algo_blocks(self):
        p = ws._BingParser()
        p.feed(_BING_HTML)
        self.assertEqual(len(p.results), 2)
        first = p.results[0]
        self.assertEqual(first["title"], "Yangon Weather Forecast")
        self.assertEqual(first["url"], "https://example.com/yangon-weather")
        self.assertIn("Hourly weather forecast", first["snippet"])

    def test_ck_a_redirect_is_unwrapped(self):
        import base64
        real = "https://example.com/real-target"
        wrapped = ("https://www.bing.com/ck/a?p=1&u=a1"
                   + base64.urlsafe_b64encode(real.encode()).decode().rstrip("="))
        self.assertEqual(ws._bing_unwrap(wrapped), real)
        self.assertEqual(ws._bing_unwrap(real), real)


class BraveParserCase(unittest.TestCase):
    def test_parses_web_snippets(self):
        p = ws._BraveParser()
        p.feed(_BRAVE_HTML)
        self.assertEqual(len(p.results), 2)
        first = p.results[0]
        self.assertEqual(first["title"], "Brave Result Title")
        self.assertEqual(first["url"], "https://example.net/brave-hit")
        self.assertIn("Brave snippet body text", first["snippet"])
        # The anchor's site-name text must not pollute the title.
        self.assertNotIn("ExampleNet", first["title"])


class ChainFailoverCase(unittest.TestCase):
    """The chain end to end, with transports stubbed per source."""

    def setUp(self):
        self._saved = {
            "_http_get": ws._http_get,
            "_fetch_page": ws._fetch_page,
            "offline_notice": ws.offline_notice,
            "_fallback_hits": ws._fallback_hits,
        }
        ws.offline_notice = lambda *a, **k: None
        ws._down_until.clear()
        self.fetch_calls = []

    def tearDown(self):
        for name, fn in self._saved.items():
            setattr(ws, name, fn)
        ws._down_until.clear()

    def _ddg(self, resp):
        ws._http_get = lambda *a, **k: resp

    def _pages(self, bing=None, brave=None):
        def fake(url, params=None, **kw):
            self.fetch_calls.append(url)
            if "bing.com" in url:
                return bing if bing is not None else ws._Response(202, [], b"")
            if "brave.com" in url:
                return brave if brave is not None else ws._Response(202, [], b"")
            return ws._Response(202, [], b"")
        ws._fetch_page = fake

    def _wiki(self, hits):
        ws._fallback_hits = lambda q, n: hits

    def test_ddg_success_never_touches_the_chain(self):
        self._ddg(ws._Response(200, [], _DDG_OK))
        self._pages(bing=ws._Response(200, [], _BING_HTML.encode()))
        out = ws.web_search("yangon weather", 3)
        self.assertIn("DDG Fixture Title", out)
        self.assertEqual(self.fetch_calls, [])
        self.assertNotIn("via", out)

    def test_ddg_blocked_fails_over_to_bing(self):
        self._ddg(ws._Response(202, [], b""))
        self._pages(bing=ws._Response(200, [], _BING_HTML.encode()))
        self._wiki([])
        out = ws.web_search("yangon weather", 3)
        self.assertIn("(Results via Bing.)", out)
        self.assertIn("Yangon Weather Forecast", out)
        self.assertIn("<untrusted>", out)
        self.assertNotIn("202", out)

    def test_bing_blocked_too_fails_over_to_brave(self):
        self._ddg(ws._Response(202, [], b""))
        self._pages(brave=ws._Response(200, [], _BRAVE_HTML.encode()))
        self._wiki([])
        out = ws.web_search("yangon weather", 3)
        self.assertIn("(Results via Brave.)", out)
        self.assertIn("Brave Result Title", out)
        self.assertNotIn("202", out)

    def test_failed_source_is_skipped_during_cooldown(self):
        self._ddg(ws._Response(202, [], b""))
        self._pages(brave=ws._Response(200, [], _BRAVE_HTML.encode()))
        self._wiki([])
        ws.web_search("yangon weather", 3)      # Bing fails → down
        self.assertTrue(ws._source_cooling("Bing"))
        self.fetch_calls = []
        ws.web_search("yangon weather", 3)      # Bing not retried
        self.assertFalse(any("bing.com" in u for u in self.fetch_calls))

    def test_zero_parseable_results_fail_over(self):
        self._ddg(ws._Response(202, [], b""))
        # Bing answers 200 with a consent-shaped page: no b_algo.
        self._pages(bing=ws._Response(200, [], b"<html><body>consent</body></html>"),
                    brave=ws._Response(200, [], _BRAVE_HTML.encode()))
        self._wiki([])
        out = ws.web_search("yangon weather", 3)
        self.assertIn("(Results via Brave.)", out)

    def test_everything_down_is_a_clean_unavailable(self):
        self._ddg(ws._Response(202, [], b""))
        self._pages()
        self._wiki([])
        out = ws.web_search("yangon weather", 3)
        self.assertTrue(out.startswith("Error: search is unavailable"), out)
        self.assertNotIn("202", out)

    def test_ddg_empty_and_chain_empty_is_no_results(self):
        self._ddg(ws._Response(200, [], b"<html><body></body></html>"))
        self._pages(bing=ws._Response(200, [], b"<html></html>"),
                    brave=ws._Response(200, [], b"<html></html>"))
        self._wiki([])
        out = ws.web_search("zzz-no-such-thing", 3)
        self.assertEqual(out, "No results found for: zzz-no-such-thing")

    def test_chain_results_shape_for_smart_search(self):
        self._ddg(ws._Response(202, [], b""))
        self._pages(bing=ws._Response(200, [], _BING_HTML.encode()))
        results = ws.chain_results("yangon weather", 3)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].source, "Bing")
        self.assertEqual(results[0].title, "Yangon Weather Forecast")
        self.assertTrue(results[0].url.startswith("https://"))


_DDG_OK = (
    "<table><tr><td><a class='result-link' href='https://x.dev/p'>"
    "DDG Fixture Title</a></td></tr>"
    "<tr><td class='result-snippet'>ddg snippet</td></tr></table>"
).encode("utf-8")


if __name__ == "__main__":
    unittest.main(verbosity=2)
