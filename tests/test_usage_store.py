"""core/usage_store.py + the Release 17 metering points.

The ledger's contracts:

* a completed cloud step's measured tokens land in the aggregates
  exactly as the provider reported them;
* a call with no usage block is still a call — tokens stay None
  (unknown), never zero, never estimated;
* recording NEVER raises into a turn, however broken the store is;
* totals survive a restart (a fresh load reads what was saved).

Plus one end-to-end pass: a real AgentMode cloud turn and a local
turn, driven through the test_agent_cloud-style rig, meter exactly
one call each.

Run:  sh tests/run.sh -p "test_usage_store.py"
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _harness import bootstrap

bootstrap()

import config  # noqa: E402
from core import usage_store  # noqa: E402
from core.usage_store import UsageStore  # noqa: E402
from modes import agent as agent_mod  # noqa: E402


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="usage-store-"))
        self.path = self.dir / "usage.json"

    def load(self) -> UsageStore:
        return UsageStore.load(self.path)


class RecordTests(StoreCase):
    def test_measured_usage_aggregates_exactly(self):
        s = self.load()
        s.record("openai", "gpt-4o-mini",
                 {"prompt_tokens": 100, "completion_tokens": 50})
        s.record("openai", "gpt-4o-mini",
                 {"prompt_tokens": 30, "completion_tokens": 10})
        agg = s.totals()["openai"]["models"]["gpt-4o-mini"]
        self.assertEqual(agg["calls"], 2)
        self.assertEqual(agg["measured_calls"], 2)
        self.assertEqual(agg["prompt_tokens"], 130)
        self.assertEqual(agg["completion_tokens"], 60)
        self.assertTrue(agg["updated"])

    def test_partial_usage_sums_what_was_reported(self):
        s = self.load()
        s.record("gemini", "gemini-2.5-flash", {"prompt_tokens": 42})
        agg = s.totals()["gemini"]["models"]["gemini-2.5-flash"]
        self.assertEqual(agg["calls"], 1)
        self.assertEqual(agg["measured_calls"], 1)
        self.assertEqual(agg["prompt_tokens"], 42)
        self.assertIsNone(agg["completion_tokens"])

    def test_unmeasured_call_keeps_tokens_unknown(self):
        s = self.load()
        s.record("openai", "gpt-4o-mini", None)
        s.record("openai", "gpt-4o-mini", {})
        s.record("openai", "gpt-4o-mini", {"total_tokens": 99})
        agg = s.totals()["openai"]["models"]["gpt-4o-mini"]
        self.assertEqual(agg["calls"], 3)
        self.assertEqual(agg["measured_calls"], 0)
        self.assertIsNone(agg["prompt_tokens"])
        self.assertIsNone(agg["completion_tokens"])

    def test_bogus_usage_values_are_not_counts(self):
        s = self.load()
        s.record("openai", "m", {"prompt_tokens": "abc",
                                 "completion_tokens": -5})
        s.record("openai", "m", {"prompt_tokens": True,
                                 "completion_tokens": 1.5})
        agg = s.totals()["openai"]["models"]["m"]
        self.assertEqual(agg["calls"], 2)
        self.assertEqual(agg["measured_calls"], 0)
        self.assertIsNone(agg["prompt_tokens"])
        self.assertIsNone(agg["completion_tokens"])

    def test_numeric_strings_and_floats_count(self):
        s = self.load()
        s.record("custom", "m", {"prompt_tokens": "120",
                                 "completion_tokens": 30.0})
        agg = s.totals()["custom"]["models"]["m"]
        self.assertEqual(agg["prompt_tokens"], 120)
        self.assertEqual(agg["completion_tokens"], 30)

    def test_models_and_providers_stay_separate(self):
        s = self.load()
        s.record("openai", "a", {"prompt_tokens": 1, "completion_tokens": 1})
        s.record("openai", "b", {"prompt_tokens": 2, "completion_tokens": 2})
        s.record("deepseek", "a", {"prompt_tokens": 4, "completion_tokens": 4})
        t = s.totals()
        self.assertEqual(t["openai"]["models"]["a"]["prompt_tokens"], 1)
        self.assertEqual(t["openai"]["models"]["b"]["prompt_tokens"], 2)
        self.assertEqual(t["deepseek"]["models"]["a"]["prompt_tokens"], 4)


class PersistenceTests(StoreCase):
    def test_round_trip_stamps_version_and_survives_reload(self):
        s = self.load()
        s.record("openai", "gpt-4o-mini",
                 {"prompt_tokens": 7, "completion_tokens": 3})
        s.save()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(raw["version"], 1)
        self.assertTrue(raw["updated"])
        again = self.load()
        agg = again.totals()["openai"]["models"]["gpt-4o-mini"]
        self.assertEqual(agg["calls"], 1)
        self.assertEqual(agg["prompt_tokens"], 7)
        self.assertEqual(agg["completion_tokens"], 3)

    def test_missing_file_is_an_empty_store(self):
        self.assertEqual(self.load().totals(), {})

    def test_corrupt_file_loads_empty_and_is_preserved(self):
        self.path.write_text("{not json", encoding="utf-8")
        s = self.load()
        self.assertEqual(s.totals(), {})
        aside = list(self.dir.glob("usage.json.corrupt-*"))
        self.assertEqual(len(aside), 1, "corrupt ledger was not preserved")


class RecordCallTests(unittest.TestCase):
    """The module-level recorder modes/agent.py feeds."""

    def setUp(self):
        p = usage_store.default_path()
        if p.exists():
            p.unlink()

    def tearDown(self):
        p = usage_store.default_path()
        if p.exists():
            p.unlink()

    def test_cloud_call_lands_under_its_provider(self):
        ok = usage_store.record_call(
            "cloud", "gpt-4o-mini", "openai",
            {"prompt_tokens": 11, "completion_tokens": 6})
        self.assertTrue(ok)
        agg = usage_store.totals()["openai"]["models"]["gpt-4o-mini"]
        self.assertEqual(agg["calls"], 1)
        self.assertEqual(agg["prompt_tokens"], 11)

    def test_cloud_without_provider_is_skipped(self):
        self.assertFalse(usage_store.record_call("cloud", "m", "", None))
        self.assertEqual(usage_store.totals(), {})

    def test_local_call_counts_without_tokens(self):
        self.assertTrue(usage_store.record_call("local", "Gemma 4 E4B"))
        agg = usage_store.totals()["local"]["models"]["Gemma 4 E4B"]
        self.assertEqual(agg["calls"], 1)
        self.assertIsNone(agg["prompt_tokens"])
        self.assertIsNone(agg["completion_tokens"])

    def test_other_backends_are_ignored(self):
        self.assertFalse(usage_store.record_call("", "m", "openai", None))
        self.assertFalse(usage_store.record_call("reset", "m", "openai", None))
        self.assertEqual(usage_store.totals(), {})

    def test_unwritable_store_never_raises(self):
        saved = config.CONFIG_DIR
        blocker = Path(tempfile.mkdtemp(prefix="usage-block-")) / "a-file"
        blocker.write_text("not a dir", encoding="utf-8")
        config.CONFIG_DIR = blocker / ".ai-agent"
        try:
            self.assertFalse(usage_store.record_call(
                "cloud", "m", "openai",
                {"prompt_tokens": 1, "completion_tokens": 1}))
            self.assertEqual(usage_store.totals(), {})
        finally:
            config.CONFIG_DIR = saved

    def test_meter_usage_swallows_store_failure(self):
        from unittest import mock
        with mock.patch.object(usage_store, "record_call",
                               side_effect=RuntimeError("boom")):
            agent_mod.meter_usage("cloud", "m", "openai",
                                  {"prompt_tokens": 1})  # must not raise


# ── end-to-end: real AgentMode turns meter exactly once ──

ALPHA = "<redacted>"


class _Transport:
    def __init__(self):
        self.calls = []

    def __call__(self, url, body, headers, timeout=30):
        self.calls.append((url, body, headers))
        return {"choices": [{"message": {"content": "CLOUD_OK"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2}}


class _FakeConversation:
    def __init__(self, log):
        self.log = log

    def send_message(self, msg):
        self.log.append(("message", msg))
        return {"content": [{"text": "LOCAL_OK"}]}


class _FakeEngine:
    def __init__(self, log):
        self.log = log

    def create_conversation(self, **kw):
        self.log.append(("create_conversation", kw))
        return _FakeConversation(self.log)


class _FakeLocalLLM:
    def __init__(self):
        self.log = []
        self._engine = None

    def load(self):
        self.log.append(("load", None))
        self._engine = _FakeEngine(self.log)


class AgentMeteringTests(unittest.TestCase):
    def setUp(self):
        from core.key_manager import KeyManager
        self.km = KeyManager()
        if self.km.path.exists():
            self.km.path.unlink()
        self.km.add_key("openai", ALPHA, "alpha")
        cfg = self.km.load()
        cfg["prefer_cloud"] = True
        cfg["cost_ack"] = True
        cfg["providers"]["openai"]["model"] = "gpt-4o-mini"
        self.km.save()
        p = usage_store.default_path()
        if p.exists():
            p.unlink()

        from modes.agent import AgentMode
        self.mode = AgentMode(_FakeLocalLLM(), None)
        self.mode._store = None
        self.mode._tools = []
        self.mode._messages = []

    def tearDown(self):
        p = usage_store.default_path()
        if p.exists():
            p.unlink()

    def test_cloud_turn_meters_one_measured_call(self):
        from core.agent_backend import AgentBackend
        self.km.set_mode("cloud")
        self.km.save()
        be = AgentBackend(keys=self.km, net_state=lambda: "online",
                          http=_Transport())
        self.mode._backend = be
        out = self.mode.handle("hello")
        self.assertEqual(out, "CLOUD_OK")
        agg = usage_store.totals()["openai"]["models"]["gpt-4o-mini"]
        self.assertEqual(agg["calls"], 1)
        self.assertEqual(agg["measured_calls"], 1)
        self.assertEqual(agg["prompt_tokens"], 5)
        self.assertEqual(agg["completion_tokens"], 2)

    def test_local_turn_meters_one_unmeasured_call(self):
        self.km.set_mode("local")
        self.km.save()
        out = self.mode.handle("hello")
        self.assertEqual(out, "LOCAL_OK")
        t = usage_store.totals()
        self.assertIn("local", t)
        models = t["local"]["models"]
        self.assertEqual(sum(m["calls"] for m in models.values()), 1)
        for m in models.values():
            self.assertIsNone(m["prompt_tokens"])


if __name__ == "__main__":
    unittest.main()
