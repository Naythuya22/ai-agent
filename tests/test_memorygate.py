"""Phase 2 memory gate — evidence-bound long-term memory.

The gate is mechanical: an owner/tool claim is accepted only when
its evidence quote is found verbatim in the harness's own records
(owner messages and tool results). The agent's prose is never
recorded as evidence, so it can never cite itself. Inferences go
to quarantine and expire. These tests pin all of that, including
the parts that must NOT work.

Run:  sh tests/run.sh -p "test_memorygate.py"
"""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from _harness import bootstrap

bootstrap()

import config
from core import memorygate
from core.memorygate import MemoryGateError
from tools import memory_agent


class MemoryGateBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="memorygate-"))
        self._saved = {
            "MEMORY_DIR": memorygate.MEMORY_DIR,
            "ENTRIES_PATH": memorygate.ENTRIES_PATH,
            "EVIDENCE_PATH": memorygate.EVIDENCE_PATH,
            "AUDIT_PATH": memorygate.AUDIT_PATH,
        }
        memorygate.MEMORY_DIR = self.tmp / "memory"
        memorygate.ENTRIES_PATH = memorygate.MEMORY_DIR / "entries.json"
        memorygate.EVIDENCE_PATH = self.tmp / "memory-evidence.jsonl"
        memorygate.AUDIT_PATH = self.tmp / "memory-audit.jsonl"
        self.addCleanup(self._restore)
        self._saved_enabled = config.MEMORY_PROPOSALS_ENABLED
        config.MEMORY_PROPOSALS_ENABLED = True
        self.addCleanup(setattr, config, "MEMORY_PROPOSALS_ENABLED",
                        self._saved_enabled)

    def _restore(self):
        for name, value in self._saved.items():
            setattr(memorygate, name, value)


class GateVerdictCase(MemoryGateBase):
    def test_owner_quote_verified_goes_active(self):
        memorygate.record_evidence(
            "owner", "ကျွန်တော် shell ကို ဖွင့်ထားလိုက်ပြီ ကြာပြီ")
        out = memorygate.propose(
            "shell_state", "Shell commands are enabled on the phone",
            "owner", evidence="shell ကို ဖွင့်ထားလိုက်ပြီ")
        self.assertEqual(out["verdict"], "active")
        self.assertEqual(out["entry"]["status"], "active")

    def test_tool_quote_verified_goes_active(self):
        memorygate.record_evidence(
            "tool", "Error: BLOCKED: protected path: "
                    "~/.ai-agent/agent_config.json (the owner's "
                    "private store is off-limits to the shell)")
        out = memorygate.propose(
            "config_protected",
            "The shell cannot read ~/.ai-agent/agent_config.json",
            "tool", evidence="protected path: "
                              "~/.ai-agent/agent_config.json")
        self.assertEqual(out["verdict"], "active")

    def test_unverifiable_quote_quarantined(self):
        out = memorygate.propose(
            "shell_state", "Shell commands are disabled",
            "tool", evidence="BLOCKED: run_command tool is disabled "
                              "on this install")
        self.assertEqual(out["verdict"], "quarantine")
        self.assertIn("not found", out["entry"]["reason"])

    def test_wrong_kind_quote_quarantined(self):
        memorygate.record_evidence("owner", "the printer is a Zy-Q821")
        out = memorygate.propose(
            "printer", "The printer is a Zy-Q821", "tool",
            evidence="the printer is a Zy-Q821")
        self.assertEqual(out["verdict"], "quarantine")

    def test_short_quote_quarantined(self):
        memorygate.record_evidence("tool", "ok")
        out = memorygate.propose("x_state", "It was ok", "tool",
                                 evidence="ok")
        self.assertEqual(out["verdict"], "quarantine")

    def test_inference_quarantined_with_expiry(self):
        out = memorygate.propose(
            "folder_value", "That folder is probably unimportant",
            "inference")
        self.assertEqual(out["verdict"], "quarantine")
        self.assertIsNotNone(out["entry"]["expires"])

    def test_conflict_quarantined_and_duplicate_detected(self):
        memorygate.record_evidence("owner", "my name is KoHtet")
        first = memorygate.propose(
            "owner_name", "The owner is KoHtet", "owner",
            evidence="my name is KoHtet")
        self.assertEqual(first["verdict"], "active")
        dup = memorygate.propose(
            "owner_name", "The owner is KoHtet", "owner",
            evidence="my name is KoHtet")
        self.assertEqual(dup["verdict"], "duplicate")
        memorygate.record_evidence("owner", "call me something else")
        clash = memorygate.propose(
            "owner_name", "The owner wants a different name",
            "owner", evidence="call me something else")
        self.assertEqual(clash["verdict"], "quarantine")
        self.assertIn("conflicts", clash["entry"]["reason"])

    def test_owner_add_supersedes(self):
        memorygate.record_evidence("owner", "my name is KoHtet")
        memorygate.propose("owner_name", "The owner is KoHtet",
                           "owner", evidence="my name is KoHtet")
        out = memorygate.add_owner("owner_name",
                                   "The owner is Ko Htet (ကိုထက်)")
        self.assertEqual(out["verdict"], "active")
        active = [e for e in memorygate.list_entries()
                  if e["key"] == "owner_name"
                  and e["status"] == "active"]
        self.assertEqual(len(active), 1)
        self.assertIn("ကိုထက်", active[0]["text"])

    def test_kill_removes_from_block(self):
        memorygate.record_evidence("tool", "listing shows alpha.txt")
        out = memorygate.propose(
            "dir_a", "dir_a contains alpha.txt", "tool",
            evidence="listing shows alpha.txt")
        self.assertIn("alpha.txt", memorygate.agent_memory_block())
        memorygate.kill(out["entry"]["id"])
        self.assertNotIn("alpha.txt", memorygate.agent_memory_block())
        st = memorygate.status()
        self.assertEqual(st["entries"]["killed"], 1)
        self.assertEqual(st["proposals"]["poison_kills"], 1)

    def test_quarantine_expires(self):
        out = memorygate.propose("guess", "A guess", "inference")
        self.assertEqual(out["entry"]["status"], "quarantine")
        future = datetime.now().astimezone() + timedelta(days=15)
        saved = memorygate._now_dt
        memorygate._now_dt = lambda: future
        self.addCleanup(setattr, memorygate, "_now_dt", saved)
        entries = memorygate.list_entries()
        by_id = {e["id"]: e for e in entries}
        self.assertEqual(by_id[out["entry"]["id"]]["status"],
                         "expired")
        self.assertNotIn("A guess", memorygate.agent_memory_block())

    def test_block_marks_quarantine_unverified(self):
        memorygate.propose("guess", "An unverified guess",
                           "inference")
        block = memorygate.agent_memory_block()
        self.assertIn("Unverified", block)
        self.assertIn("An unverified guess", block)

    def test_empty_memory_block_is_empty(self):
        self.assertEqual(memorygate.agent_memory_block(), "")

    def test_kill_switch_refuses(self):
        config.MEMORY_PROPOSALS_ENABLED = False
        with self.assertRaises(MemoryGateError):
            memorygate.propose("k", "Some fact", "inference")

    def test_agent_prose_is_not_evidence(self):
        # There is no API to record agent prose; a "tool" claim
        # quoting text that only the agent ever wrote cannot pass.
        memorygate.record_evidence("tool", "real tool output here")
        out = memorygate.propose(
            "fake", "A fabricated gate message", "tool",
            evidence="text only the agent wrote in a reply")
        self.assertEqual(out["verdict"], "quarantine")
        memorygate.record_evidence("agent", "should be ignored")
        self.assertEqual(memorygate._evidence_texts("agent"), [])

    def test_evidence_fifo_prunes(self):
        saved = memorygate._EVIDENCE_KEEP
        memorygate._EVIDENCE_KEEP = 5
        self.addCleanup(setattr, memorygate, "_EVIDENCE_KEEP", saved)
        for i in range(12):
            memorygate.record_evidence("tool", f"record number {i}")
        texts = memorygate._evidence_texts("tool")
        self.assertIn("record number 11", " ".join(texts))
        self.assertNotIn("record number 0", " ".join(texts))

    def test_tool_wrappers(self):
        memorygate.record_evidence("tool", "du shows 99M total")
        ok = memory_agent.propose_memory(
            "apk_size", "The folder totals 99M", "tool",
            evidence="du shows 99M total")
        self.assertIn("Remembered", ok)
        bad = memory_agent.propose_memory(
            "nope", "Something unproven", "tool",
            evidence="a quote that exists nowhere")
        self.assertIn("UNVERIFIED", bad)
        listing = memory_agent.list_memory()
        self.assertIn("apk_size", listing)


class HookIntegrationCase(MemoryGateBase):
    """The two harness funnels feed the evidence log: begin_turn
    records the owner's words; a real streamed turn records tool
    results via _guard_result. The agent's final prose must NOT be
    in the log afterwards."""

    def test_begin_turn_and_tool_hooks(self):
        from test_failure_replies import (
            ActivityStreamBackend, FailureReplyBase)

        rig = FailureReplyBase(methodName="run")
        rig.setUp()
        try:
            mode = rig.new_mode()
            mode._backend = ActivityStreamBackend(str(rig.dir_a))
            mode.begin_turn("မှတ်ထားပေးပါ ဒါအရေးကြီးတယ်")
            mode._messages.append(
                {"role": "user", "text": "မှတ်ထားပေးပါ ဒါအရေးကြီးတယ်"})
            frames = list(mode._handle_cloud_stream())
            self.assertEqual(frames[-1][0], "end")
        finally:
            rig.tearDown()

        owner_texts = " ".join(memorygate._evidence_texts("owner"))
        self.assertIn("ဒါအရေးကြီးတယ်", owner_texts)
        tool_texts = " ".join(memorygate._evidence_texts("tool"))
        self.assertIn("alpha.txt", tool_texts)
        # the assistant's own closing prose is not evidence
        out = memorygate.propose(
            "prose_claim", "Everything is all done", "tool",
            evidence="all done")
        self.assertEqual(out["verdict"], "quarantine")
        # and the real tool evidence does verify
        out2 = memorygate.propose(
            "dir_a_contents", "dir_a contains alpha.txt", "tool",
            evidence="alpha.txt")
        self.assertEqual(out2["verdict"], "active")


if __name__ == "__main__":
    unittest.main()
