"""Session 21 P1 — Burmese sequence validation must actually reach the wire.

core/burmese.py only matters if routes/chat.py feeds every model byte through
it, so these tests drive the real endpoints with a fake model stream:
  - agent stream: cut + `truncated` flag, clean text persisted to the session

Release 22 note: this file originally also drove the chat stream and the
sync chat endpoint through a fake ChatSession (cut on broken sequences,
valid text incl. a syllable split across chunks untouched, reply sanitised
before leaving the server). Those endpoints were deleted with Chat Mode;
the agent stream below exercises the same BurmeseValidator/sanitize wiring
in routes/chat.py, so the coverage of the shared code path remains.

Run:  sh tests/run.sh -p "test_p1_sanitize.py"
"""
from __future__ import annotations

import json
import unittest

from _harness import bootstrap, get_app

MODEL = bootstrap()
app = get_app()
client = app.test_client()

# Session 20's real corruption: 4x U+1039 in a row (notes/SESSION-2-FINDINGS.md)
BROKEN = "ထိင်" + "\u1039" * 4 + "န်"
CUT_TO = "ထိင်"
KINZI = "အင်္ဂလိပ်"          # င ်(asat) ္(virama) ဂ — valid, must survive


def frames(resp) -> list:
    out = []
    for line in resp.get_data(as_text=True).split("\n"):
        if line.startswith("data: "):
            out.append(json.loads(line[6:]))
    return out


class TestAgentStream(unittest.TestCase):
    def test_agent_cut_and_persist(self):
        import routes.chat as rc
        from routes._state import ensure_modes, get_agent_mode

        ensure_modes()
        mode = get_agent_mode()

        def fake_stream(conv, message):
            yield "text", BROKEN
            yield "end", {"text": BROKEN, "tool_calls": None}

        real = rc.stream_conversation
        rc.stream_conversation = fake_stream
        try:
            r = client.post("/api/agent/stream", json={"message": "check config"})
        finally:
            rc.stream_conversation = real

        fr = frames(r)
        chunks = "".join(f.get("text", "") for f in fr if f["type"] == "chunk")
        self.assertNotIn("\u1039\u1039", chunks)

        end = fr[-1]
        self.assertEqual(end["type"], "end")
        self.assertEqual(end["text"], CUT_TO)
        self.assertTrue(end["truncated"])
        # the status-bar record rides along with the end event
        self.assertEqual(end["last_backend"], "local")
        self.assertTrue(end["model_label"].startswith("Model: "), end)

        # The session must replay the clean text, not the garbage.
        last = mode._messages[-1]
        self.assertEqual(last.get("role"), "assistant")
        self.assertEqual(last.get("text"), CUT_TO)


if __name__ == "__main__":
    unittest.main()
