"""Session 21 P1 — Burmese sequence validation must actually reach the wire.

core/burmese.py only matters if routes/chat.py feeds every model byte through
it, so these tests drive the real endpoints with a fake model stream:
  - chat stream: broken sequence → cut, drain, `end` carries the clean text
  - chat stream: valid text (incl. a syllable split across chunks) → untouched
  - sync chat:   reply sanitised before it leaves the server
  - agent stream: cut + `truncated` flag, clean text persisted to the session

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


class FakeChatSession:
    """Duck-typed stand-in for ChatSession (route only calls send_stream)."""

    def __init__(self, events):
        self._events = events

    def send_stream(self, msg):
        for kind, data in self._events:
            yield kind, data

    def send(self, msg):
        return self._events

    def get_messages(self):
        return []


class ChatSanitizeBase(unittest.TestCase):
    events = []

    def stream(self):
        from routes._state import ensure_modes, get_chat_mode
        ensure_modes()
        mode = get_chat_mode()
        real = mode.session
        mode.session = FakeChatSession(self.events)
        try:
            return client.post("/api/chat/stream", json={"message": "hi"})
        finally:
            mode.session = real

    def frames(self):
        return frames(self.stream())


class TestChatStreamCut(ChatSanitizeBase):
    events = [
        ("text", BROKEN),                       # cut lands here
        ("text", " ဆက်လုပ်ပါ။"),              # must never be emitted
        ("end", BROKEN + " ဆက်လုပ်ပါ။"),      # client renders this
    ]

    def test_cut_stops_the_stream(self):
        fr = self.frames()
        chunks = "".join(f.get("text", "") for f in fr if f["type"] == "chunk")
        self.assertNotIn("\u1039\u1039", chunks, "repeated virama reached the UI")
        self.assertNotIn("ဆက်လုပ်", chunks, "text after the marker was emitted")

    def test_end_carries_the_clean_text(self):
        fr = self.frames()
        end = fr[-1]
        self.assertEqual(end["type"], "end")
        self.assertEqual(end["text"], CUT_TO)
        self.assertTrue(end["truncated"], "cut not reported as truncated")

    def test_stream_still_terminates(self):
        # A cut must drain the generator (F7b), never abandon it mid-flight.
        fr = self.frames()
        self.assertEqual(fr[-1]["status"], "ready")


class TestChatStreamValid(ChatSanitizeBase):
    events = [
        ("text", "မင်္ဂလာပါ။ "),     # kinzi sits inside this chunk
        ("text", "ခင်ဗျာ"),
        ("end", "မင်္ဂလာပါ။ ခင်ဗျာ"),
    ]

    def test_valid_text_is_untouched(self):
        fr = self.frames()
        self.assertEqual(fr[-1]["text"], "မင်္ဂလာပါ။ ခင်ဗျာ")
        self.assertFalse(fr[-1]["truncated"])

    def test_chunks_pass_through(self):
        chunks = "".join(f["text"] for f in self.frames() if f["type"] == "chunk")
        self.assertEqual(chunks, "မင်္ဂလာပါ။ ခင်ဗျာ")


class TestChatStreamChunkBoundary(ChatSanitizeBase):
    # The virama lands at the end of chunk 1 — that is only wrong if nothing
    # follows, so the validator must wait for chunk 2 instead of cutting.
    events = [
        ("text", "အင်္"),
        ("text", "ဂလိပ်"),
        ("end", KINZI),
    ]

    def test_split_syllable_survives(self):
        fr = self.frames()
        self.assertEqual(fr[-1]["text"], KINZI)
        self.assertFalse(fr[-1]["truncated"])
        chunks = "".join(f["text"] for f in fr if f["type"] == "chunk")
        self.assertEqual(chunks, KINZI)


class TestSyncChatReply(unittest.TestCase):
    def test_reply_is_sanitised(self):
        from routes._state import ensure_modes, get_chat_mode
        ensure_modes()
        mode = get_chat_mode()
        real = mode.session
        mode.session = FakeChatSession(BROKEN)
        try:
            r = client.post("/api/chat", json={"message": "hi"})
        finally:
            mode.session = real
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["reply"], CUT_TO)


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
