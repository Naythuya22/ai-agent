"""core.model_registry + core.context_trim — model-aware context windows.

The context meter used to divide every request by ``Config.context_size`` (the
local engine's 8192-token KV cache), so a 1M-token cloud model reported 0.1%
and a 64K model that was about to be refused reported nothing. These tests pin
the four resolution layers, the learned-limit store, the offline estimator and
the trimmer that has to fit the transcript into whatever window was resolved.

No network, no provider, no key: the learned limits are written to a temp path
and the resolver is handed explicit ``limits=`` maps wherever a file would
otherwise be involved.

Run:  sh tests/run.sh -p "test_model_registry.py"
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _harness import bootstrap

bootstrap()

from core import context_trim as CT          # noqa: E402
from core import model_registry as MR        # noqa: E402

BURMESE = "ကိုကြီး ညီကို မင်္ဂလာပါ ဒီနေ့ လောက်စားချင်တာလား"


# ── names and endpoints ──────────────────────────────────────────────────

class NameTests(unittest.TestCase):
    def test_gemini_models_prefix_is_dropped(self):
        self.assertEqual(MR.normalize_model("models/gemini-2.5-flash"),
                         "gemini-2.5-flash")

    def test_case_and_whitespace_are_normalised(self):
        self.assertEqual(MR.normalize_model("  DeepSeek-Chat "), "deepseek-chat")

    def test_openrouter_name_yields_bare_candidates(self):
        """``google/gemini-2.5-flash`` must still hit the table.

        The ``:free`` slug for this model was retired upstream, so it is no
        longer the input under test; the routing-suffix peeling it used to
        exercise is pinned below with a slug OpenRouter still serves.
        """
        names = MR.lookup_names("google/gemini-2.5-flash")
        self.assertIn("google/gemini-2.5-flash", names)
        self.assertIn("gemini-2.5-flash", names)

        free = MR.lookup_names("qwen/qwen3.6-plus:free")
        self.assertIn("qwen/qwen3.6-plus:free", free)
        self.assertIn("qwen/qwen3.6-plus", free)
        self.assertIn("qwen3.6-plus", free)

    def test_no_model_yields_no_candidates(self):
        self.assertEqual(MR.lookup_names(""), [])

    def test_endpoint_classification(self):
        self.assertEqual(MR.classify_endpoint("https://api.deepseek.com"), "cloud")
        self.assertEqual(MR.classify_endpoint("http://127.0.0.1:11434/v1"), "local")
        self.assertEqual(MR.classify_endpoint("http://localhost:8080"), "local")
        self.assertEqual(MR.classify_endpoint("https://my-box.lan/v1"), "unknown")
        self.assertEqual(MR.classify_endpoint(""), "unknown")


# ── resolution ───────────────────────────────────────────────────────────

class ResolveTests(unittest.TestCase):
    def test_exact_match_wins(self):
        info = MR.describe_limit("gemini-2.5-flash", "",
                                 limits={})
        self.assertEqual(info["limit"], 1_048_576)
        self.assertEqual(info["source"], "exact")

    def test_prefix_match_for_a_dated_variant(self):
        info = MR.describe_limit("gpt-4o-2024-08-06", "", limits={})
        self.assertEqual(info["source"], "prefix")
        self.assertEqual(info["matched"], "gpt-4o")
        self.assertEqual(info["limit"], 128_000)

    def test_longest_prefix_wins(self):
        """`claude-3-5-sonnet-latest` must not settle for `claude-3`."""
        info = MR.describe_limit("claude-3-5-sonnet-latest", "", limits={})
        self.assertEqual(info["source"], "prefix")
        self.assertEqual(info["limit"], 200_000)

    def test_runtime_layer_beats_the_table(self):
        info = MR.describe_limit("gemini-2.5-flash", "", limits={
            MR.store_key("gemini-2.5-flash"): 32_000})
        self.assertEqual(info["limit"], 32_000)
        self.assertEqual(info["source"], "runtime")

    def test_local_endpoint_default(self):
        info = MR.describe_limit("some-unreleased-thing",
                                "http://localhost:11434/v1", limits={})
        self.assertEqual(info["limit"], MR.LOCAL_CONTEXT_DEFAULT)
        self.assertEqual(info["source"], "endpoint_local")

    def test_cloud_endpoint_default(self):
        info = MR.describe_limit("some-unreleased-thing",
                                "https://api.deepseek.com", limits={})
        self.assertEqual(info["limit"], MR.UNKNOWN_CLOUD_DEFAULT)
        self.assertEqual(info["source"], "endpoint_cloud")

    def test_unknown_host_default_is_the_safe_one(self):
        """A private LAN box must not be handed a 1M guess."""
        info = MR.describe_limit("mystery-model", "https://box.lan/v1", limits={})
        self.assertEqual(info["limit"], MR.UNKNOWN_CLOUD_DEFAULT)

    def test_empty_model_still_answers(self):
        info = MR.describe_limit("", "", limits={})
        self.assertEqual(info["limit"], MR.UNKNOWN_CLOUD_DEFAULT)
        self.assertTrue(info["source"].startswith("endpoint"))

    def test_resolve_limit_matches_describe(self):
        self.assertEqual(
            MR.resolve_limit("deepseek-chat", "", limits={}),
            MR.describe_limit("deepseek-chat", "", limits={})["limit"])

    def test_required_table_entries_present(self):
        """The models this install actually talks to must be in the table."""
        for model in ("mistral-small-4", "devstral-2", "command-a",
                      "deepseek-flash", "gemini-2.5-flash"):
            self.assertIn(model, MR.MODEL_CONTEXT_LIMITS)
            self.assertEqual(
                MR.describe_limit(model, "", limits={})["source"], "exact",
                "%s should resolve exactly" % model)

    def test_registry_values_are_plausible(self):
        for name, limit in MR.MODEL_CONTEXT_LIMITS.items():
            self.assertIsInstance(limit, int, name)
            self.assertGreaterEqual(limit, MR.MIN_WINDOW, name)
            self.assertLessEqual(limit, MR.MAX_LEARNED, name)


# ── provider error text ──────────────────────────────────────────────────

class ErrorTextTests(unittest.TestCase):
    def test_openai_style_message(self):
        msg = ("This model's maximum context length is 65536 tokens. "
               "However, your messages resulted in 70000 tokens.")
        self.assertTrue(MR.is_context_error(msg))
        self.assertEqual(MR.parse_limit(msg), 65536)

    def test_anthropic_style_message(self):
        msg = "prompt is too long: 210000 tokens > 200000 maximum"
        self.assertTrue(MR.is_context_error(msg))
        self.assertEqual(MR.parse_limit(msg), 200_000)

    def test_gemini_style_message(self):
        msg = ("RESOURCE_EXHAUSTED: The input token count (250000) exceeds the "
               "maximum number of tokens allowed (200000).")
        self.assertTrue(MR.is_context_error(msg))

    def test_comparison_form_takes_the_limit_not_the_request(self):
        """'130000 tokens > 65536 tokens' — the smaller number is the window."""
        self.assertEqual(MR.parse_limit("130000 tokens > 65536 tokens"), 65536)

    def test_thousands_separators_are_accepted(self):
        self.assertEqual(MR.parse_limit("context length of 1,048,576 tokens"),
                         1_048_576)

    def test_nonsense_is_not_a_context_error(self):
        for msg in ("", "rate limit reached", "invalid api key",
                    "model not found", None):
            self.assertFalse(MR.is_context_error(msg))

    def test_reasoning_content_400_is_not_a_context_error(self):
        """It has its own retry; borrowing the context one would hide it."""
        msg = ("400 Bad Request: the `reasoning_content` field must be passed "
               "back to the API, but this model's maximum context length is "
               "65536 tokens")
        self.assertFalse(MR.is_context_error(msg))

    def test_absurd_numbers_are_rejected(self):
        self.assertIsNone(MR.parse_limit("maximum context length is 7 tokens"))


# ── the learned-limit store ──────────────────────────────────────────────

class StoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="ai-agent-limits-"))
        self.path = self.dir / "model_limits.json"

    def test_record_then_learn(self):
        self.assertEqual(
            MR.record_limit("mystery-1", "https://api.deepseek.com",
                            text="maximum context length is 32768 tokens",
                            path=self.path),
            32768)
        self.assertEqual(
            MR.load_limits(self.path)[
                MR.store_key("mystery-1", "https://api.deepseek.com")],
            32768)

    def test_recorded_limit_reaches_resolution(self):
        """A learned number is what the meter and the trimmer both read."""
        MR.record_limit("mystery-6", "", limit=3000, path=self.path)
        info = MR.describe_limit("mystery-6", "", limits=MR.load_limits(self.path))
        self.assertEqual(info["limit"], 3000)
        self.assertEqual(info["source"], "runtime")

    def test_saved_file_is_written_and_readable(self):
        MR.record_limit("mystery-2", "", limit=4096, path=self.path)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data["version"], 1)
        self.assertEqual(data["limits"]["mystery-2"], 4096)

    def test_explicit_limit_beats_parsed_text(self):
        MR.record_limit("mystery-3", "", limit=8192,
                        text="maximum context length is 2048 tokens",
                        path=self.path)
        self.assertEqual(MR.load_limits(self.path)["mystery-3"], 8192)

    def test_unparsable_message_writes_nothing(self):
        self.assertIsNone(MR.record_limit("mystery-4", "", text="rate limited",
                                          path=self.path))
        self.assertFalse(self.path.exists())

    def test_empty_model_writes_nothing(self):
        self.assertIsNone(MR.record_limit("", "", limit=8192, path=self.path))

    def test_corrupt_file_is_ignored_not_fatal(self):
        self.path.write_text("{not json at all", encoding="utf-8")
        self.assertEqual(MR.load_limits(self.path), {})

    def test_junk_values_in_the_file_are_dropped(self):
        self.path.write_text(json.dumps({
            "limits": {"ok": 8192, "bad": "abc", "tiny": 3, "huge": 10 ** 9},
        }), encoding="utf-8")
        self.assertEqual(MR.load_limits(self.path), {"ok": 8192})

    def test_entries_are_capped(self):
        for i in range(MR._MAX_ENTRIES + 25):
            MR.record_limit("model-%d" % i, "", limit=4096, path=self.path)
        self.assertLessEqual(len(MR.load_limits(self.path)), MR._MAX_ENTRIES)

    def test_store_key_is_qualified_by_host(self):
        self.assertEqual(MR.store_key("llama3:8b", ""), "llama3:8b")
        self.assertEqual(
            MR.store_key("llama3:8b", "http://192.168.1.9:11434/v1"),
            "192.168.1.9::llama3:8b")
        self.assertEqual(
            MR.store_key("llama3:8b", "https://api.deepseek.com/v1"),
            "api.deepseek.com::llama3:8b")

    def test_reset_removes_the_file(self):
        MR.record_limit("mystery-5", "", limit=8192, path=self.path)
        MR.reset_limits(self.path)
        self.assertFalse(self.path.exists())


# ── estimation ───────────────────────────────────────────────────────────

class EstimateTests(unittest.TestCase):
    def test_english_uses_three_and_a_half_chars_per_token(self):
        text = "a" * 350
        self.assertEqual(CT.estimate_tokens(text), 100)

    def test_burmese_costs_more_per_character(self):
        """Same length, more tokens — that is the whole point of the split."""
        english = CT.estimate_tokens("a" * 100)
        burmese = CT.estimate_tokens("က" * 100)
        self.assertGreater(burmese, english)

    def test_mixed_text_counts_both(self):
        mixed = CT.estimate_tokens("a" * 100 + "က" * 100)
        self.assertGreater(mixed, CT.estimate_tokens("a" * 200))

    def test_empty_text_is_zero_tokens(self):
        self.assertEqual(CT.estimate_tokens(""), 0)
        self.assertEqual(CT.estimate_messages_tokens([]), 0)

    def test_tool_arguments_are_counted(self):
        args = {"path": "/x" * 500}
        with_args = CT.estimate_message_tokens(
            {"role": "tool_call", "tool": "read_file", "args": args})
        without = CT.estimate_message_tokens(
            {"role": "tool_call", "tool": "read_file"})
        self.assertGreater(with_args, without)

    def test_json_string_arguments_are_counted(self):
        """The wire form is a string; the stored form is a dict."""
        as_dict = CT.estimate_message_tokens(
            {"role": "tool_call", "tool": "t", "args": {"k": "v" * 200}})
        as_text = CT.estimate_message_tokens(
            {"role": "tool_call", "tool": "t", "arguments": '{"k": "vvvvv"}'})
        self.assertGreater(as_dict, 0)
        self.assertGreater(as_text, 0)

    def test_reasoning_content_is_counted(self):
        """It is replayed on every turn — it has to be in the estimate."""
        base = CT.estimate_message_tokens({"role": "assistant", "text": "hi"})
        with_reasoning = CT.estimate_message_tokens(
            {"role": "assistant", "text": "hi", "reasoning_content": "x" * 400})
        self.assertGreater(with_reasoning, base)

    def test_system_prompt_and_tools_count_toward_the_request(self):
        only_messages = CT.estimate_request_tokens(
            "", [{"role": "user", "content": "hello"}], [])
        with_prompt = CT.estimate_request_tokens(
            "you are a helpful assistant" * 20,
            [{"role": "user", "content": "hello"}], [])
        self.assertGreater(with_prompt, only_messages)
        with_tools = CT.estimate_request_tokens(
            "", [{"role": "user", "content": "hello"}],
            [{"type": "function", "function": {"name": "read_file",
                                               "description": "x" * 500}}])
        self.assertGreater(with_tools, only_messages)


# ── the trimmer ──────────────────────────────────────────────────────────

def orphan_problems(entries):
    """Tool entries whose partner went missing — providers reject these.

    Both storage shapes are checked: ours (tool_call → tool_result) and the
    normalized OpenAI one (assistant carrying tool_calls → role: tool keyed by
    tool_call_id).
    """
    problems = []
    pending_call = False
    pending_ids = set()
    for m in entries:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        if role == "tool_call":
            pending_call = True
        elif role == "tool_result":
            pending_call = False
        elif role == "tool":
            tid = str(m.get("tool_call_id") or "")
            if tid and tid in pending_ids:
                pending_ids.discard(tid)
            else:
                problems.append("tool message without its call")
            continue
        else:
            if pending_call:
                problems.append("tool_call whose result never arrived")
            pending_call = False
            if pending_ids:
                problems.append("assistant calls whose answers were dropped")
                pending_ids = set()
        for call in m.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("id"):
                pending_ids.add(str(call["id"]))
    if pending_call:
        problems.append("trailing tool_call")
    if pending_ids:
        problems.append("trailing unanswered calls")
    return problems


def conversation(turns=6, tool_every=2):
    """A tool-heavy transcript: user / assistant / call / result, repeated."""
    msgs = [{"role": "system", "text": "system prompt " * 20}]
    for i in range(turns):
        msgs.append({"role": "user", "text": "question %d %s" % (i, "x" * 200)})
        msgs.append({"role": "assistant", "text": "answer %d %s" % (i, "y" * 400)})
        if tool_every and i % tool_every == 0:
            msgs.append({"role": "tool_call", "tool": "read_file",
                         "args": {"path": "/tmp/%d" % i}})
            msgs.append({"role": "tool_result", "tool": "read_file",
                         "text": "file body %d %s" % (i, "z" * 600)})
    return msgs


class TrimTests(unittest.TestCase):
    def test_fits_transcript_is_returned_untouched(self):
        msgs = [{"role": "user", "text": "hello"}]
        out = CT.trim_messages(msgs, limit=8000)
        self.assertFalse(out["trimmed"])
        self.assertEqual(out["messages"], msgs)
        self.assertEqual(out["dropped"], 0)

    def test_caller_list_is_never_mutated(self):
        msgs = conversation()
        snapshot = [dict(m) for m in msgs]
        CT.trim_messages(msgs, limit=1500)
        self.assertEqual(msgs, snapshot)

    def test_result_is_a_deep_copy(self):
        msgs = [{"role": "user", "text": "hello", "args": {"a": ["x"]}}]
        out = CT.trim_messages(msgs, limit=8000)
        out["messages"][0]["args"]["a"].append("mutated")
        self.assertEqual(msgs[0]["args"]["a"], ["x"])

    def test_over_budget_history_is_trimmed_into_the_window(self):
        out = CT.trim_messages(conversation(), limit=2000)
        self.assertTrue(out["trimmed"])
        self.assertLess(out["after"], out["before"])
        self.assertLessEqual(out["after"], 2000 - CT.RESERVE)

    def test_system_prompt_survives(self):
        out = CT.trim_messages(conversation(), limit=1800)
        self.assertEqual(out["messages"][0]["role"], "system")

    def test_current_turn_survives(self):
        msgs = conversation()
        out = CT.trim_messages(msgs, limit=1800)
        self.assertEqual(out["messages"][-1], msgs[-1])

    def test_old_material_goes_before_the_current_turn(self):
        msgs = conversation()
        out = CT.trim_messages(msgs, limit=1800)
        kept = [m.get("text") or m.get("tool") for m in out["messages"]]
        self.assertLess(len(out["messages"]), len(msgs))
        self.assertTrue(any(m == msgs[-1].get("text") for m in kept))

    def test_no_dangling_tool_entries(self):
        """A tool_result without its call is what draws a provider 400."""
        out = CT.trim_messages(conversation(), limit=1600)
        self.assertEqual(orphan_problems(out["messages"]), [])


class OrphanTests(unittest.TestCase):
    """`_drop_orphans` — the invariant providers enforce on tool pairing."""

    def test_result_without_its_call_is_dropped(self):
        out = CT._drop_orphans([
            {"role": "user", "text": "hi"},
            {"role": "tool_result", "tool": "read_file", "text": "body"},
        ])
        self.assertEqual([m["role"] for m in out], ["user"])

    def test_call_without_its_result_is_dropped(self):
        out = CT._drop_orphans([
            {"role": "user", "text": "hi"},
            {"role": "tool_call", "tool": "read_file", "args": {}},
            {"role": "user", "text": "next"},
        ])
        self.assertEqual([m["role"] for m in out], ["user", "user"])

    def test_our_pair_survives(self):
        msgs = [{"role": "tool_call", "tool": "t", "args": {}},
                {"role": "tool_result", "tool": "t", "text": "ok"}]
        self.assertEqual(CT._drop_orphans(msgs), msgs)

    def test_normalized_pair_survives(self):
        msgs = [
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "function": {"name": "t",
                                                      "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
        self.assertEqual(CT._drop_orphans(msgs), msgs)

    def test_normalized_answer_without_its_call_is_dropped(self):
        out = CT._drop_orphans([
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "c9", "content": "ok"},
        ])
        self.assertEqual([m["role"] for m in out], ["user"])

    def test_assistant_with_unanswered_calls_is_dropped(self):
        out = CT._drop_orphans([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "function": {"name": "t",
                                                      "arguments": "{}"}}]},
            {"role": "assistant", "content": "never mind"},
        ])
        self.assertEqual([m.get("content") for m in out], ["hi", "never mind"])

    def test_middle_history_is_summarized_when_dropping_is_not_enough(self):
        out = CT.trim_messages(conversation(turns=10), limit=1500)
        self.assertGreaterEqual(out["summarized"], 1)
        self.assertTrue(any(m.get("summary") for m in out["messages"]))

    def test_summary_is_not_dropped_in_favour_of_the_current_turn(self):
        out = CT.trim_messages(conversation(turns=10), limit=1500)
        if out["summarized"]:
            self.assertTrue(any(m.get("summary") for m in out["messages"]))

    def test_custom_summariser_is_used(self):
        seen = []

        def summariser(entries):
            seen.append(len(entries))
            return "SUMMARY"

        out = CT.trim_messages(conversation(turns=10), limit=1500,
                               summarize=summariser)
        self.assertTrue(seen)
        self.assertTrue(any("SUMMARY" in (m.get("text") or "")
                            for m in out["messages"]))

    def test_broken_summariser_falls_back_to_dropping(self):
        def summariser(entries):
            raise RuntimeError("summarizer down")

        out = CT.trim_messages(conversation(turns=10), limit=1500,
                               summarize=summariser)
        self.assertLessEqual(out["after"], 1500 - CT.RESERVE)
        self.assertEqual(out["summarized"], 0)

    def test_over_budget_is_reported_not_hidden(self):
        """System prompt + current turn alone past the window."""
        msgs = [{"role": "system", "text": "s" * 4000},
                {"role": "user", "text": "u" * 4000}]
        out = CT.trim_messages(msgs, limit=1200)
        self.assertTrue(out["over_budget"])

    def test_zero_budget_is_reported(self):
        out = CT.trim_messages(conversation(), limit=100)
        self.assertTrue(out["over_budget"])

    def test_reserve_is_honoured(self):
        out = CT.trim_messages(conversation(), limit=2000, reserve=500)
        self.assertLessEqual(out["after"], 1500)

    def test_empty_transcript(self):
        out = CT.trim_messages([], limit=8000)
        self.assertEqual(out["messages"], [])
        self.assertFalse(out["trimmed"])

    def test_removed_pct_is_reported(self):
        out = CT.trim_messages(conversation(), limit=1800)
        self.assertGreater(out["removed_pct"], 0)
        self.assertLessEqual(out["removed_pct"], 100)

    def test_normalized_openai_tool_pairs_survive_trimming(self):
        """The wire shape, trimmed: an assistant's call may not outlive it."""
        msgs = [
            {"role": "system", "content": "sys " * 50},
            {"role": "user", "content": "old question " + "x" * 300},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "function": {"name": "read_file",
                                                      "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "read_file",
             "content": "body " + "y" * 400},
            {"role": "assistant", "content": "answer " + "z" * 400},
            {"role": "user", "content": "now question"},
        ]
        out = CT.trim_messages(msgs, limit=1200)
        self.assertTrue(out["trimmed"])
        self.assertEqual(orphan_problems(out["messages"]), [])
        self.assertEqual(out["messages"][-1]["role"], "user")


class SummaryTests(unittest.TestCase):
    def test_local_summary_needs_no_model(self):
        text = CT.local_summary(conversation(turns=3))
        self.assertIn("Earlier conversation", text)
        self.assertIn("User asked", text)
        self.assertIn("Tools used", text)

    def test_local_summary_of_nothing_is_empty(self):
        self.assertEqual(CT.local_summary([]), "")

    def test_burmese_survives_the_summary(self):
        text = CT.local_summary([{"role": "user", "text": BURMESE}])
        self.assertIn("ကိုကြီး", text)


if __name__ == "__main__":
    unittest.main()