"""Tool schemas — every tool that leaves AgentBackend must be JSON.

The bug: `AgentBackend.step()` handed LiteRT's live `tool_from_function`
objects straight to the adapter, so a Cloud Only turn with tools died with

    TypeError: Object of type _FunctionTool is not JSON serializable

before the request was ever built. The old tests could not see it: the test
stub returns `{"name": ..., "fn": ...}` (a dict), and the cloud integration
tests set `_tools = []`. Here the tools are shaped the way the phone shapes
them, and the wire is dumped with `json.dumps`.

Run:  sh tests/run.sh -p "test_tool_schema.py"
"""
from __future__ import annotations

import json
import unittest

from _harness import bootstrap

bootstrap()

from core import agent_backend as AB  # noqa: E402
from core.agent_backend import (  # noqa: E402
    AnthropicAdapter, GeminiAdapter, OpenAIAdapter,
    _tool_schemas, function_to_schema, tool_schema,
)
from core.key_manager import KeyManager  # noqa: E402
from tools import ALL_TOOLS  # noqa: E402
from tools.file import list_files  # noqa: E402

ALPHA = "sk-tool-schema-aaaaaaaaaaaaaaaa1111"


class FunctionTool:
    """What `litert_lm.tool_from_function` returns on the phone.

    A plain object with no dict behaviour — sending it to `json.dumps` is
    exactly what raised the TypeError.
    """

    def __init__(self, fn):
        self.fn = fn


class DescribeTool:
    """A tool that knows how to describe itself (the `get_tool_description`
    path litert_lm also uses)."""

    def __init__(self, name, description="", parameters=None):
        self._desc = {"name": name, "description": description,
                      "parameters": parameters or {"type": "object",
                                                   "properties": {}}}

    def get_tool_description(self):
        return dict(self._desc)


class SchemaShapeTests(unittest.TestCase):
    def test_bare_function_is_described_from_signature_and_docstring(self):
        s = function_to_schema(list_files)
        self.assertEqual(s["type"], "function")
        fn = s["function"]
        self.assertEqual(fn["name"], "list_files")
        self.assertIn("directory", fn["description"])
        params = fn["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertEqual(params["required"], ["directory"])
        self.assertEqual(params["properties"]["directory"]["type"], "string")
        # The Args: block is what tells the model what to pass.
        self.assertIn("directory path", params["properties"]["directory"]
                      .get("description", ""))

    def test_defaults_become_optional_parameters_with_a_type(self):
        def greet(name: str, times: int = 2, loud: bool = False) -> str:
            """Say hello.

            Args:
                name: Who to greet.
                times: How many times.
                loud: Shout it.
            """
            return name

        p = function_to_schema(greet)["function"]["parameters"]
        self.assertEqual(p["required"], ["name"])
        self.assertEqual(p["properties"]["times"]["type"], "integer")
        self.assertEqual(p["properties"]["times"]["default"], 2)
        self.assertEqual(p["properties"]["loud"]["type"], "boolean")
        self.assertEqual(p["properties"]["loud"]["description"], "Shout it.")

    def test_test_stub_dict_with_a_wrapped_function(self):
        s = tool_schema({"name": "list_files", "fn": list_files})
        self.assertEqual(s["function"]["name"], "list_files")

    def test_openai_dict_passes_through_untouched(self):
        original = {"type": "function",
                    "function": {"name": "x", "description": "d",
                                 "parameters": {"type": "object",
                                                "properties": {}}}}
        self.assertIs(tool_schema(original), original)
        # `{"function": ...}` without a type is wrapped, never rewritten.
        fn_only = {"function": dict(original["function"])}
        self.assertEqual(tool_schema(fn_only), original)

    def test_object_that_describes_itself(self):
        t = DescribeTool("web_search", "Search the web.")
        s = tool_schema(t)
        self.assertEqual(s["function"]["name"], "web_search")
        self.assertEqual(s["function"]["description"], "Search the web.")

    def test_wrapper_object_falls_back_to_the_wrapped_function(self):
        """The `_FunctionTool` shape: an object with `.fn` and no dict body."""
        s = tool_schema(FunctionTool(list_files))
        self.assertEqual(s["type"], "function")
        self.assertEqual(s["function"]["name"], "list_files")
        self.assertEqual(s["function"]["parameters"]["required"], ["directory"])

    def test_an_object_nothing_can_describe_is_dropped(self):
        self.assertIsNone(tool_schema(object()))
        self.assertIsNone(tool_schema(None))
        self.assertEqual(_tool_schemas([object(), None]), [])


class WireTests(unittest.TestCase):
    def test_every_project_tool_survives_the_wire(self):
        tools = [FunctionTool(fn) for fn in ALL_TOOLS]
        schemas = _tool_schemas(tools)
        self.assertEqual(len(schemas), len(tools),
                         "a tool was dropped: %s" % sorted(
                             {s["function"]["name"] for s in schemas}))
        payload = json.dumps(schemas, ensure_ascii=False)
        self.assertIsInstance(payload, str)
        for s in schemas:
            self.assertEqual(s["type"], "function")
            self.assertIsInstance(s["function"]["name"], str)
            self.assertEqual(s["function"]["parameters"]["type"], "object")

    def test_a_schema_that_cannot_be_serialised_is_dropped_not_raised(self):
        bad = DescribeTool("broken")
        # `default` is not JSON — the whole schema must go, not the turn.
        bad._desc["parameters"] = {"type": "object",
                                   "properties": {"x": {"default": object()}}}
        good = FunctionTool(list_files)
        schemas = _tool_schemas([good, bad])
        self.assertEqual([s["function"]["name"] for s in schemas],
                         ["list_files"])
        json.dumps(schemas)   # no exception = the guarantee

    def test_every_adapter_body_with_tools_json_dumps(self):
        tools = _tool_schemas([FunctionTool(fn) for fn in ALL_TOOLS])
        messages = [{"role": "user", "content": "list my files"}]
        for adapter in (OpenAIAdapter(), GeminiAdapter(),
                        AnthropicAdapter()):
            url, body, headers = adapter.request(
                key=ALPHA, model="m", system="s", messages=messages,
                tools=tools, max_tokens=64)
            self.assertIsInstance(body, dict)
            self.assertIn("tools", body, adapter.__class__.__name__)
            json.dumps(body, ensure_ascii=False)   # the actual wire encoding


class StepSendsToolsTest(unittest.TestCase):
    """The original TypeError, at the place it was raised."""

    def setUp(self):
        self.km = KeyManager()
        if self.km.path.exists():
            self.km.path.unlink()
        self.km.add_key("openai", ALPHA, "alpha")
        cfg = self.km.load()
        cfg["prefer_cloud"] = True
        cfg["cost_ack"] = True
        self.km.set_mode("cloud")
        self.km.save()
        self.calls = []

        def transport(url, body, headers, timeout=30):
            self.calls.append((url, body, headers))
            return {"choices": [{"message": {"content": "ANSWER_OK"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1}}

        self.be = AB.AgentBackend(keys=self.km, net_state=lambda: "online",
                                  http=transport)

    def test_step_json_encodes_the_tools_it_was_given(self):
        tools = [FunctionTool(fn) for fn in ALL_TOOLS]
        step = self.be.step(system="s",
                            messages=[{"role": "user", "content": "hi"}],
                            tools=tools)
        self.assertEqual(step.text, "ANSWER_OK")
        self.assertEqual(len(self.calls), 1)
        url, body, headers = self.calls[0]
        # The failure was raised *building* this body, so the assertions that
        # matter are these two.
        json.dumps(body, ensure_ascii=False)
        sent = body["tools"]
        self.assertEqual(len(sent), len(tools))
        names = sorted(s["function"]["name"] for s in sent)
        self.assertEqual(names, sorted(fn.__name__ for fn in ALL_TOOLS))

    def test_step_with_no_tools_sends_no_tools_field(self):
        self.be.step(system="s",
                     messages=[{"role": "user", "content": "hi"}],
                     tools=[])
        url, body, headers = self.calls[0]
        self.assertNotIn("tools", body)


if __name__ == "__main__":
    unittest.main()
