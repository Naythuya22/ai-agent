"""No-litert_lm boot: the app must run cloud-only on a device where the
local engine library is not installed at all.

Proven necessary on a clean Redmi Note 12 (2026-11-18): install.sh
finished "Done", but ``python3 server.py`` died at import —
``core/llm.py`` imported litert_lm unconditionally while no install
path provides it (requirements.txt excludes it, and pip cannot install
it on Termux: no Android wheel, source builds of cmake/ninja/numpy
fail). The Xiaomi never showed it because an October manual install
left the package behind.

The suite's own environment always has the litert_lm stub on sys.path,
so this test re-runs the imports in a subprocess whose interpreter
genuinely lacks the package (the project venv has no litert_lm). On a
machine where litert_lm IS installed the subprocess reports a skip.

Run:  sh tests/run.sh -p "test_no_litert.py"
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from _harness import bootstrap

bootstrap()

ROOT = Path(__file__).resolve().parent.parent

_SCRIPT = textwrap.dedent(
    """
    import sys

    try:
        import litert_lm  # noqa: F401
        print("SKIP: litert_lm is installed in this interpreter")
        sys.exit(42)
    except ImportError:
        pass

    import core.llm as llm_mod
    assert llm_mod.litert_lm is None, "import guard did not engage"

    from core.llm import LLM, LLMError
    try:
        LLM(None).load()
        print("FAIL: load() did not raise without the engine library")
        sys.exit(1)
    except LLMError as e:
        assert "litert" in str(e).lower(), f"unclear load error: {e}"

    import modes.agent

    def _sample(x: int) -> int:
        \"\"\"Sample tool.\"\"\"
        return x

    assert modes.agent.tool_from_function(_sample) is _sample, (
        "fallback wrapper must pass bare functions through")

    from core.agent_backend import function_to_schema
    schema = function_to_schema(_sample)
    assert schema and schema["function"]["name"] == "_sample", schema

    import server
    assert server.app is not None

    print("ABSENCE OK")
    """
)


class TestNoLitertLm(unittest.TestCase):
    def test_cloud_only_without_engine_library(self):
        with tempfile.TemporaryDirectory() as home:
            env = dict(os.environ)
            env["HOME"] = home
            env["PYTHONPATH"] = str(ROOT)
            proc = subprocess.run(
                [sys.executable, "-c", _SCRIPT],
                cwd=str(ROOT), env=env, capture_output=True, text=True,
                timeout=120,
            )
        if proc.returncode == 42:
            self.skipTest("litert_lm is installed in this interpreter")
        self.assertEqual(proc.returncode, 0,
                         f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("ABSENCE OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()
