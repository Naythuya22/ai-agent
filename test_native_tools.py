"""Test LiteRT native tool calling API."""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import Config
from core.llm import LLM
from tools.file import list_files

# Import native helper
try:
    from litert_lm import tool_from_function
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---- Define a simple test tool ----
def get_time() -> str:
    """Return the current time as a string."""
    import time
    return time.strftime("%H:%M:%S")


def main() -> int:
    section("Native Tool Calling Test")

    if not HAS_NATIVE:
        print("❌ tool_from_function not available")
        return 1
    print("✅ tool_from_function available")

    # ---- Convert functions to tools ----
    try:
        t_time = tool_from_function(get_time)
        t_list = tool_from_function(list_files)
        print(f"✅ Converted: get_time → {type(t_time).__name__}")
        print(f"✅ Converted: list_files → {type(t_list).__name__}")
    except Exception as e:
        print(f"❌ Convert failed: {type(e).__name__}: {e}")
        return 2

    # ---- Inspect tool description ----
    print("\n[Tool description: get_time]")
    try:
        desc = t_time.get_tool_description()
        import json
        print(json.dumps(desc, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"  (no get_tool_description: {e})")

    # ---- Load engine ----
    section("Load Engine")
    cfg = Config.load()
    llm = LLM(cfg)
    llm.load()
    print(f"✅ Loaded | backend={llm.backend_used} | ctx={cfg.context_size}")

    # ---- Test 1: No-tool chat ----
    section("Test 1 — No-tool chat")
    try:
        conv = llm._engine.create_conversation(
            tools=[t_time, t_list],
            automatic_tool_calling=True,
            system_message="You are a helpful assistant. Use tools when needed.",
        )
        resp = conv.send_message("Hello, how are you?")
        print(f"Type: {type(resp).__name__}")
        print(f"Keys: {list(resp.keys()) if hasattr(resp, 'keys') else 'N/A'}")
        print(f"Raw: {resp}")
        conv.__exit__(None, None, None)
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}")

    # ---- Test 2: Should call get_time ----
    section("Test 2 — Tool call (get_time)")
    try:
        conv = llm._engine.create_conversation(
            tools=[t_time, t_list],
            automatic_tool_calling=True,
            system_message="You are a helpful assistant. Use tools when needed.",
        )
        resp = conv.send_message("What time is it? Use the get_time tool.")
        print(f"Type: {type(resp).__name__}")
        print(f"Keys: {list(resp.keys()) if hasattr(resp, 'keys') else 'N/A'}")
        print(f"Raw: {resp}")
        conv.__exit__(None, None, None)
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}")

    # ---- Test 3: Should call list_files ----
    section("Test 3 — Tool call (list_files)")
    try:
        conv = llm._engine.create_conversation(
            tools=[t_time, t_list],
            automatic_tool_calling=True,
            system_message="You are a helpful assistant. Use tools when needed.",
        )
        resp = conv.send_message(
            "List the files in /data/data/com.termux/files/home/ai-agent using the list_files tool."
        )
        print(f"Type: {type(resp).__name__}")
        print(f"Keys: {list(resp.keys()) if hasattr(resp, 'keys') else 'N/A'}")
        print(f"Raw: {resp}")
        conv.__exit__(None, None, None)
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}")

    # ---- Cleanup ----
    llm.unload()
    section("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
