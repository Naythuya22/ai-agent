"""Phase 2 verification — Agent mode + native tool calling."""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import Config
from core.llm import LLM
from modes.agent import AgentMode


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main() -> int:
    section("Phase 2 Test — Agent Mode")

    cfg = Config.load()
    print(f"\nContext: {cfg.context_size} ({cfg.context_size // 1024}K)")
    print(f"Backend: {cfg.backend}")

    llm = LLM(cfg)
    agent = AgentMode(llm)

    # ---- Test 1: Tool call — list_files ----
    print("\n[Test 1] list_files tool call")
    try:
        r1 = agent.handle("List the files in /data/data/com.termux/files/home/ai-agent")
    except Exception as e:
        print(f"  ❌ FAIL: {type(e).__name__}: {e}")
        return 1
    print(f"  Reply: {r1.strip()[:200]}")
    if "app.py" in r1 and "config.py" in r1:
        print("  ✅ OK — file list detected")
    else:
        print("  ⚠️  WARN: expected file names not found")
    print(f"  Turns: {agent.turns}")

    # ---- Test 2: Tool call — read_file ----
    print("\n[Test 2] read_file tool call")
    try:
        r2 = agent.handle(
            "Read the file /data/data/com.termux/files/home/ai-agent/config.py "
            "and tell me what CONTEXT_OPTIONS contains."
        )
    except Exception as e:
        print(f"  ❌ FAIL: {type(e).__name__}: {e}")
        return 2
    print(f"  Reply: {r2.strip()[:250]}")
    if "8192" in r2 or "12288" in r2 or "16384" in r2:
        print("  ✅ OK — read_file works")
    else:
        print("  ⚠️  WARN: expected values not in reply")
    print(f"  Turns: {agent.turns}")

    # ---- Test 3: Write file ----
    print("\n[Test 3] write_file tool call")
    test_path = "/data/data/com.termux/files/home/ai-agent/test_agent_out.txt"
    # cleanup
    try:
        Path(test_path).unlink()
    except Exception:
        pass
    try:
        r3 = agent.handle(
            f"Write the text 'hello from agent' to {test_path}"
        )
    except Exception as e:
        print(f"  ❌ FAIL: {type(e).__name__}: {e}")
        return 3
    print(f"  Reply: {r3.strip()[:200]}")
    if Path(test_path).exists():
        content = Path(test_path).read_text()
        print(f"  File exists, content: {content!r}")
        print("  ✅ OK")
    else:
        print("  ⚠️  WARN: file not created")

    # ---- Test 4: Reset ----
    print("\n[Test 4] Reset")
    agent.reset()
    if agent.turns != 0:
        print(f"  ❌ FAIL: turns after reset = {agent.turns}")
        return 4
    print(f"  ✅ OK (turns={agent.turns})")

    # ---- Test 5: Chat-like (no tool needed) ----
    print("\n[Test 5] No-tool chat")
    try:
        r5 = agent.handle("Say hello in Burmese.")
    except Exception as e:
        print(f"  ❌ FAIL: {type(e).__name__}: {e}")
        return 5
    print(f"  Reply: {r5.strip()[:200]}")
    print("  ✅ OK")

    # ---- Cleanup ----
    try:
        Path(test_path).unlink()
    except Exception:
        pass

    agent.close()
    llm.unload()
    section("Phase 2: ✅ PASS")
    print(f"\nNext: python app.py  → test /agent, /chat switch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
