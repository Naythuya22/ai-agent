"""Phase 1 verification — Chat mode + multi-turn + commands."""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import Config
from core.llm import LLM, LLMError
from modes.chat import ChatMode


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main() -> int:
    section("Phase 1 Test — Chat Mode")

    cfg = Config.load()
    print(f"\nContext: {cfg.context_size} ({cfg.context_size // 1024}K)")
    print(f"Backend: {cfg.backend}")

    llm = LLM(cfg)
    chat = ChatMode(llm)
    chat.reset()  # ensure clean state (session persistence aware)

    # ---- Test 1: First turn ----
    print("\n[Test 1] First turn")
    try:
        r1 = chat.handle("My name is Aung.")
    except Exception as e:
        print(f"  ❌ FAIL: {type(e).__name__}: {e}")
        return 1
    print(f"  Reply: {r1.strip()[:120]}")
    print(f"  Turns: {chat.turns}")
    if chat.turns != 1:
        print("  ❌ FAIL: turns != 1")
        return 1
    print("  ✅ OK")

    # ---- Test 2: Multi-turn memory ----
    print("\n[Test 2] Multi-turn memory (distraction)")
    try:
        chat.handle("I work as a software engineer.")
        r3 = chat.handle("What is my name?")
    except Exception as e:
        print(f"  ❌ FAIL: {type(e).__name__}: {e}")
        return 2
    print(f"  Reply: {r3.strip()[:160]}")
    if "aung" not in r3.lower():
        print("  ⚠️  WARN: 'Aung' not found (model may be inconsistent)")
    else:
        print("  ✅ Memory retained")
    print(f"  Turns: {chat.turns}")

    # ---- Test 3: Reset ----
    print("\n[Test 3] Reset")
    chat.reset()
    if chat.turns != 0:
        print(f"  ❌ FAIL: turns after reset = {chat.turns}")
        return 3
    print(f"  ✅ OK (turns={chat.turns})")

    # ---- Test 4: Empty guard ----
    print("\n[Test 4] Empty message guard")
    try:
        chat.handle("   ")
        print("  ❌ FAIL: empty should raise ValueError")
        return 4
    except ValueError:
        print("  ✅ Correctly rejected empty")

    # ---- Test 5: Reply after reset (fresh state) ----
    print("\n[Test 5] After reset — model should not know 'Aung'")
    try:
        r5 = chat.handle("What is my name?")
    except Exception as e:
        print(f"  ❌ FAIL: {type(e).__name__}: {e}")
        return 5
    print(f"  Reply: {r5.strip()[:160]}")
    if "aung" in r5.lower():
        print("  ⚠️  WARN: 'Aung' leaked after reset (unexpected)")
    else:
        print("  ✅ Fresh state confirmed")

    # ---- Cleanup ----
    chat.close()
    llm.unload()
    section("Phase 1: ✅ PASS")
    print(f"\nNext: python app.py  → interactive REPL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
