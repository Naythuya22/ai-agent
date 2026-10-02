"""Phase 0 verification: Config + LLM + Context size."""
from __future__ import annotations

import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import Config, CONTEXT_OPTIONS
from core.llm import LLM, LLMError


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main() -> int:
    section("Phase 0 Test — Config + LLM + Context Size")

    # ---- Test 1: Config load ----
    print("\n[Test 1] Config load")
    cfg = Config.load()
    print(f"  model_path: {Path(cfg.model_path).name}")
    print(f"  backend:    {cfg.backend}")
    print(f"  context:    {cfg.context_size} ({cfg.context_size // 1024}K)")
    errors = cfg.validate()
    if errors:
        print(f"  ❌ VALIDATION ERRORS:")
        for e in errors:
            print(f"     - {e}")
        return 1
    print("  ✅ OK")

    # ---- Test 2: Load engine ----
    print(f"\n[Test 2] LLM load (context={cfg.context_size})")
    llm = LLM(cfg)
    try:
        t0 = time.time()
        llm.load()
    except LLMError as e:
        print(f"  ❌ FAIL: {e}")
        return 2
    print(f"  ✅ Loaded in {llm.load_time:.2f}s")
    print(f"  Backend used: {llm.backend_used}")
    print(f"  is_loaded: {llm.is_loaded}")

    # ---- Test 3: Simple prompt ----
    print("\n[Test 3] Simple prompt")
    try:
        t0 = time.time()
        resp = llm.ask("Reply with exactly one word: hello")
        elapsed = time.time() - t0
    except (LLMError, ValueError) as e:
        print(f"  ❌ FAIL: {e}")
        llm.unload()
        return 3
    print(f"  Prompt:   'Reply with exactly one word: hello'")
    print(f"  Response: {resp.strip()[:120]}")
    print(f"  Time:     {elapsed:.2f}s")

    # ---- Test 4: Empty prompt guard ----
    print("\n[Test 4] Empty prompt guard")
    try:
        llm.ask("   ")
        print("  ❌ FAIL: empty prompt should raise ValueError")
        llm.unload()
        return 4
    except ValueError:
        print("  ✅ Correctly rejected empty prompt")

    # ---- Test 5: Unload ----
    print("\n[Test 5] Unload")
    llm.unload()
    print(f"  is_loaded: {llm.is_loaded}")
    if llm.is_loaded:
        print("  ❌ FAIL: still loaded")
        return 5
    print("  ✅ OK")

    # ---- Summary ----
    section("Phase 0: ✅ PASS")
    print(f"\nCurrent context: {cfg.context_size} ({cfg.context_size // 1024}K)")
    print("\nNext steps:")
    print("  1. python settings.py  → change context size")
    print("  2. python test_phase0.py  → verify new size works")
    print(f"\nValid options: {CONTEXT_OPTIONS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
