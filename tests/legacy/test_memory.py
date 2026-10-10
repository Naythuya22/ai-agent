"""Unit test for Memory class (Phase 3C-1)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from memory import Memory


def section(t: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {t}")
    print("=" * 60)


def main() -> int:
    section("Phase 3C-1 — Memory Unit Test")

    # Use temp file (isolated from real memory.json)
    with tempfile.TemporaryDirectory() as td:
        test_path = Path(td) / "memory.json"

        # ---- Test 1: Empty init ----
        print("\n[Test 1] Empty init")
        m = Memory.load(test_path)
        s = m.stats()
        if sum(s.values()) != 0:
            print(f"  ❌ FAIL: expected empty, got {s}")
            return 1
        print(f"  ✅ OK stats={s}")

        # ---- Test 2: add_fact + save ----
        print("\n[Test 2] add_fact + save")
        ok, reason = m.add_fact("User is learning Python", source="test")
        if not ok:
            print(f"  ❌ FAIL: add_fact returned ({ok}, {reason!r})")
            return 2
        if not test_path.exists():
            print("  ❌ FAIL: file not created")
            return 2
        print("  ✅ OK — file created")

        # ---- Test 3: Reload persistence ----
        print("\n[Test 3] Reload persistence")
        m2 = Memory.load(test_path)
        if m2.stats()["facts"] != 1:
            print(f"  ❌ FAIL: facts={m2.stats()['facts']}")
            return 3
        print(f"  ✅ OK facts=1")

        # ---- Test 4: Dedup ----
        print("\n[Test 4] Dedup facts")
        ok2, reason2 = m2.add_fact("User is learning Python")
        if ok2:
            print("  ❌ FAIL: duplicate accepted")
            return 4
        if reason2 != "exact duplicate":
            print(f"  ❌ FAIL: expected exact duplicate, got {reason2!r}")
            return 4
        print("  ✅ OK duplicate rejected")

        # ---- Test 5: set_identity ----
        print("\n[Test 5] set_identity")
        m2.set_identity("name", "မောင်မောင်")
        m3 = Memory.load(test_path)
        if m3.data["identity"].get("name") != "မောင်မောင်":
            print(f"  ❌ FAIL: identity={m3.data['identity']}")
            return 5
        print("  ✅ OK")

        # ---- Test 6: get_context ----
        print("\n[Test 6] get_context")
        ctx = m3.get_context()
        print(f"  Context:\n{ctx}")
        if "မောင်မောင်" not in ctx or "Python" not in ctx:
            print("  ❌ FAIL: missing content")
            return 6
        print("  ✅ OK")

        # ---- Test 7: forget ----
        print("\n[Test 7] forget 'Python'")
        n = m3.forget("Python")
        if n < 1:
            print(f"  ❌ FAIL: forget count={n}")
            return 7
        if m3.stats()["facts"] != 0:
            print(f"  ❌ FAIL: facts not removed: {m3.stats()}")
            return 7
        print(f"  ✅ OK removed {n} items")

        # ---- Test 8: Identity preserved after forget ----
        print("\n[Test 8] Identity preserved")
        if m3.data["identity"].get("name") != "မောင်မောင်":
            print(f"  ❌ FAIL: identity lost")
            return 8
        print("  ✅ OK")

        # ---- Test 9: Empty forget ----
        print("\n[Test 9] forget empty keyword")
        n = m3.forget("")
        if n != 0:
            print(f"  ❌ FAIL: forget('') returned {n}")
            return 9
        print("  ✅ OK returned 0")

        # ---- Test 10: Corrupt file recovery ----
        print("\n[Test 10] Corrupt file recovery")
        test_path.write_text("{not json at all", encoding="utf-8")
        m4 = Memory.load(test_path)
        if sum(m4.stats().values()) != 0:
            print(f"  ❌ FAIL: corrupt recovery gave {m4.stats()}")
            return 10
        print("  ✅ OK — recovered to defaults")

    section("Phase 3C-1: ✅ PASS (10/10)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
