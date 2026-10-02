"""Settings menu — context size slider (8K / 12K / 16K)."""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import Config, CONTEXT_OPTIONS, CONTEXT_LABELS

# Try questionary; fallback to plain input
try:
    import questionary
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False


def select_context_questionary(current: int) -> int | None:
    """Arrow-key slider (up/down). Returns selected value or None."""
    choices = [
        questionary.Choice(
            title=CONTEXT_LABELS[n],
            value=n,
            checked=(n == current),
        )
        for n in CONTEXT_OPTIONS
    ]
    return questionary.select(
        "Context Size ကို ↑↓ arrow နဲ့ ရွေးပါ:",
        choices=choices,
        default=current,
    ).ask()


def select_context_text(current: int) -> int | None:
    """Fallback text menu."""
    print("\nContext Size:")
    for i, n in enumerate(CONTEXT_OPTIONS, 1):
        marker = " ← current" if n == current else ""
        print(f"  [{i}] {CONTEXT_LABELS[n]}{marker}")
    print("  [q] Cancel")
    raw = input("Select (1/2/3/q): ").strip().lower()
    if raw in ("q", ""):
        return None
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(CONTEXT_OPTIONS):
            return CONTEXT_OPTIONS[idx]
    except ValueError:
        pass
    print("Invalid choice.")
    return None


def main() -> int:
    cfg = Config.load()

    model = cfg.resolve_model_path()
    print("=" * 60)
    print("  ai-agent Settings")
    print("=" * 60)
    print(f"  Model:   {Path(model).name if model else '(not found)'}")
    print(f"  From:    {Path(model).parent if model else '-'}")
    print(f"  Backend: {cfg.backend}")
    print(f"  Context: {cfg.context_size} ({cfg.context_size // 1024}K)")
    print()

    if HAS_QUESTIONARY:
        new_ctx = select_context_questionary(cfg.context_size)
    else:
        print("(questionary not found — using text mode)")
        new_ctx = select_context_text(cfg.context_size)

    if new_ctx is None:
        print("Canceled. No changes.")
        return 0

    if new_ctx == cfg.context_size:
        print(f"No change (still {new_ctx}).")
        return 0

    cfg.context_size = new_ctx
    cfg.save()

    print()
    print(f"✓ Context changed: {new_ctx} ({new_ctx // 1024}K)")
    print("  Next LLM run will use this size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
