"""Android intent tools for agent mode.

Uses Termux's `am` command (com.termux app) to send Android Intents.
No Termux:API app required for these specific intents.

Most intents open the target app with pre-filled data — the user must
tap the confirm button (e.g. ✓ on Clock). Tools return instructions.
"""
from __future__ import annotations

import shutil
import subprocess
from datetime import datetime

# ── Config ──
_AM = shutil.which("am") or "am"
_TIMEOUT = 10

# AOSP Clock standard (android.intent.extra.alarm.DAYS):
#   Calendar.SUNDAY=1, MON=2, TUE=4, WED=8, THU=16, FRI=32, SAT=64
# NOTE: Samsung OneUI ignores this extra; repeat must be set manually.
_DAY_MAP = {
    "sun": 1, "mon": 2, "tue": 4, "wed": 8,
    "thu": 16, "fri": 32, "sat": 64,
}


def _parse_days(days) -> int:
    """Parse day spec (list, string, or int) → bitmask 0-127."""
    if days is None or days == "" or days == 0:
        return 0
    if isinstance(days, int):
        return max(0, min(127, days))
    if isinstance(days, str):
        parts = [p.strip().lower()[:3] for p in days.split(",") if p.strip()]
    elif isinstance(days, (list, tuple)):
        parts = [str(p).strip().lower()[:3] for p in days if str(p).strip()]
    else:
        return 0
    mask = 0
    for p in parts:
        if p in _DAY_MAP:
            mask |= _DAY_MAP[p]
    return mask


def _decode_days(mask: int) -> str:
    if mask == 0:
        return "once"
    names = [k for k, v in _DAY_MAP.items() if mask & v]
    return ",".join(names)


def _run_am(action: str, extras: list) -> tuple:
    """Run `am start` with action + extras. Returns (ok, output)."""
    try:
        r = subprocess.run(
            [_AM, "start", "-a", action] + extras,
            capture_output=True, text=True, timeout=_TIMEOUT,
        )
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        if r.returncode != 0:
            return False, out or f"am exit {r.returncode}"
        return True, out
    except subprocess.TimeoutExpired:
        return False, f"am timed out after {_TIMEOUT}s"
    except FileNotFoundError:
        return False, "am command not found (Termux required)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def set_alarm(hour: int, minute: int, message: str = "Alarm", days: str = "") -> str:
    """Set an Android alarm via system intent.

    Args:
        hour: Hour in 24-hour format (0-23).
        minute: Minute (0-59).
        message: Label shown on the alarm (match user's language).
        days: Comma-separated day abbreviations for recurring alarms. Leave
            empty for one-time. Example: "sat" or "mon,wed,fri". Valid:
            mon, tue, wed, thu, fri, sat, sun.
    """
    try:
        h = int(hour)
        m = int(minute)
    except Exception:
        return f"Error: hour and minute must be integers, got {hour!r}, {minute!r}"
    if not (0 <= h <= 23):
        return f"Error: hour must be 0-23, got {h}"
    if not (0 <= m <= 59):
        return f"Error: minute must be 0-59, got {m}"

    mask = _parse_days(days)
    msg = (message or "Alarm").strip()[:100] or "Alarm"

    extras = [
        "--ei", "android.intent.extra.alarm.HOUR", str(h),
        "--ei", "android.intent.extra.alarm.MINUTES", str(m),
        "--es", "android.intent.extra.alarm.MESSAGE", msg,
    ]
    if mask > 0:
        extras += ["--ei", "android.intent.extra.alarm.DAYS", str(mask)]

    ok, out = _run_am("android.intent.action.SET_ALARM", extras)
    if not ok:
        return f"Error: could not send alarm intent — {out}"

    hhmm = f"{h:02d}:{m:02d}"
    when = _decode_days(mask)
    note = (
        "Clock app ပွင့်နေတယ် — အချိန် မှန်ကန်ရင် ✓ (checkmark) ကို tap လုပ်ပါ။"
    )
    if mask > 0:
        note += (
            f" ⚠️ Repeat ({when}) ကို Samsung Clock က auto သတ်မှတ်ပေးမှာ မဟုတ်လို့၊ "
            f"Repeat ကို ကိုယ်တိုင် ရွေးပေးပါ။"
        )
    return (
        f"OK: alarm intent sent ({hhmm}, {when}) label='{msg}'. {note}"
    )


def set_timer(seconds: int, message: str = "Timer") -> str:
    """Set a countdown timer on the device via system intent.

    Args:
        seconds: Timer duration in seconds (1 to 86400 = 24 hours).
        message: Label shown on the timer (match user's language).
    """
    try:
        secs = int(seconds)
    except Exception:
        return f"Error: seconds must be an integer, got {seconds!r}"
    if secs <= 0:
        return "Error: seconds must be greater than 0"
    if secs > 86400:
        return "Error: seconds must be 86400 or less (24 hours)"

    msg = (message or "Timer").strip()[:100] or "Timer"
    extras = [
        "--ei", "android.intent.extra.alarm.LENGTH", str(secs),
        "--es", "android.intent.extra.alarm.MESSAGE", msg,
    ]
    ok, out = _run_am("android.intent.action.SET_TIMER", extras)
    if not ok:
        return f"Error: could not send timer intent — {out}"

    return (
        f"OK: timer intent sent ({secs}s) label='{msg}'. "
        f"Clock app (Timer tab) is opening — ask the user to tap the "
        f"play button to start the timer."
    )


def open_url(url: str) -> str:
    """Open a URL in the device's default browser via system intent.

    Args:
        url: Full URL (http:// or https://). Missing scheme defaults to https.
    """
    u = (url or "").strip()
    if not u:
        return "Error: url is required"
    if "://" not in u:
        u = "https://" + u
    if not (u.startswith("http://") or u.startswith("https://")):
        return f"Error: only http/https URLs are allowed, got {u!r}"

    extras = ["-d", u]
    ok, out = _run_am("android.intent.action.VIEW", extras)
    if not ok:
        return f"Error: could not open URL — {out}"
    return f"OK: opened {u} in the default browser."


def create_calendar_event(title: str, start_unix_ts: int, duration_min: int = 60) -> str:
    """Create a calendar event via system intent.

    Args:
        title: Event title (match user's language).
        start_unix_ts: Event start time as Unix timestamp (seconds since 1970).
            Use the local time zone for the target wall-clock time.
        duration_min: Duration in minutes (1-1440). Default 60.
    """
    t = (title or "").strip()
    if not t:
        return "Error: title is required"
    try:
        start = int(start_unix_ts)
    except Exception:
        return f"Error: start_unix_ts must be an integer, got {start_unix_ts!r}"
    try:
        dur = int(duration_min)
    except Exception:
        return f"Error: duration_min must be an integer, got {duration_min!r}"
    if dur <= 0 or dur > 1440:
        return "Error: duration_min must be 1-1440"

    end = start + dur * 60
    extras = [
        "-t", "vnd.android.cursor.dir/event",
        "--el", "beginTime", str(start * 1000),
        "--el", "endTime", str(end * 1000),
        "--es", "title", t[:100],
    ]
    ok, out = _run_am("android.intent.action.INSERT", extras)
    if not ok:
        return f"Error: could not create calendar event — {out}"

    dt = datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M")
    return (
        f"OK: calendar event intent sent ('{t}' at {dt}, {dur} min). "
        f"Calendar app is opening — ask the user to save the event."
    )


ALL_TOOLS = [set_alarm, set_timer, open_url, create_calendar_event]
__all__ = [
    "set_alarm", "set_timer", "open_url", "create_calendar_event", "ALL_TOOLS",
]
