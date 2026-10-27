"""Video generation tools — Google Flow human-in-the-loop (Release 32).

The phone cannot generate video locally, and video APIs have no free
tier, so v1 drives Google Flow's free web tier with the user in the
loop — the same pattern as tools/android_intent.py: the agent prepares
everything, opens the right place with the data pre-filled, and the
user taps the final buttons (sign-in, Generate, Download).

Workflow:
  1. prepare_video()  — crafts the Flow prompt, records a job, copies
     the prompt to the clipboard when Termux:API is present, and opens
     Flow in the browser. NEVER auto-retries a generation: a duplicate
     submission would spend the user's daily credits twice.
  2. The user generates + downloads in Flow (MP4 lands in Download/).
  3. import_video()   — finds that MP4, copies it into the agent's
     video library (~/.ai-agent/videos/) and marks the job imported.
     The reply then carries a [video:<job_id>] marker the PWA renders
     as an inline player (served by routes/video_api.py).

Measured on-device 2026-10-27 (KoHtet's pilot, Flow free tier):
  - Omni 1.1 Flash speaks Burmese dialogue; Veo 3.1 Lite cannot
    (audio generation fails for Burmese prompts — no charge).
  - Omni credits: 360p 6s=5, 8s=6, 10s=7; 720p 6s=10, 8s=12, 10s=15.
  - Veo 3.1 Lite: flat 10 credits, no dialogue.
  - Failed generations are never charged.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

FLOW_URL = "https://flow.google.com"

# Measured credit costs (see module docstring). Combos that were not
# measured are absent on purpose: the estimate then reports "unknown"
# instead of inventing a number — Flow shows the real cost before the
# user taps Generate anyway.
_OMNI_CREDITS = {
    ("360p", 6): 5, ("360p", 8): 6, ("360p", 10): 7,
    ("720p", 6): 10, ("720p", 8): 12, ("720p", 10): 15,
}
_VEO_LITE_CREDITS = 10

_MODELS = ("omni", "veo_lite")
_DURATIONS = (4, 6, 8, 10)
_RESOLUTIONS = ("360p", "720p")

_JOB_STATES = ("awaiting_human", "imported", "failed", "expired")


def _data_dir() -> Path:
    """The agent's private data dir (~/.ai-agent), created on demand."""
    d = Path.home() / ".ai-agent"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return d


def _jobs_file() -> Path:
    override = os.environ.get("AI_AGENT_VIDEO_JOBS")
    return Path(override) if override else _data_dir() / "video_jobs.json"


def _videos_dir() -> Path:
    override = os.environ.get("AI_AGENT_VIDEOS_DIR")
    d = Path(override) if override else _data_dir() / "videos"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return d


def _load_jobs() -> list:
    try:
        raw = _jobs_file().read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _save_jobs(jobs: list) -> None:
    f = _jobs_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(jobs, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(f)


def _estimate_credits(model: str, resolution: str, duration_s: int):
    """Measured credit cost, or None when that combo was never measured."""
    if model == "veo_lite":
        return _VEO_LITE_CREDITS
    return _OMNI_CREDITS.get((resolution, int(duration_s)))


def _provider_enabled() -> bool:
    try:
        from core.key_manager import KeyManager
        return KeyManager().video_provider() == "flow"
    except Exception:
        return False


def _disabled_msg() -> str:
    return (
        "Error: video generation is OFF. Ask the user to enable it in "
        "Settings → Video generation → Google Flow, then try again. "
        "Do not attempt any workaround."
    )


def _craft_prompt(idea: str, dialogue: str, style: str, model: str) -> str:
    """Build the Flow prompt: English visual direction + exact dialogue."""
    parts = [idea.strip()]
    if style == "cartoon":
        parts.append("cute colorful 2D cartoon animation, family-friendly")
    elif style == "cinematic":
        parts.append("cinematic, realistic lighting")
    dlg = (dialogue or "").strip()
    if dlg:
        parts.append(
            'the character speaks to the camera in Burmese (Myanmar '
            f'language), saying exactly: "{dlg}" — clear Burmese '
            "pronunciation, accurate lip sync"
        )
    else:
        parts.append("cheerful background music, no dialogue")
    parts.append("vertical video")
    return ", ".join(p for p in parts if p)


def _copy_to_clipboard(text: str) -> bool:
    exe = shutil.which("termux-clipboard-set")
    if not exe:
        return False
    try:
        r = subprocess.run([exe], input=text, capture_output=True,
                           text=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def _open_flow() -> bool:
    try:
        from tools.android_intent import open_url
        return open_url(FLOW_URL).startswith("OK")
    except Exception:
        return False


def prepare_video(idea: str, dialogue: str = "", style: str = "cartoon",
                  model: str = "omni", duration_s: int = 8,
                  resolution: str = "360p") -> str:
    """Prepare a Google Flow video job and hand it to the user.

    Crafts the final prompt, records the job, copies the prompt to the
    clipboard (when Termux:API is installed) and opens Flow in the
    browser. The user pastes the prompt, sets the model/outputs to 1,
    taps Generate, then Download — Flow sign-in and generation can
    never be automated. NEVER call this twice for the same request: a
    duplicate job risks the user generating twice and spending daily
    credits twice. If generation fails in Flow, the user is NOT
    charged; prepare a fresh job only when the user asks to retry.

    Args:
        idea: What the video shows, in plain words (any language). Keep
            one scene, one action — 4–10 second clips cannot do more.
        dialogue: Optional exact Burmese line the character speaks.
            One short sentence only. Requires model="omni" — Veo Lite
            cannot generate Burmese audio and fails the whole job.
        style: "cartoon" (default), "cinematic", or "plain".
        model: "omni" (Omni 1.1 Flash — Burmese dialogue, cheaper) or
            "veo_lite" (Veo 3.1 Lite — silent/musical clips only).
        duration_s: 4, 6, 8 or 10 seconds.
        resolution: "360p" (default; fine for cartoons) or "720p".
    """
    if not _provider_enabled():
        return _disabled_msg()
    idea = (idea or "").strip()
    if not idea:
        return "Error: idea is required — describe what the video shows."
    model = (model or "omni").strip().lower()
    if model not in _MODELS:
        return f"Error: model must be one of {_MODELS}, got {model!r}"
    style = (style or "cartoon").strip().lower()
    if style not in ("cartoon", "cinematic", "plain"):
        return f"Error: style must be cartoon/cinematic/plain, got {style!r}"
    try:
        dur = int(duration_s)
    except Exception:
        return f"Error: duration_s must be an integer, got {duration_s!r}"
    if dur not in _DURATIONS:
        return f"Error: duration_s must be one of {_DURATIONS}, got {dur}"
    resolution = (resolution or "360p").strip().lower()
    if resolution not in _RESOLUTIONS:
        return f"Error: resolution must be one of {_RESOLUTIONS}"
    dialogue = (dialogue or "").strip()
    if dialogue and model == "veo_lite":
        return (
            "Error: Veo 3.1 Lite cannot generate Burmese speech audio "
            "(verified on-device: the whole generation fails). Use "
            "model='omni' for dialogue, or drop the dialogue."
        )

    prompt = _craft_prompt(idea, dialogue, style, model)
    est = _estimate_credits(model, resolution, dur)
    job = {
        "id": uuid.uuid4().hex[:12],
        "state": "awaiting_human",
        "idea": idea,
        "dialogue": dialogue,
        "style": style,
        "model": model,
        "duration_s": dur,
        "resolution": resolution,
        "prompt": prompt,
        "est_credits": est,
        "created_at": time.time(),
        "file": None,
    }
    jobs = _load_jobs()
    jobs.append(job)
    _save_jobs(jobs)

    copied = _copy_to_clipboard(prompt)
    opened = _open_flow()
    model_label = "Omni 1.1 Flash" if model == "omni" else "Veo 3.1 - Lite"
    est_txt = f"~{est} credits" if est is not None else "credits shown in Flow"
    lines = [
        f"OK: video job {job['id']} prepared ({model_label}, "
        f"{resolution}, {dur}s, {est_txt} from the free daily 50).",
    ]
    if opened:
        lines.append("Flow is opening in the browser.")
    else:
        lines.append("Could not open the browser automatically — ask "
                     "the user to open flow.google.com.")
    if copied:
        lines.append("The prompt is on the clipboard — the user pastes "
                     "it into Flow's prompt box.")
    else:
        lines.append("Clipboard unavailable (Termux:API not found) — "
                     "show the user this exact prompt to copy:")
        lines.append(prompt)
    lines.append(
        "Tell the user, in their language, step by step: "
        f"1) paste the prompt, 2) model = {model_label}, outputs x1, "
        f"aspect 9:16, {resolution}, {dur}s, 3) tap Generate, "
        "4) when finished tap Download. "
        "Do NOT prepare another job for this request. When the user "
        "says the download finished, call import_video with this job id."
    )
    return "\n".join(lines)


def video_status(job_id: str = "") -> str:
    """List video jobs and their states (read-only).

    Args:
        job_id: Optional single job id to inspect.
    """
    jobs = _load_jobs()
    if job_id:
        jobs = [j for j in jobs if j.get("id") == job_id]
        if not jobs:
            return f"Error: no video job with id {job_id!r}"
    if not jobs:
        return "No video jobs yet."
    out = []
    for j in jobs[-10:]:
        out.append(
            f"{j.get('id')} [{j.get('state')}] {j.get('model')} "
            f"{j.get('resolution')} {j.get('duration_s')}s — "
            f"{(j.get('idea') or '')[:60]}"
        )
    return "\n".join(out)


def _download_dirs() -> list:
    override = os.environ.get("AI_AGENT_DOWNLOAD_DIRS")
    if override:
        return [Path(p) for p in override.split(os.pathsep) if p]
    return [
        Path("/storage/emulated/0/Download"),
        Path.home() / "storage" / "shared" / "Download",
        Path.home() / "Downloads",
    ]


def _find_candidate(since_ts: float, taken: set):
    """Newest .mp4 in the Download dirs newer than since_ts, or None."""
    best = None
    seen = set()
    for d in _download_dirs():
        try:
            if not d.is_dir():
                continue
            real = d.resolve()
            if real in seen:
                continue
            seen.add(real)
            for f in d.iterdir():
                if f.suffix.lower() != ".mp4" or not f.is_file():
                    continue
                if str(f) in taken:
                    continue
                try:
                    mt = f.stat().st_mtime
                except OSError:
                    continue
                if mt < since_ts - 5:  # small clock-skew allowance
                    continue
                if best is None or mt > best[1]:
                    best = (f, mt)
        except OSError:
            continue
    return best[0] if best else None


def import_video(job_id: str) -> str:
    """Import the downloaded MP4 for a prepared video job.

    Finds the newest .mp4 in the device Download folder created after
    the job was prepared, copies it into the agent's video library and
    marks the job imported. On success the reply to the user MUST end
    with the marker line printed at the end of the result, verbatim —
    the app renders it as a playable video.

    Args:
        job_id: The job id returned by prepare_video.
    """
    if not _provider_enabled():
        return _disabled_msg()
    jobs = _load_jobs()
    job = next((j for j in jobs if j.get("id") == job_id), None)
    if job is None:
        return f"Error: no video job with id {job_id!r}"
    if job.get("state") == "imported" and job.get("file"):
        return (f"OK: job {job_id} was already imported.\n"
                f"[video:{job_id}]")
    taken = {j.get("source") for j in jobs if j.get("source")}
    cand = _find_candidate(float(job.get("created_at") or 0), taken)
    if cand is None:
        return (
            "Error: no new .mp4 found in the Download folder since "
            "this job was prepared. Ask the user whether the Flow "
            "download finished; if it did, the file may be in a "
            "different folder — do not guess, ask."
        )
    dest = _videos_dir() / f"{job_id}.mp4"
    try:
        shutil.copy2(cand, dest)
    except OSError as e:
        return f"Error: could not copy {cand} into the library: {e}"
    job["state"] = "imported"
    job["file"] = dest.name
    job["source"] = str(cand)
    job["imported_at"] = time.time()
    _save_jobs(jobs)
    return (
        f"OK: imported {cand.name} into the video library "
        f"({dest.stat().st_size // 1024} KB). The video is ready to "
        "watch in this chat.\n"
        f"[video:{job_id}]"
    )


ALL_TOOLS = [prepare_video, video_status, import_video]
__all__ = [
    "prepare_video", "video_status", "import_video", "ALL_TOOLS",
    "FLOW_URL",
]
