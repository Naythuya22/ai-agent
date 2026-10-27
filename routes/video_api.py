"""Serve imported video clips to the PWA (Release 32).

GET /api/video/<job_id> streams ~/.ai-agent/videos/<job_id>.mp4 with
range support (Flask send_file conditional) so the chat's inline
<video> player can seek. The job id is the only path parameter and is
validated to a bare filename token — no traversal, no arbitrary file
reads. Job state lives in tools/video.py; this route only serves the
bytes for ids whose file actually exists in the library.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from flask import Blueprint, abort, send_file

bp = Blueprint("video_api", __name__)

_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _videos_dir() -> Path:
    # Same resolution as tools/video.py (env override for tests).
    override = os.environ.get("AI_AGENT_VIDEOS_DIR")
    return Path(override) if override else Path.home() / ".ai-agent" / "videos"


@bp.route("/api/video/<job_id>", methods=["GET"])
def api_video(job_id: str):
    if not _JOB_ID.match(job_id or ""):
        abort(404)
    path = _videos_dir() / f"{job_id}.mp4"
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype="video/mp4", conditional=True)
