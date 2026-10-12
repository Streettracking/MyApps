"""Laptop copy of the onboard status, about twice a second.

Testers read ``logs/onboard_status_<date>.jsonl`` instead of GET /status on
the dog. A GET does not keep the link up, so a passive poll is the wrong
place to watch a run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

STATUS_PERIOD_S = 0.50


def onboard_status_path(root, wall: float | None = None) -> Path:
    """One file per calendar day. ``wall`` is seconds since the epoch."""
    stamp = time.strftime("%Y-%m-%d", time.localtime(wall))
    return Path(root) / "logs" / ("onboard_status_%s.jsonl" % stamp)


def append_onboard_status(root, status, now: float, state: dict, wall: float | None = None):
    """Append one snapshot when ``STATUS_PERIOD_S`` has passed. Otherwise None.

    ``now`` is the caller's clock and only gates the rate. The file date and
    ``logged_at`` use the wall clock.
    """
    if not isinstance(status, dict) or not status:
        return None
    gate = float(state.get("status_log_t", -1e9))
    if float(now) - gate < STATUS_PERIOD_S:
        return None
    state["status_log_t"] = float(now)
    when = time.time() if wall is None else float(wall)
    path = onboard_status_path(root, when)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(status)
    row["logged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(when))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
    return path
