"""
capai.cron
===========
Entry point for Render cron jobs.

Usage (via render.yaml startCommand):
    python -m capai.cron snapshot      # back up registry to timestamped JSON
    python -m capai.cron cleanup       # retire capabilities that never get used
    python -m capai.cron report        # print a summary of all live capabilities

Run any command locally too:
    CAPAI_HOME=.capai python -m capai.cron report
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .registry import CapabilityRegistry


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ── commands ─────────────────────────────────────────────────────────────────

def snapshot():
    """
    Copy registry.json to registry.<timestamp>.json in the same directory.
    Keeps the last 7 snapshots, deletes older ones.
    """
    src = config.CAPAI_HOME / "registry.json"
    if not src.exists():
        print("[snapshot] No registry found — nothing to back up.")
        return

    backup_dir = config.CAPAI_HOME / "snapshots"
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / f"registry.{_ts()}.json"
    shutil.copy2(src, dest)
    print(f"[snapshot] Saved → {dest}")

    # Keep only the 7 most recent
    all_snaps = sorted(backup_dir.glob("registry.*.json"), key=lambda p: p.stat().st_mtime)
    for old in all_snaps[:-7]:
        old.unlink()
        print(f"[snapshot] Removed old snapshot: {old.name}")


def cleanup():
    """
    Retire capabilities whose hit_count is 0 after at least 7 days in the registry.
    Safe to run daily — won't touch anything that's been used even once.
    """
    registry = CapabilityRegistry()
    caps = registry.list_active()
    now = datetime.now(timezone.utc)
    retired = 0

    for cap in caps:
        age_days = None
        if hasattr(cap, "created_at") and cap.created_at:
            try:
                created = datetime.fromisoformat(cap.created_at)
                if created.tzinfo is None:
                    from datetime import timezone as tz
                    created = created.replace(tzinfo=tz.utc)
                age_days = (now - created).days
            except Exception:
                pass

        hit_count = getattr(cap, "hit_count", None)
        if hit_count == 0 and age_days is not None and age_days >= 7:
            registry.retire(cap.name)
            print(f"[cleanup] Retired unused capability: {cap.name} (age {age_days}d, 0 hits)")
            retired += 1

    if retired == 0:
        print("[cleanup] Nothing to retire.")
    else:
        print(f"[cleanup] Retired {retired} capability(s).")


def report():
    """
    Print a human-readable summary of all active capabilities.
    Useful for daily digest or debugging.
    """
    registry = CapabilityRegistry()
    caps = registry.list_active()

    if not caps:
        print("[report] Registry is empty.")
        return

    print(f"\n{'─'*60}")
    print(f"  CapAI Registry Report — {_ts()}")
    print(f"{'─'*60}")
    print(f"  Total capabilities: {len(caps)}")
    print(f"{'─'*60}")
    for cap in sorted(caps, key=lambda c: c.name):
        hits = getattr(cap, "hit_count", "?")
        print(f"  {cap.name:<35} hits={hits:<6} verified={cap.verified}")
    print(f"{'─'*60}\n")


# ── entry point ───────────────────────────────────────────────────────────────

COMMANDS = {
    "snapshot": snapshot,
    "cleanup": cleanup,
    "report": report,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: python -m capai.cron [{' | '.join(COMMANDS)}]")
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
