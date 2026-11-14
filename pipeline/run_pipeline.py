"""
Run one clip end to end: trigger -> judgment -> database.

A thin entry point over the worker pipeline in supervisor.py. It used to be a
synchronous loop of its own, which is why live cameras were a rewrite rather
than a new caller; now a file and a camera are the same code with different
sources, and anything fixed here is fixed for both.

Usage:
    python -m pipeline.run_pipeline <video_path> [--dry-run]

--dry-run runs only the trigger layer (no Claude calls, nothing recorded),
which is the free way to tune thresholds on new footage.
"""
import os
import sys
import time

from .supervisor import CameraPipeline


def fmt_time(sec: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(sec))


def run(video_path: str, engine=None, dry_run: bool = False):
    """
    Scan one clip and judge what it finds. `engine` lets a caller processing
    several clips reuse one loaded model.
    """
    if not dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("[pipeline] ANTHROPIC_API_KEY is not set. Add it to .env or use --dry-run.")

    name = os.path.basename(video_path)
    print(f"[pipeline] scanning {name} for candidate events")

    pipe = CameraPipeline(video_path, dry_run=dry_run, engine=engine).start()
    stats = pipe.run_to_completion()

    found = stats["trigger"].get("events", 0)
    sup = stats["suppression"]
    if not found:
        print("[pipeline] no candidate events found. Try lowering MOTION_THRESHOLD in config.py")
        return stats

    print(f"[pipeline] {found} candidate event(s); {sup['judged']} judged, "
          f"{sup['suppressed']} suppressed")
    if not dry_run:
        judged = sum(j["judged"] for j in stats["judges"])
        failed = sum(j["failed"] for j in stats["judges"])
        print(f"[pipeline] {judged} judgment(s) recorded"
              + (f", {failed} failed" if failed else ""))
        print("[pipeline] review them in the dashboard: python -m dashboard.app")
    return stats


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print("Usage: python -m pipeline.run_pipeline <video_path> [--dry-run]")
        sys.exit(1)
    run(args[0], dry_run="--dry-run" in sys.argv)
