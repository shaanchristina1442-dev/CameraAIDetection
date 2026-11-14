"""
Move existing events from events/events.jsonl into the SQLite database.

One-off. Safe to re-run: events already present are skipped, so a partial
run can simply be repeated. The JSONL file is left untouched - deleting it
is a separate decision, and keeping it until the database is trusted costs
nothing.

Usage:
    python scripts/migrate_jsonl_to_sqlite.py
    python scripts/migrate_jsonl_to_sqlite.py --dry-run
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import config          # noqa: E402
from storage import repository       # noqa: E402

REVIEWS_FILE = os.path.join(config.EVENTS_DIR, "reviews.json")


def main():
    dry = "--dry-run" in sys.argv
    repository.init()

    if not os.path.exists(config.EVENTS_LOG):
        print(f"No {config.EVENTS_LOG} - nothing to migrate.")
        return

    reviews = {}
    if os.path.exists(REVIEWS_FILE):
        try:
            reviews = json.load(open(REVIEWS_FILE))
        except json.JSONDecodeError:
            print(f"warning: {REVIEWS_FILE} is not valid JSON - ignoring it")

    migrated = skipped = bad = 0
    with open(config.EVENTS_LOG) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                print(f"  line {n}: not valid JSON - skipped")
                continue

            event_id = r.get("event_id")
            if not event_id:
                bad += 1
                print(f"  line {n}: no event_id - skipped")
                continue
            if repository.event_exists(event_id):
                skipped += 1
                continue

            if dry:
                migrated += 1
                continue

            video = r.get("video")
            camera_id = f"file:{os.path.splitext(video)[0]}" if video else None
            if camera_id:
                repository.ensure_camera(camera_id, name=video, source=video)

            repository.insert_event(
                event_id,
                judgment=r.get("judgment", {}) or {},
                frames=r.get("frames", []) or [],
                video=video,
                camera_id=camera_id,
                start_time=r.get("start_time", 0.0) or 0.0,
                end_time=r.get("end_time", 0.0) or 0.0,
                trigger_reason=r.get("trigger_reason"),
                peak_motion=r.get("peak_motion"),
                person_count=r.get("person_count"),
                alert=bool(r.get("alert")),
                created_at=r.get("created_at"),
            )
            # review state came from either the record or the separate file
            status = reviews.get(event_id, r.get("review", "pending"))
            if status and status != "pending":
                repository.set_review(event_id, status, reviewer="migrated")
            migrated += 1

    verb = "would migrate" if dry else "migrated"
    print(f"{verb} {migrated} event(s); {skipped} already present; {bad} unreadable")
    if not dry:
        print(f"database: {repository.db.DB_PATH}")
        print(f"stats: {repository.stats()}")


if __name__ == "__main__":
    main()
