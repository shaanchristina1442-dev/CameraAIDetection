"""
The only place that touches the database.

Both the pipeline and the dashboard import this. If each wrote its own SQL
they would drift, and this is also the natural home for the alert-suppression
rules task K adds - they belong next to the write, not in whichever caller
happens to remember them.

Events are returned in the same shape the old JSONL records had, so the
dashboard's templates and JavaScript did not change when storage did.
"""
import json
import time

from . import db

REVIEW_STATES = ("pending", "confirmed", "dismissed")


def init():
    return db.init_db()


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# ---------------------------------------------------------------- cameras

def ensure_camera(camera_id: str, name: str = None, source: str = None) -> str:
    """Create the camera if it does not exist. Returns its id."""
    with db.session() as conn:
        row = conn.execute("SELECT id FROM cameras WHERE id = ?", (camera_id,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO cameras (id, name, source, created_at) VALUES (?, ?, ?, ?)",
                (camera_id, name or camera_id, source, _now()))
    return camera_id


def list_cameras() -> list[dict]:
    with db.session() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM cameras ORDER BY name").fetchall()]


# ----------------------------------------------------------------- events

def insert_event(event_id: str, judgment: dict, frames: list[str], *,
                 video: str = None, camera_id: str = None,
                 start_time: float = 0.0, end_time: float = 0.0,
                 trigger_reason: str = None, peak_motion: float = None,
                 person_count: int = None, alert: bool = False,
                 created_at: str = None) -> str:
    """Record one judged event and its frames."""
    with db.session() as conn:
        conn.execute("""
            INSERT INTO events (
                event_id, camera_id, video, start_time, end_time, trigger_reason,
                peak_motion, person_count, category, confidence,
                visible_description, reasoning, recommend_human_review,
                judge_error, alert, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            event_id, camera_id, video, start_time, end_time, trigger_reason,
            peak_motion, person_count,
            judgment.get("category"), judgment.get("confidence"),
            judgment.get("visible_description"), judgment.get("reasoning"),
            1 if judgment.get("recommend_human_review") else 0,
            judgment.get("error"), 1 if alert else 0, created_at or _now(),
        ))
        conn.executemany(
            "INSERT INTO event_frames (event_id, idx, path) VALUES (?,?,?)",
            [(event_id, i, p) for i, p in enumerate(frames)])
    return event_id


def _row_to_event(row, frames: list[str], review: str) -> dict:
    """Rebuild the record shape the dashboard already understands."""
    judgment = {
        "category": row["category"],
        "confidence": row["confidence"],
        "visible_description": row["visible_description"],
        "reasoning": row["reasoning"],
        "recommend_human_review": bool(row["recommend_human_review"]),
    }
    if row["judge_error"]:
        judgment["error"] = row["judge_error"]
    return {
        "event_id": row["event_id"],
        "camera_id": row["camera_id"],
        "video": row["video"],
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "frames": frames,
        "judgment": judgment,
        "peak_motion": row["peak_motion"],
        "person_count": row["person_count"],
        "trigger_reason": row["trigger_reason"],
        "alert": bool(row["alert"]),
        "review": review,
        "created_at": row["created_at"],
    }


def _frames_for(conn, event_ids: list[str]) -> dict:
    if not event_ids:
        return {}
    marks = ",".join("?" * len(event_ids))
    out = {}
    for r in conn.execute(
            f"SELECT event_id, path FROM event_frames WHERE event_id IN ({marks}) "
            "ORDER BY event_id, idx", event_ids):
        out.setdefault(r["event_id"], []).append(r["path"])
    return out


def _reviews_for(conn, event_ids: list[str]) -> dict:
    """Latest review per event. The table is append-only, so the newest row wins."""
    if not event_ids:
        return {}
    marks = ",".join("?" * len(event_ids))
    return {r["event_id"]: r["status"] for r in conn.execute(
        f"""SELECT event_id, status FROM reviews
            WHERE id IN (SELECT MAX(id) FROM reviews
                         WHERE event_id IN ({marks}) GROUP BY event_id)""", event_ids)}


def list_events(limit: int = None, camera_id: str = None,
                alerts_only: bool = False) -> list[dict]:
    """Newest first."""
    sql = "SELECT * FROM events"
    where, params = [], []
    if camera_id:
        where.append("camera_id = ?"); params.append(camera_id)
    if alerts_only:
        where.append("alert = 1")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC, rowid DESC"
    if limit:
        sql += " LIMIT ?"; params.append(limit)

    with db.session() as conn:
        rows = conn.execute(sql, params).fetchall()
        ids = [r["event_id"] for r in rows]
        frames, reviews = _frames_for(conn, ids), _reviews_for(conn, ids)
    return [_row_to_event(r, frames.get(r["event_id"], []),
                          reviews.get(r["event_id"], "pending")) for r in rows]


def get_event(event_id: str) -> dict | None:
    with db.session() as conn:
        row = conn.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
        if row is None:
            return None
        frames = _frames_for(conn, [event_id]).get(event_id, [])
        review = _reviews_for(conn, [event_id]).get(event_id, "pending")
    return _row_to_event(row, frames, review)


def event_exists(event_id: str) -> bool:
    with db.session() as conn:
        return conn.execute("SELECT 1 FROM events WHERE event_id = ?",
                            (event_id,)).fetchone() is not None


# ---------------------------------------------------------------- reviews

def set_review(event_id: str, status: str, reviewer: str = None,
               note: str = None) -> dict:
    """
    Append a review decision. Never updates or deletes an earlier one: the
    history of who decided what, and when, is the point.
    """
    if status not in REVIEW_STATES:
        raise ValueError(f"status must be one of {REVIEW_STATES}")
    with db.session() as conn:
        conn.execute(
            "INSERT INTO reviews (event_id, status, reviewer, note, created_at) "
            "VALUES (?,?,?,?,?)", (event_id, status, reviewer, note, _now()))
    return {"event_id": event_id, "review": status}


def review_history(event_id: str) -> list[dict]:
    with db.session() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM reviews WHERE event_id = ? ORDER BY id", (event_id,))]


def delete_event(event_id: str) -> bool:
    """
    Remove an event and, by cascade, its frames and review history. Image
    files on disk are the caller's to clean up. Task T's retention purge
    uses this.
    """
    with db.session() as conn:
        cur = conn.execute("DELETE FROM events WHERE event_id = ?", (event_id,))
        return cur.rowcount > 0


def event_ids_with_prefix(prefix: str) -> list[str]:
    with db.session() as conn:
        return [r["event_id"] for r in conn.execute(
            "SELECT event_id FROM events WHERE event_id LIKE ? ORDER BY event_id",
            (prefix + "%",))]


# ------------------------------------------------------------------ stats

def stats() -> dict:
    with db.session() as conn:
        row = conn.execute("""
            SELECT COUNT(*) AS events,
                   SUM(alert) AS alerts,
                   SUM(CASE WHEN category IS NOT NULL
                             AND category != 'nothing_notable' THEN 1 ELSE 0 END) AS notable,
                   SUM(CASE WHEN judge_error IS NOT NULL THEN 1 ELSE 0 END) AS errors
            FROM events""").fetchone()
        reviewed = conn.execute(
            "SELECT COUNT(DISTINCT event_id) AS n FROM reviews "
            "WHERE status != 'pending'").fetchone()
    return {"events": row["events"] or 0, "alerts": row["alerts"] or 0,
            "notable": row["notable"] or 0, "judge_errors": row["errors"] or 0,
            "reviewed": reviewed["n"] or 0}
