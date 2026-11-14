-- Store monitoring schema.
--
-- Replaces events/events.jsonl and events/reviews.json. The file-based store
-- could not survive two writers: appending from several camera workers
-- interleaves lines, and the dashboard had to keep reviews in a separate file
-- purely to avoid rewriting the log while the pipeline was appending to it.
-- Phase 2 introduces exactly those concurrent writers.
--
-- Only the tables with a current use are defined here. zones, users and
-- devices appear in the roadmap (tasks Q, R, U) and are deliberately absent:
-- a table designed before anything reads it tends to be wrong, and an empty
-- one invites code to assume a shape nobody has tested.

CREATE TABLE IF NOT EXISTS cameras (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    -- RTSP URL or file path. Credentials get encrypted in task P; until then
    -- this holds file paths only.
    source      TEXT,
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    camera_id       TEXT REFERENCES cameras(id),
    video           TEXT,
    -- seconds into the source video
    start_time      REAL NOT NULL,
    end_time        REAL NOT NULL,
    trigger_reason  TEXT,
    peak_motion     REAL,
    person_count    INTEGER,
    -- judgment layer output, flattened so it can be queried and indexed
    category        TEXT,
    confidence      REAL,
    visible_description TEXT,
    reasoning       TEXT,
    recommend_human_review INTEGER,
    -- set when the judge failed; the event is still recorded
    judge_error     TEXT,
    alert           INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_frames (
    event_id    TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    path        TEXT NOT NULL,
    PRIMARY KEY (event_id, idx)
);

-- Append-only. The current state of an event is its most recent row; earlier
-- rows are never updated or deleted. This is what makes "a human decided"
-- verifiable rather than merely asserted, and it is already the shape task T
-- needs for the audit log.
CREATE TABLE IF NOT EXISTS reviews (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    status      TEXT NOT NULL CHECK (status IN ('pending', 'confirmed', 'dismissed')),
    reviewer    TEXT,
    note        TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_camera_created ON events (camera_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_alert_created  ON events (alert, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_created        ON events (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reviews_event         ON reviews (event_id, id DESC);
