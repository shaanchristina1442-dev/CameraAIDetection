"""
SQLite connection handling.

A connection per operation rather than a shared one: sqlite3 connections are
not safe to use across threads, Flask serves requests on several, and Phase 2
adds worker processes. WAL mode is what makes that workable - readers do not
block the writer and the writer does not block readers, which is the whole
reason for moving off the JSONL file.
"""
import os
import sqlite3
from contextlib import contextmanager

from pipeline import config

DB_PATH = os.path.join(config.EVENTS_DIR, "store_monitor.db")
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")

# How long a writer waits for another writer's lock before giving up. Several
# camera workers finishing an event at the same moment is normal, not an error.
BUSY_TIMEOUT_MS = 5000


def connect(path: str = None) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path or DB_PATH), exist_ok=True)
    conn = sqlite3.connect(path or DB_PATH, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    return conn


@contextmanager
def session(path: str = None):
    """Connection that commits on success and rolls back on error."""
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(path: str = None) -> str:
    """Create the schema if absent. Safe to call on every start."""
    target = path or DB_PATH
    with open(SCHEMA_PATH) as f:
        schema = f.read()
    with session(target) as conn:
        conn.executescript(schema)
    return target
