"""
Watches FOOTAGE_DIR for new video files and runs each one through the pipeline.

Usage:
    python -m pipeline.watcher [--dry-run]

Copy or export store footage into footage/ while this is running. Files that
are already there when the watcher starts are processed too.
"""
import os
import queue
import sys
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from . import config
from .run_pipeline import run
from .trigger import TriggerEngine


def is_video(path: str) -> bool:
    return path.lower().endswith(config.VIDEO_EXTENSIONS)


def wait_until_written(path: str, settle_seconds: float = 2.0):
    """Wait until the file stops growing, so we don't read a half-copied video."""
    last = -1
    while True:
        try:
            size = os.path.getsize(path)
        except FileNotFoundError:
            return False
        if size == last and size > 0:
            return True
        last = size
        time.sleep(settle_seconds)


class NewClipHandler(FileSystemEventHandler):
    def __init__(self, todo: queue.Queue):
        self.todo = todo

    def on_created(self, event):
        if not event.is_directory and is_video(event.src_path):
            self.todo.put(event.src_path)

    def on_moved(self, event):
        if not event.is_directory and is_video(event.dest_path):
            self.todo.put(event.dest_path)


def main(dry_run: bool = False):
    if not dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("[watcher] ANTHROPIC_API_KEY is not set. Add it to .env or use --dry-run.")
    os.makedirs(config.FOOTAGE_DIR, exist_ok=True)
    engine = TriggerEngine()  # load YOLO once and reuse it for every clip
    todo: queue.Queue = queue.Queue()
    done: set[str] = set()

    for name in sorted(os.listdir(config.FOOTAGE_DIR)):
        path = os.path.join(config.FOOTAGE_DIR, name)
        if is_video(path):
            todo.put(path)

    observer = Observer()
    observer.schedule(NewClipHandler(todo), config.FOOTAGE_DIR, recursive=False)
    observer.start()
    print(f"[watcher] watching {config.FOOTAGE_DIR} for new footage (Ctrl+C to stop)")

    try:
        while True:
            path = todo.get()
            if path in done or not wait_until_written(path):
                continue
            done.add(path)
            try:
                run(path, engine=engine, dry_run=dry_run)
            except Exception as e:
                # one bad file shouldn't stop the watcher
                print(f"[watcher] failed on {os.path.basename(path)}: {e}")
    except KeyboardInterrupt:
        print("\n[watcher] stopping")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
