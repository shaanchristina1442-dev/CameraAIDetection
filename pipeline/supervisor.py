"""
Starts the workers for a camera, keeps them alive, and shuts them down cleanly.

This becomes a thing that runs for weeks. A capture worker dying at 3am must
not silently stop monitoring that camera - the failure a surveillance system
cannot afford is believing it has coverage it does not. So the supervisor
watches its workers and restarts a capture that dies unexpectedly, logging
every restart so the gap is visible rather than invisible.

Restarting only makes sense for a live source. A file that ends has ended, and
restarting would re-scan it forever, so it is off by default and on for
realtime sources.

Usage:
    python -m pipeline.supervisor footage/clip.mp4
    python -m pipeline.supervisor footage/clip.mp4 --dry-run
    python -m pipeline.supervisor rtsp://... --camera front-door --realtime
"""
import argparse
import os
import queue
import sys
import threading
import time

from . import config
from .dedup import Suppressor
from .trigger import TriggerEngine
from .workers import CaptureWorker, TriggerWorker, JudgeWorker
from .workers.capture import END_OF_STREAM
from storage import repository


class CameraPipeline:
    """capture -> trigger -> judge pool, for one camera."""

    def __init__(self, source, camera_id: str = None, *, realtime: bool = False,
                 dry_run: bool = False, judge_workers: int = None,
                 restart_on_crash: bool = None, engine=None, log=print):
        self.source = source
        self.source_name = os.path.basename(str(source))
        self.camera_id = camera_id or self._default_camera_id(source)
        self.realtime = realtime
        self.dry_run = dry_run
        self.log = log
        self.judge_count = judge_workers or config.JUDGE_WORKERS
        self.restart_on_crash = realtime if restart_on_crash is None else restart_on_crash

        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=config.CAPTURE_QUEUE_MAX)
        self.candidate_queue = queue.Queue(maxsize=config.JUDGE_QUEUE_MAX)
        self.suppressor = Suppressor()
        # a caller processing many clips can pass one loaded model in; a
        # camera must not share one, since the tracker state lives on it
        self.engine = engine
        self.capture = None
        self.trigger = None
        self.judges = []
        self.restarts = 0

    @staticmethod
    def _default_camera_id(source) -> str:
        stem = os.path.splitext(os.path.basename(str(source)))[0]
        return f"file:{stem}"

    def _make_capture(self):
        return CaptureWorker(
            self.camera_id, self.source, out_queue=self.frame_queue,
            stop_event=self.stop_event, source_name=self.source_name,
            realtime=self.realtime)

    def start(self):
        if not self.dry_run:
            repository.init()
            repository.ensure_camera(self.camera_id, name=self.source_name,
                                     source=str(self.source))
        # one engine per camera: ultralytics keeps tracker state on the model,
        # so sharing one across cameras would mix their tracks together
        if self.engine is None:
            self.engine = TriggerEngine()
        self.capture = self._make_capture()
        self.trigger = TriggerWorker(self.engine, self.capture,
                                     out_queue=self.candidate_queue,
                                     stop_event=self.stop_event)
        self.judges = [
            JudgeWorker(self.candidate_queue, stop_event=self.stop_event,
                        suppressor=self.suppressor, dry_run=self.dry_run,
                        index=i, log=self.log)
            for i in range(self.judge_count)
        ]
        for w in [self.capture, self.trigger, *self.judges]:
            w.start()
        self.log(f"[supervisor] {self.camera_id}: capture + trigger + "
                 f"{self.judge_count} judge worker(s) running")
        return self

    def _watch_once(self):
        """Restart a capture worker that died without being asked to."""
        if not self.restart_on_crash or self.stop_event.is_set():
            return
        if self.capture and not self.capture.is_alive() and not self.capture.finished:
            self.restarts += 1
            self.log(f"[supervisor] {self.camera_id}: capture died unexpectedly, "
                     f"restarting (restart #{self.restarts})")
            self.capture = self._make_capture()
            self.trigger.capture = self.capture
            self.capture.start()

    def run_to_completion(self, poll: float = 0.25, timeout: float = None):
        """
        For a finite source: wait until the stream ends and every candidate
        has been judged, then stop. Live sources use run_forever instead.
        """
        started = time.monotonic()
        while self.trigger.is_alive():
            self._watch_once()
            if timeout and time.monotonic() - started > timeout:
                self.log("[supervisor] timeout waiting for the trigger")
                break
            time.sleep(poll)
        # the trigger has emitted everything; let the judges drain the queue
        while any(j.is_alive() for j in self.judges):
            if self.candidate_queue.unfinished_tasks == 0 and self._only_sentinel_left():
                break
            if timeout and time.monotonic() - started > timeout:
                self.log("[supervisor] timeout waiting for the judges")
                break
            time.sleep(poll)
        self.stop()
        return self.stats()

    def _only_sentinel_left(self) -> bool:
        with self.candidate_queue.mutex:
            items = list(self.candidate_queue.queue)
        return all(i is END_OF_STREAM for i in items)

    def run_forever(self, poll: float = 1.0):
        """For a live source: keep workers alive until interrupted."""
        try:
            while not self.stop_event.is_set():
                self._watch_once()
                time.sleep(poll)
        except KeyboardInterrupt:
            self.log("\n[supervisor] stopping")
        finally:
            self.stop()
        return self.stats()

    def stop(self, join_timeout: float = 5.0):
        self.stop_event.set()
        for w in [self.capture, self.trigger, *self.judges]:
            if w and w.is_alive():
                w.join(timeout=join_timeout)

    def stats(self) -> dict:
        return {
            "camera": self.camera_id,
            "capture": self.capture.stats() if self.capture else {},
            "trigger": self.trigger.stats() if self.trigger else {},
            "judges": [j.stats() for j in self.judges],
            "suppression": self.suppressor.stats(),
            "restarts": self.restarts,
        }


def main():
    ap = argparse.ArgumentParser(description="Run the live pipeline for one camera.")
    ap.add_argument("source", help="video file, or an RTSP URL with --realtime")
    ap.add_argument("--camera", help="camera id (default: derived from the source)")
    ap.add_argument("--realtime", action="store_true",
                    help="pace capture to the source's frame rate, and keep running")
    ap.add_argument("--dry-run", action="store_true",
                    help="trigger only: no Claude calls, nothing recorded")
    ap.add_argument("--judges", type=int, default=None, help="judge workers")
    args = ap.parse_args()

    if not args.dry_run and not args.realtime and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("[supervisor] ANTHROPIC_API_KEY is not set. Add it to .env or use --dry-run.")

    pipe = CameraPipeline(args.source, args.camera, realtime=args.realtime,
                          dry_run=args.dry_run, judge_workers=args.judges).start()
    stats = pipe.run_forever() if args.realtime else pipe.run_to_completion()

    print("\n[supervisor] done")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
