"""
Judge worker: turns a candidate event into a recorded judgment.

Suppression happens here, before any money is spent. A candidate that is
suppressed is still written to the database - the owner should be able to
see that activity occurred - but it is not judged and never becomes an alert.
"""
import os
import queue
import threading
import time
import uuid

import cv2

from .. import config
from ..judge import judge_event, crop_to_people
from storage import repository
from .capture import END_OF_STREAM


def _new_event_id() -> str:
    # timestamp keeps ids sorting chronologically; the random suffix stops two
    # events in the same second from overwriting each other's frames
    return f"{int(time.time())}_{uuid.uuid4().hex[:8]}"


def _sample(frames, boxes, n):
    """n frames spread evenly across the event, with their boxes aligned."""
    if len(frames) <= n:
        picked = list(range(len(frames)))
    else:
        step = len(frames) / n
        picked = [int(i * step) for i in range(n)]
    return ([frames[i] for i in picked],
            [boxes[i] if i < len(boxes) else [] for i in picked])


def _save_frames(event_id: str, frames) -> list[str]:
    os.makedirs(config.CLIPS_DIR, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        name = f"{event_id}_frame{i}.jpg"
        cv2.imwrite(os.path.join(config.CLIPS_DIR, name), frame)
        paths.append(f"clips/{name}")
    return paths


def is_alert(judgment: dict) -> bool:
    return (judgment.get("category") not in (None, "nothing_notable")
            and judgment.get("confidence", 0.0) >= config.ALERT_CONFIDENCE_FLOOR)


def process_candidate(event, *, suppressor=None, dry_run: bool = False, log=print):
    """
    Judge one candidate and record it. Returns the stored record, or None if
    it was suppressed or this is a dry run.

    Shared by the worker and by the file-scan entry point, so both behave
    identically - the live path is not a second implementation.
    """
    camera_id = event.camera_id
    window = f"{event.start_time_sec:.0f}s-{event.end_time_sec:.0f}s"

    if suppressor is not None:
        judge_it, reason = suppressor.check(camera_id, event.track_ids, event.start_time_sec)
        if not judge_it:
            if not dry_run:
                repository.insert_event(
                    _new_event_id(), judgment={}, frames=[],
                    video=event.source_name, camera_id=camera_id,
                    start_time=event.start_time_sec, end_time=event.end_time_sec,
                    trigger_reason=event.reason, peak_motion=event.peak_motion,
                    person_count=event.person_count_at_peak,
                    alert=False, suppressed_reason=reason)
            log(f"[judge]   {window} suppressed ({reason})")
            return None

    if dry_run:
        return None

    event_id = _new_event_id()
    frames, boxes = _sample(event.sampled_frames, event.sampled_boxes,
                            config.FRAMES_PER_JUDGMENT)
    # pre-roll frames carry no boxes, so they are sent whole: close crops for
    # detail, plus the wider scene for context
    frames = [crop_to_people(f, b) for f, b in zip(frames, boxes)]
    frame_paths = _save_frames(event_id, frames)

    cropped = sum(1 for b in boxes if b)
    log(f"[judge]   {window} sending {len(frames)} frames ({cropped} cropped)...")
    judgment = judge_event(frames, event.reason)
    alert = is_alert(judgment)

    repository.insert_event(
        event_id, judgment=judgment, frames=frame_paths,
        video=event.source_name, camera_id=camera_id,
        start_time=event.start_time_sec, end_time=event.end_time_sec,
        trigger_reason=event.reason, peak_motion=event.peak_motion,
        person_count=event.person_count_at_peak, alert=alert)

    flag = "  ALERT - needs human review" if alert else ""
    log(f"[judge]   {window} -> {judgment.get('category')} "
        f"({judgment.get('confidence', 0):.2f}){flag}")
    if "error" in judgment:
        log(f"[judge]   {window} judge error: {judgment['error']}")
    return {"event_id": event_id, "judgment": judgment, "alert": alert}


class JudgeWorker(threading.Thread):
    def __init__(self, in_queue: queue.Queue, *, stop_event: threading.Event,
                 suppressor=None, dry_run: bool = False, index: int = 0, log=print):
        super().__init__(name=f"judge-{index}", daemon=True)
        self.in_queue = in_queue
        self.stop_event = stop_event
        self.suppressor = suppressor
        self.dry_run = dry_run
        self.log = log
        self.judged = 0
        self.failed = 0
        self.finished = False

    def run(self):
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.in_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is END_OF_STREAM:
                    # put it back so sibling workers also see the end
                    self.in_queue.put(END_OF_STREAM)
                    break
                try:
                    if process_candidate(item, suppressor=self.suppressor,
                                         dry_run=self.dry_run, log=self.log):
                        self.judged += 1
                except Exception as e:
                    # one bad event must not take the worker down with it
                    self.failed += 1
                    self.log(f"[judge]   failed on an event: {type(e).__name__}: {e}")
                finally:
                    self.in_queue.task_done()
        finally:
            self.finished = True

    def stats(self) -> dict:
        return {"judged": self.judged, "failed": self.failed}
