"""
Capture worker: reads frames and never blocks on anything downstream.
"""
import queue
import threading
import time

import cv2

from .. import config
from ..framebuffer import FrameRing

# end-of-stream marker put on the queue when a source runs out
END_OF_STREAM = object()


class CaptureWorker(threading.Thread):
    def __init__(self, camera_id: str, source, *, out_queue: queue.Queue,
                 stop_event: threading.Event, source_name: str = None,
                 frame_skip: int = None, preroll_seconds: float = None,
                 realtime: bool = False):
        super().__init__(name=f"capture:{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.source = source
        self.source_name = source_name or str(source)
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.frame_skip = max(1, frame_skip or config.FRAME_SKIP)
        self.realtime = realtime
        # A camera will not wait, so when the trigger falls behind the only
        # choices are to drop a frame or to fall further behind - and falling
        # behind compounds. A FILE will wait, and dropping from it means
        # scanning something other than the footage, so there we block.
        self.drop_when_full = realtime

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open source: {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        # the ring only ever holds frames the trigger would sample, so its
        # capacity is in sampled frames, not raw ones
        seconds = config.PREROLL_SECONDS if preroll_seconds is None else preroll_seconds
        self.ring = FrameRing(max(1, int(seconds * self.fps / self.frame_skip)))

        self.frames_read = 0
        self.frames_sampled = 0
        self.frames_dropped = 0
        self.finished = False

    def run(self):
        interval = self.frame_skip / self.fps
        next_due = time.monotonic()
        frame_no = -1
        try:
            while not self.stop_event.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    break
                frame_no += 1
                self.frames_read += 1
                if frame_no % self.frame_skip:
                    continue

                self.frames_sampled += 1
                self.ring.push(frame_no, frame)
                if self.drop_when_full:
                    try:
                        self.out_queue.put_nowait((frame_no, frame))
                    except queue.Full:
                        self.frames_dropped += 1
                else:
                    # finite source: apply backpressure instead of losing frames
                    while not self.stop_event.is_set():
                        try:
                            self.out_queue.put((frame_no, frame), timeout=0.5)
                            break
                        except queue.Full:
                            continue

                if self.realtime:
                    next_due += interval
                    delay = next_due - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_due = time.monotonic()
        finally:
            self.finished = True
            self.cap.release()
            try:
                self.out_queue.put(END_OF_STREAM, timeout=2)
            except queue.Full:
                pass

    def stats(self) -> dict:
        return {"read": self.frames_read, "sampled": self.frames_sampled,
                "dropped": self.frames_dropped, "buffered": len(self.ring),
                "buffer_bytes": self.ring.bytes_held()}
