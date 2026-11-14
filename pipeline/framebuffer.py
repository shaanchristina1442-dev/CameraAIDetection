"""
A rolling buffer of recent frames, so a trigger can reach backwards in time.

Scanning a file, pre-event context is free: seek backwards. A live camera has
no backwards - a frame not kept is gone. For concealment especially this
matters, because the informative footage is usually just BEFORE the motion
that fired the trigger, not after it.

Frames are held JPEG-encoded. Raw 1080p is ~6 MB a frame, so 30 seconds would
be gigabytes per camera; the same frames at quality 85 are ~250 KB, which puts
a 30-second buffer in the tens of megabytes. Decoding happens only for the
handful of frames an event actually uses.
"""
import threading
from collections import deque

import cv2
import numpy as np

from . import config


class FrameRing:
    """Thread-safe fixed-capacity ring of recent frames."""

    def __init__(self, capacity: int, quality: int = None):
        self.capacity = max(1, int(capacity))
        self.quality = quality if quality is not None else config.RING_JPEG_QUALITY
        self._buf = deque(maxlen=self.capacity)   # (frame_no, jpeg bytes)
        self._lock = threading.Lock()

    def push(self, frame_no: int, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        with self._lock:
            self._buf.append((frame_no, buf.tobytes()))

    def window(self, start_frame: int, end_frame: int, limit: int = None):
        """
        Decoded frames whose number falls in [start_frame, end_frame].

        Returns [(frame_no, frame)] oldest first. Frames that have already
        fallen out of the ring are simply absent - the caller gets what
        survived, which is the honest answer.
        """
        with self._lock:
            picked = [(n, b) for n, b in self._buf if start_frame <= n <= end_frame]
        if limit and len(picked) > limit:
            step = len(picked) / limit
            picked = [picked[int(i * step)] for i in range(limit)]
        out = []
        for n, b in picked:
            frame = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                out.append((n, frame))
        return out

    def oldest_frame_no(self) -> int | None:
        with self._lock:
            return self._buf[0][0] if self._buf else None

    def bytes_held(self) -> int:
        with self._lock:
            return sum(len(b) for _, b in self._buf)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)
