"""
Trigger worker: motion, detection and tracking on the frames capture hands it.
"""
import queue
import threading

from .. import config
from ..trigger import StreamingTrigger
from .capture import END_OF_STREAM


class TriggerWorker(threading.Thread):
    def __init__(self, engine, capture, *, out_queue: queue.Queue,
                 stop_event: threading.Event):
        super().__init__(name=f"trigger:{capture.camera_id}", daemon=True)
        self.engine = engine
        self.capture = capture
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.stream = StreamingTrigger(engine, capture.fps,
                                       camera_id=capture.camera_id,
                                       source_name=capture.source_name)
        self.events_found = 0
        self.finished = False

    def _with_preroll(self, event):
        """
        Prepend a little of what happened just before the trigger fired.

        These frames carry no bounding boxes, so the judge sends them whole
        rather than cropped - giving the model the wider scene alongside the
        close crops, which is the context it said tight cropping costs it.
        """
        n = config.PREROLL_FRAMES_IN_EVENT
        if n <= 0:
            return event
        span = int(config.PREROLL_SECONDS * self.capture.fps)
        window = self.capture.ring.window(
            event.start_frame - span, event.start_frame - 1, limit=n)
        if window:
            event.sampled_frames[0:0] = [f for _, f in window]
            event.sampled_boxes[0:0] = [[] for _ in window]
            event.start_time_sec = min(event.start_time_sec,
                                       window[0][0] / self.capture.fps)
        return event

    def _emit(self, event):
        if event is None:
            return
        self.events_found += 1
        self.out_queue.put(self._with_preroll(event))

    def run(self):
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.capture.out_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is END_OF_STREAM:
                    break
                frame_no, frame = item
                self._emit(self.stream.feed(frame_no, frame))
        finally:
            self._emit(self.stream.flush())
            self.finished = True
            self.out_queue.put(END_OF_STREAM)

    def stats(self) -> dict:
        return {"events": self.events_found}
