"""
Trigger layer: cheap, fast pass over frames using YOLO detection and
tracking plus simple frame-differencing motion energy.

This layer's only job is to answer "is something happening here that MIGHT
be worth a closer (expensive) look?" It does NOT judge whether something is
theft, damage, etc. - that's the judgment layer's job.

The detection logic lives in StreamingTrigger, which is fed one frame at a
time and returns an event when one closes. A file scan and a live camera are
then the same code with different frame sources: scan_video is a loop over
feed(), and the capture worker calls feed() as frames arrive. Writing it as
a self-contained loop over a file was what made live cameras a rewrite
rather than a new caller.

Tracking gives each person a stable id while they remain visible, which is
what lets the judge layer suppress forty alerts about one person loitering.
Ultralytics keeps tracker state on the model object, so a TriggerEngine
belongs to exactly one camera - sharing one across cameras would mix their
tracks together.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from ultralytics import YOLO

from . import config


@dataclass
class TriggerEvent:
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    peak_motion: float
    person_count_at_peak: int
    reason: str
    sampled_frames: list[np.ndarray]
    # person bounding boxes for each kept frame, index-aligned with
    # sampled_frames, so the judgment layer can crop to where people are
    sampled_boxes: list[list[tuple]]
    # tracker ids seen during the event; used to suppress repeat alerts about
    # the same person. Discarded when the event is written - never persisted
    # against an individual.
    track_ids: set = field(default_factory=set)
    camera_id: str = None
    source_name: str = None


class TriggerEngine:
    def __init__(self, model_path: str = config.YOLO_MODEL,
                 motion_threshold: float = config.MOTION_THRESHOLD,
                 person_count_threshold: int = 1,
                 track: bool = True):
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.motion_threshold = motion_threshold
        self.person_count_threshold = person_count_threshold
        self.track = track

    def _detect_objects(self, frame):
        """Detections for one frame, with a tracker id where tracking is on."""
        if self.track:
            results = self.model.track(
                frame, conf=config.YOLO_CONF_THRESHOLD, persist=True,
                tracker=config.TRACKER, verbose=False)[0]
        else:
            results = self.model.predict(
                frame, conf=config.YOLO_CONF_THRESHOLD, verbose=False)[0]

        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class_id": class_id,
                "class_name": self.class_names[class_id],
                "confidence": float(box.conf[0]),
                "bbox": (x1, y1, x2, y2),
                # None until the tracker is confident enough to assign an id
                "track_id": int(box.id[0]) if getattr(box, "id", None) is not None else None,
            })
        return detections

    def _prep_gray(self, frame):
        # downscale + blur so camera noise and compression flicker don't count as motion
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (320, max(1, int(h * 320 / w))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def _motion_energy(self, prev_gray, gray):
        """Fraction of pixels (0-1) that changed noticeably since the last checked frame."""
        if prev_gray is None:
            return 0.0
        diff = cv2.absdiff(prev_gray, gray)
        return float(np.count_nonzero(diff > config.MOTION_PIXEL_DELTA) / diff.size)

    def scan_video(self, video_path: str) -> list[TriggerEvent]:
        """Scan a whole file. A loop over StreamingTrigger - no logic of its own."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_skip = max(1, config.FRAME_SKIP)
        stream = StreamingTrigger(self, fps)

        events, frame_no = [], -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            if frame_no % frame_skip:
                continue
            if total_frames and frame_no and frame_no % int(fps * 60) == 0:
                print(f"[trigger] {frame_no / fps / 60:.0f} / {total_frames / fps / 60:.0f} "
                      f"min scanned, {len(events)} events so far")
            event = stream.feed(frame_no, frame)
            if event:
                events.append(event)

        cap.release()
        final = stream.flush()
        if final:
            events.append(final)
        return events


class StreamingTrigger:
    """
    Fed one sampled frame at a time; returns a TriggerEvent when one closes.

    The caller decides which frames to feed (every FRAME_SKIP-th, normally) and
    supplies the frame number, so timings stay correct whether frames arrive
    from a file as fast as it decodes or from a camera in real time.
    """

    def __init__(self, engine: TriggerEngine, fps: float,
                 camera_id: str = None, source_name: str = None):
        self.engine = engine
        self.fps = fps or 30.0
        self.camera_id = camera_id
        self.source_name = source_name

        self.max_gap_frames = int(config.MIN_SECONDS_BETWEEN_EVENTS * self.fps)
        self.min_event_frames = int(config.MIN_EVENT_SECONDS * self.fps)
        self.max_event_frames = int(config.MAX_EVENT_SECONDS * self.fps)

        self._prev_gray = None
        self._prev_person_count = 0
        self._event = None
        self._last_active_frame = 0
        self._keep_stride = 1
        self._active_ticks = 0

    def _close(self):
        """Return the open event if it is long enough to be worth judging."""
        event, self._event = self._event, None
        if event is None:
            return None
        if event.end_frame - event.start_frame >= self.min_event_frames:
            return event
        return None

    def feed(self, frame_no: int, frame) -> TriggerEvent | None:
        gray = self.engine._prep_gray(frame)
        motion = self.engine._motion_energy(self._prev_gray, gray)
        self._prev_gray = gray

        # YOLO is the slow part. With no event open and nothing moving, nothing
        # can trigger, so skip it entirely on quiet footage.
        if self._event is None and motion < self.engine.motion_threshold:
            self._prev_person_count = 0
            return None

        detections = self.engine._detect_objects(frame)
        people = [d for d in detections if d["class_name"] == "person"]
        person_count = len(people)
        boxes = [d["bbox"] for d in people]
        ids = {d["track_id"] for d in people if d["track_id"] is not None}

        active = ((person_count >= self.engine.person_count_threshold
                   and motion >= self.engine.motion_threshold)
                  or (self._event and person_count != self._prev_person_count))

        closed = None
        if active:
            self._last_active_frame = frame_no
            if self._event is None:
                self._keep_stride, self._active_ticks = 1, 0
                self._event = TriggerEvent(
                    start_frame=frame_no,
                    end_frame=frame_no,
                    start_time_sec=frame_no / self.fps,
                    end_time_sec=frame_no / self.fps,
                    peak_motion=motion,
                    person_count_at_peak=person_count,
                    reason="person present + high motion energy",
                    sampled_frames=[frame],
                    sampled_boxes=[boxes],
                    track_ids=set(ids),
                    camera_id=self.camera_id,
                    source_name=self.source_name,
                )
            else:
                ev = self._event
                ev.end_frame = frame_no
                ev.end_time_sec = frame_no / self.fps
                ev.peak_motion = max(ev.peak_motion, motion)
                ev.person_count_at_peak = max(ev.person_count_at_peak, person_count)
                ev.track_ids |= ids
                self._active_ticks += 1
                if self._active_ticks % self._keep_stride == 0:
                    ev.sampled_frames.append(frame)
                    ev.sampled_boxes.append(boxes)
                # cap memory on long events: thin what we hold and keep less often
                if len(ev.sampled_frames) > config.MAX_FRAMES_KEPT_PER_EVENT:
                    ev.sampled_frames = ev.sampled_frames[::2]
                    ev.sampled_boxes = ev.sampled_boxes[::2]
                    self._keep_stride *= 2
                # split sustained activity (a cashier always in view) into
                # chunks a reviewer can actually watch
                if ev.end_frame - ev.start_frame >= self.max_event_frames:
                    closed = self._close()
        elif self._event is not None and frame_no - self._last_active_frame > self.max_gap_frames:
            closed = self._close()

        self._prev_person_count = person_count
        return closed

    def flush(self) -> TriggerEvent | None:
        """End of stream: hand back whatever is still open."""
        return self._close()
