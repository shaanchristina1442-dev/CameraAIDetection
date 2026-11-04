"""
Trigger layer: cheap, fast pass over video using YOLO object detection
plus simple frame-differencing motion energy.
 
This layer's only job is to answer "is something happening here that
MIGHT be worth a closer (expensive) look?" It does NOT judge whether
something is theft, damage, etc. - that's the judgment layer's job.
 
Design choice: we trigger on "person present + high motion energy" or
"person count changes" as a generic proxy for "an event is occurring".
A production system would use pose-estimation (hand-to-shelf, hand-to-bag)
for a far more precise trigger; that's flagged as a future improvement
in the README rather than built here, since pose-based gesture detection
needs labeled retail-specific training data to be reliable.
"""


import cv2
import numpy as np
from dataclasses import dataclass
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

class TriggerEngine:
    def __init__(self, model_path: str = config.YOLO_MODEL, motion_threshold: float = config.MOTION_THRESHOLD, person_count_threshold: int = 1):
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.motion_threshold = motion_threshold
        self.person_count_threshold = person_count_threshold

    def _detect_objects(self, frame):
        results = self.model.predict(
            frame, conf = config.YOLO_CONF_THRESHOLD, verbose=False)[0]
        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2)
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
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_skip = max(1, config.FRAME_SKIP)
        # an event only ends after this many quiet frames, so a brief pause doesn't split it
        max_gap_frames = int(config.MIN_SECONDS_BETWEEN_EVENTS * fps)
        min_event_frames = int(config.MIN_EVENT_SECONDS * fps)
        max_event_frames = int(config.MAX_EVENT_SECONDS * fps)

        prev_gray = None
        prev_person_count = 0
        events = []
        current_event = None
        last_active_frame = 0
        frame_count = -1
        keep_stride = 1   # keep every Nth active frame of the current event
        active_ticks = 0

        def close(event):
            if event.end_frame - event.start_frame >= min_event_frames:
                events.append(event)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_skip:
                continue
            if total_frames and frame_count and frame_count % int(fps * 60) == 0:
                print(f"[trigger] {frame_count / fps / 60:.0f} / {total_frames / fps / 60:.0f} min scanned, {len(events)} events so far")

            gray = self._prep_gray(frame)
            motion_energy = self._motion_energy(prev_gray, gray)
            prev_gray = gray

            # YOLO is the slow part (~100ms/frame on CPU). With no event open and
            # nothing moving, nothing can trigger, so skip it on quiet footage.
            if current_event is None and motion_energy < self.motion_threshold:
                prev_person_count = 0
                continue

            detections = self._detect_objects(frame)
            person_count = sum(1 for d in detections if d["class_name"] == "person")
            person_boxes = [d["bbox"] for d in detections if d["class_name"] == "person"]

            if (person_count >= self.person_count_threshold and motion_energy >= self.motion_threshold) or (current_event and person_count != prev_person_count):
                last_active_frame = frame_count
                if current_event is None:
                    keep_stride, active_ticks = 1, 0
                    current_event = TriggerEvent(
                        start_frame=frame_count,
                        end_frame=frame_count,
                        start_time_sec=frame_count / fps,
                        end_time_sec=frame_count / fps,
                        peak_motion=motion_energy,
                        person_count_at_peak=person_count,
                        reason="person present + high motion energy" if motion_energy >= self.motion_threshold else "person count change",
                        sampled_frames=[frame],
                        sampled_boxes=[person_boxes]
                    )
                else:
                    current_event.end_frame = frame_count
                    current_event.end_time_sec = frame_count / fps
                    current_event.peak_motion = max(current_event.peak_motion, motion_energy)
                    current_event.person_count_at_peak = max(current_event.person_count_at_peak, person_count)
                    active_ticks += 1
                    if active_ticks % keep_stride == 0:
                        current_event.sampled_frames.append(frame)
                        current_event.sampled_boxes.append(person_boxes)
                    # cap memory on long events: drop every other frame and keep half as often from now on
                    if len(current_event.sampled_frames) > config.MAX_FRAMES_KEPT_PER_EVENT:
                        current_event.sampled_frames = current_event.sampled_frames[::2]
                        current_event.sampled_boxes = current_event.sampled_boxes[::2]
                        keep_stride *= 2
                    # split very long activity (e.g. a cashier always in view) into reviewable chunks
                    if current_event.end_frame - current_event.start_frame >= max_event_frames:
                        close(current_event)
                        current_event = None
            elif current_event is not None and frame_count - last_active_frame > max_gap_frames:
                close(current_event)
                current_event = None
            prev_person_count = person_count

        if current_event:
            close(current_event)
        cap.release()
        return events

    def find_issues(self, events: list[TriggerEvent]) -> list[str]:
        issues = []
        for event in events:
            if event.peak_motion < self.motion_threshold:
                issues.append(f"Event with low peak motion: {event.start_time_sec}s - {event.end_time_sec}s")
            if event.person_count_at_peak == 0:
                issues.append(f"Event with no persons detected: {event.start_time_sec}s - {event.end_time_sec}s")
        return issues

    def define_issues(self, events: list[TriggerEvent]) -> list[str]:
        issues = []
        if not issues:
            issues.append("No issues found on monitor")
        else:
            issues.append("Issues found on monitor")
            print(issues)
        return issues
    
    def recognize_face(self, frame) -> list[str]:
        return []  # TODO: not implemented yet
            
    def camera_results(self, events, issues: list[str]) -> None:
        print("Events:")
        for event in events:
            print(
                f"Start: {event.start_time_sec}s, End: {event.end_time_sec}s, "
                f"Peak Motion: {event.peak_motion}, "
                f"Person Count at Peak: {event.person_count_at_peak}"
            )
            print(f'Issue: {issues}')
        print("Camera results displayed.")
        print("Issues:")
        for issue in issues:
            print(f"- {issue}")
        return issues
        return issue
    

