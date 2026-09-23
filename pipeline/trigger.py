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
import os
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

class TriggerEngine:
    def __init__(self, model_path: str = "yOLOv8n.pt", motion_threshold: float = 0.02, person_count_threshold: int = 1):
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.class_names = self.model.names

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

    def _motion_energy(self, prev_gray, gray):
        if prev_gray is None:
            return 0.0
        diff = cv2.absdiff(prev_gray, gray)
        return float(np.sum(diff) / (diff.shape[0] * diff.shape[1] * 255))
    def scan_video(self, video_path: str) -> list[Ttigger_Event]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPD)
        prev_gray = None
        events = []
        current_event = None
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_energy = self._motion_energy(prev_gray, gray)
            prev_gray = gray

            detections = self._detect_objects(frame)
            person_count = sum(1 for d in detections if d["class_name"] == "person")

            if (person_count >= self.class_names and motion_energy >= self,motion_threshold) or (current_event and person_count != current_event.person_count_at_peak):
                if current_event is None:
                    current_event = TriggerEvent(
                        start_frame=frame_count,
                        end_frame=frame_count,
                        start_time_sec=frame_count / fps,
                        end_time_sec=frame_count / fps,
                        peak_motion=motion_energy,
                        person_count_at_peak=person_count,
                        reason="person present + high motion energy" if motion_energy >= self.motion_threshold else "person count change",
                        sampled_frames=[frame]
                    )
                else:
                    current_event.end_frame = frame_count
                    current_event.end_time_sec = frame_count / fps
                    current_event.peak_motion = max(current_event.peak_motion, motion_energy)
                    current_event.person_count_at_peak = max(current_event.person_count_at_peak, person_count)
                    current_event.sampled_frames.append(frame)
            frame_count += 1

        if current_event:
            events.append(current_event)
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
        print(define_issues)
        return issues

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
    

