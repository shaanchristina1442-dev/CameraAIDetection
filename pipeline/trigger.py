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
