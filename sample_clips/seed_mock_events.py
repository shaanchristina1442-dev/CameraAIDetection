"""
Write a few fake events into events/events.jsonl so the dashboard can be
checked without running YOLO or calling Claude.

Usage:
    python sample_clips/seed_mock_events.py          # append mock events
    python sample_clips/seed_mock_events.py --clear  # remove mock events and exit
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config  # noqa: E402

MOCKS = [
    ("theft_concealment", 0.82, 2, "person present + high motion energy",
     "A person in a dark jacket picks up a small boxed item from the left shelf, looks toward the aisle end, and places the item inside their jacket.",
     "The item moves from shelf to inside clothing with no basket or bag visible, matching concealment more than normal shopping."),
    ("altercation_or_aggression", 0.64, 3, "person count change",
     "Two people near the register stand close together; one pushes the other's arm away and a third person steps in between them.",
     "Visible pushing and an intervening third person suggest a physical disagreement, though the frames are brief."),
    ("nothing_notable", 0.91, 1, "person present + high motion energy",
     "An employee restocks the middle shelf from a cart, placing items in rows and moving the cart down the aisle.",
     "Movements are consistent with routine restocking."),
    ("employee_unauthorized_giveaway", 0.38, 2, "person count change",
     "An employee behind the counter hands a small bag to a customer; the register screen is not visible in frame.",
     "A handoff occurs but the register is out of view, so a transaction may have happened off-camera. Low confidence."),
]


def placeholder_frame(label: str, i: int):
    img = np.full((240, 320, 3), (60 + i * 25, 70, 90), np.uint8)
    cv2.putText(img, "MOCK", (100, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(img, f"{label[:22]} #{i + 1}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return img


def main():
    os.makedirs(config.CLIPS_DIR, exist_ok=True)

    if "--clear" in sys.argv:
        # remove and stop. Previously this stripped the old mock rows and then
        # fell through to the append below, re-seeding them with new timestamps,
        # so it looked like clearing had done nothing.
        removed = 0
        if os.path.exists(config.EVENTS_LOG):
            with open(config.EVENTS_LOG) as f:
                lines = f.readlines()
            keep = [line for line in lines if '"mock": true' not in line]
            removed = len(lines) - len(keep)
            with open(config.EVENTS_LOG, "w") as f:
                f.writelines(keep)
        for path in glob.glob(os.path.join(config.CLIPS_DIR, "mock_*.jpg")):
            os.remove(path)
        print(f"removed {removed} mock event(s) and their frames")
        return

    now = int(time.time())
    with open(config.EVENTS_LOG, "a") as log:
        for idx, (category, conf, people, reason, desc, why) in enumerate(MOCKS):
            event_id = f"mock_{now}_{idx}"
            frames = []
            for i in range(3):
                name = f"{event_id}_frame{i}.jpg"
                cv2.imwrite(os.path.join(config.CLIPS_DIR, name), placeholder_frame(category, i))
                frames.append(f"clips/{name}")
            start = 30.0 + idx * 95
            record = {
                "event_id": event_id,
                "video": "mock_store_cam.mp4",
                "start_time": start,
                "end_time": start + 8,
                "frames": frames,
                "judgment": {
                    "category": category,
                    "confidence": conf,
                    "visible_description": desc,
                    "reasoning": why,
                    "recommend_human_review": category != "nothing_notable",
                },
                "peak_motion": 0.05,
                "person_count": people,
                "trigger_reason": reason,
                "alert": category != "nothing_notable" and conf >= config.ALERT_CONFIDENCE_FLOOR,
                "review": "pending",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mock": True,
            }
            log.write(json.dumps(record) + "\n")

    print(f"seeded {len(MOCKS)} mock events into {config.EVENTS_LOG}")


if __name__ == "__main__":
    main()
