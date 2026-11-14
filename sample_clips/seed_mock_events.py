"""
Write a few fake events into the database so the dashboard can be checked
without running YOLO or calling Claude.

Usage:
    python sample_clips/seed_mock_events.py          # add mock events
    python sample_clips/seed_mock_events.py --clear  # remove them and exit
"""
import glob
import os
import sys
import time
import uuid

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config          # noqa: E402
from storage import repository       # noqa: E402

MOCK_PREFIX = "mock_"
MOCK_CAMERA = "file:mock_store_cam"

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


def clear():
    ids = repository.event_ids_with_prefix(MOCK_PREFIX)
    for event_id in ids:
        repository.delete_event(event_id)
    frames = glob.glob(os.path.join(config.CLIPS_DIR, f"{MOCK_PREFIX}*.jpg"))
    for path in frames:
        os.remove(path)
    print(f"removed {len(ids)} mock event(s) and {len(frames)} frame(s)")


def main():
    os.makedirs(config.CLIPS_DIR, exist_ok=True)
    repository.init()

    if "--clear" in sys.argv:
        clear()
        return

    repository.ensure_camera(MOCK_CAMERA, name="mock_store_cam.mp4", source="mock")
    for idx, (category, conf, people, reason, desc, why) in enumerate(MOCKS):
        event_id = f"{MOCK_PREFIX}{uuid.uuid4().hex[:8]}"
        frames = []
        for i in range(3):
            name = f"{event_id}_frame{i}.jpg"
            cv2.imwrite(os.path.join(config.CLIPS_DIR, name), placeholder_frame(category, i))
            frames.append(f"clips/{name}")
        start = 30.0 + idx * 95
        repository.insert_event(
            event_id,
            judgment={
                "category": category, "confidence": conf,
                "visible_description": desc, "reasoning": why,
                "recommend_human_review": category != "nothing_notable",
            },
            frames=frames,
            video="mock_store_cam.mp4",
            camera_id=MOCK_CAMERA,
            start_time=start, end_time=start + 8,
            trigger_reason=reason, peak_motion=0.05, person_count=people,
            alert=category != "nothing_notable" and conf >= config.ALERT_CONFIDENCE_FLOOR,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    print(f"seeded {len(MOCKS)} mock events into {repository.db.DB_PATH}")


if __name__ == "__main__":
    main()
