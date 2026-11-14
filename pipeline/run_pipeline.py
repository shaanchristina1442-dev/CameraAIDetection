"""
Orchestrates one clip end to end: trigger scan -> Claude judgment -> events.jsonl.

Usage:
    python -m pipeline.run_pipeline <video_path> [--dry-run]

--dry-run runs only the YOLO trigger layer (no Claude calls, nothing logged),
which is the cheap way to tune thresholds on new store footage.
"""
import os
import sys
import uuid
import cv2
import json
import time


from . import config
from .trigger import TriggerEngine
from .judge import judge_event, crop_to_people
from storage import repository


def ensure_dirs():
    os.makedirs(config.CLIPS_DIR, exist_ok=True)
    repository.init()

def sample_frames(frames, boxes, n):
    """
    Pick n frames spread evenly across the event, with their person boxes.

    Frames and boxes are returned index-aligned: the judgment layer crops
    each frame using its own boxes, so they must not drift apart here.
    """
    if len(frames) <= n:
        picked = list(range(len(frames)))
    else:
        step = len(frames) / n
        picked = [int(i * step) for i in range(n)]
    return ([frames[i] for i in picked],
            [boxes[i] if i < len(boxes) else [] for i in picked])

def save_event_frames(event_id:str, frames)-> list[str]:
    """Save frames as JPEGs; returns paths relative to EVENTS_DIR for the dashboard."""
    paths = []
    for i, frame in enumerate(frames):
        name = f"{event_id}_frame{i}.jpg"
        cv2.imwrite(os.path.join(config.CLIPS_DIR, name), frame)
        paths.append(f"clips/{name}")
    return paths

def camera_for(video_path: str) -> str:
    """
    One camera row per source. For file runs that is the clip; from task M it
    is the actual camera, and events stay grouped the same way either side.
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    camera_id = f"file:{stem}"
    repository.ensure_camera(camera_id, name=os.path.basename(video_path), source=video_path)
    return camera_id

def is_alert(judgment: dict) -> bool:
    return (judgment.get("category") != "nothing_notable"
            and judgment.get("confidence", 0.0) >= config.ALERT_CONFIDENCE_FLOOR)

def fmt_time(sec: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(sec))

def run(video_path: str, engine: TriggerEngine | None = None, dry_run: bool = False):
    ensure_dirs()

    if not dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("[pipeline] ANTHROPIC_API_KEY is not set. Add it to .env or use --dry-run.")

    engine = engine or TriggerEngine()
    video_name = os.path.basename(video_path)
    camera_id = camera_for(video_path)
    print(f"[pipeline] scanning {video_name} for candidate events")

    trigger_events = engine.scan_video(video_path)

    if not trigger_events:
        print("[pipeline] no candidate events found. Try lowering MOTION_THRESHOLD in config.py")
        return trigger_events

    print(f"[pipeline] {len(trigger_events)} candidate event(s) found")

    for idx, event in enumerate(trigger_events):
        # timestamp keeps ids sorting chronologically; the random suffix stops
        # two clips processed in the same second from overwriting each other
        event_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        print(f"[pipeline] event {idx + 1}: {fmt_time(event.start_time_sec)} - {fmt_time(event.end_time_sec)} "
              f"({event.reason}, {event.person_count_at_peak} people, motion {event.peak_motion:.3f})")

        if dry_run:
            continue

        frames, boxes = sample_frames(event.sampled_frames, event.sampled_boxes,
                                      config.FRAMES_PER_JUDGMENT)
        # crop before saving, so what the dashboard shows a reviewer is exactly
        # what the model was given - the frames are the evidence for its answer
        frames = [crop_to_people(f, b) for f, b in zip(frames, boxes)]
        frame_paths = save_event_frames(event_id, frames)

        cropped = sum(1 for f, b in zip(frames, boxes) if b)
        print(f"[pipeline]   -> sending {len(frames)} frames to Claude "
              f"({cropped} cropped to people) for judgment...")
        judgment = judge_event(frames, event.reason)

        alert = is_alert(judgment)
        repository.insert_event(
            event_id,
            judgment=judgment,
            frames=frame_paths,
            video=video_name,
            camera_id=camera_id,
            start_time=event.start_time_sec,
            end_time=event.end_time_sec,
            trigger_reason=event.reason,
            peak_motion=event.peak_motion,
            person_count=event.person_count_at_peak,
            alert=alert,
        )

        category = judgment.get("category", "unknown")
        confidence = judgment.get("confidence", 0.0)
        flag = "⚠️  ALERT - needs human review" if alert else ""
        print(f"[pipeline]   -> judgment: {category} (confidence: {confidence:.2f}) {flag}")
        if "error" in judgment:
            print(f"[pipeline]   -> judge error: {judgment['error']}: {judgment.get('reasoning', '')}")

    if not dry_run:
        print(f"[pipeline] events saved to {repository.db.DB_PATH}")
        print("[pipeline] review them in the dashboard: python -m dashboard.app")

    return trigger_events


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print("Usage: python -m pipeline.run_pipeline <video_path> [--dry-run]")
        sys.exit(1)
    run(args[0], dry_run="--dry-run" in sys.argv)
