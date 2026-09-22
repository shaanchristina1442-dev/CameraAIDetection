
import os
import sys
import cv2
import json
import time
import anthropic


from . import config
from .trigger import TriggerEngine
from .judge import judge_event


def ensure_dirs():
    os.makedirs(config.CLIPS_DIR, exist_ok=True)

def save_event_frames(event_id:str, frames)-> list[str]:
    paths = []
    for i, frame in enumerate(frames):
        path = os.path.join(config.CLIPS_DIR, f"{event_id}_frame{i}.jpg")
        cv2.imwrite(path, frame)
        paths.append(path)
    return paths

def append_event_log(record: dict):
    with open(config.EVENTS_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

def run(video_path: str):
    ensure_dirs()

    if not os.path.exists(config.EVENTS_LOG):
        open(config.EVENTS_LOG, "w").close()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    engine = TriggerEngine()
    print(f"[pipeline] scanning {video_path} for candidate events")

    trigger_events = engine.scan_video(video_path)

    if not trigger_events:
        print("[pipeline] no candidate events found. Try lowering the threshold")
        return trigger_events

    for idx, event in enumerate(trigger_events):
        event_id = f"{int(time.time())}_{idx}"
        print(f"[pipeline] processing event {event_id} - {event['start_time']} to {event['end_time']}")

        print(f"({event.get('reason', '')})")

        frames = engine.extract_event_frames(video_path, event, config.FRAMES_PER_EVENT)
        frame_path = save_event_frames(event_id, frames)

        print(f"[pipeline]   -> sending {len(frames)} frames to Claude for judgment...")
        judgment = judge_event(frames, client)

        record = {
            "event_id": event_id,
            "start_time": event["start_time"],
            "end_time": event["end_time"],
            "frames": frame_path,
            "judgment": judgment,
            "peak_motion": event["peak_motion"],
            "trigger_engine": event["trigger_engine"],
            "review": "pending",

            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }

        json_record = json.dumps(record, indent=2)
        print(f"[pipeline]  -> event record:\n{json_record}")

        append_event_log(record)

        category = judgment.get("category", "unknown")
        confidence = judgment.get("confidence", 0.0)
        flag = "⚠️" if category == "suspicious" and confidence > 0.8 else ""
        print(f"[pipeline]  -> judgment: {category} (confidence: {confidence:.2f}) {flag}")

        print(f"[pipeline]  -> event record saved to {config.EVENTS_LOG}")
        print(f"run dashboard to review events: `python -m pipeline.dashboard`")


        if category == "suspicious" and confidence > 0.8:
            print(f"[pipeline]  -> event {event_id} flagged as suspicious. Consider reviewing it.")

    return trigger_events


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.run_pipeline <video_path>")
        sys.exit(1)
    run(sys.argv[1])




            

