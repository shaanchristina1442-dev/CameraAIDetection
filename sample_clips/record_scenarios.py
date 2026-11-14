"""
Record staged test footage from a webcam, with known ground truth.

The synthetic clip proves the pipeline runs; it cannot tell you whether the
thresholds suit real footage, because it is a photograph panned across a drawn
background. This records real video of real people in a real room: real
lighting, real motion, real compression artefacts. Not a store, but far closer
than anything generated.

The point is the LABELS. Each clip is recorded against a known expected
outcome, so you can measure precision and recall instead of eyeballing it.
That file (benchmarks/ground_truth.json) is the seed of the eval set in
roadmap task Y.

Usage:
    python sample_clips/record_scenarios.py                 # all scenarios
    python sample_clips/record_scenarios.py --list          # show them
    python sample_clips/record_scenarios.py --only browsing # just one
    python sample_clips/record_scenarios.py --camera 1      # a different camera

Set the camera up like a security camera: high, looking down at an angle,
covering a space with a shelf or table in it. Put a few small objects on it.
A phone on a shelf works as well as a laptop.

Staged footage is not real store footage - people behave differently when
performing. This gets the pipeline validated and the thresholds roughly
right; final tuning needs a real camera in a real shop. If you record anyone
other than yourself, ask them first.
"""
import argparse
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config  # noqa: E402

GROUND_TRUTH = os.path.join(config.PROJECT_ROOT, "benchmarks", "ground_truth.json")

# (name, seconds, expected category, what to do)
SCENARIOS = [
    ("quiet_room", 10, "__no_event__",
     "Stay OUT of frame. Nothing should happen at all.\n"
     "     Tests whether the motion threshold fires on an empty room."),

    ("walk_through", 12, "nothing_notable",
     "Walk across the frame and out. Touch nothing.\n"
     "     Should trigger an event, and be judged unremarkable."),

    ("browsing", 15, "nothing_notable",
     "Walk in, pick an item up, look at it, put it BACK, walk out.\n"
     "     Normal shopping. Must not be flagged."),

    ("own_bag", 15, "nothing_notable",
     "Carry a bag in. Reach into YOUR OWN bag, take something out,\n"
     "     put it back in, walk out. Do not touch the shelf.\n"
     "     THE IMPORTANT ONE: looks almost identical to concealment."),

    ("concealment_pocket", 15, "theft_concealment",
     "Walk in, take an item from the shelf, put it in your POCKET,\n"
     "     walk out. The one it is supposed to catch."),

    ("concealment_bag", 15, "theft_concealment",
     "Walk in, take an item from the shelf, put it into a BAG you\n"
     "     are carrying, walk out."),

    ("obscured_concealment", 15, "theft_concealment",
     "Same as concealment_pocket, but turn your back to the camera\n"
     "     as you do it. Tests behaviour when the view is blocked -\n"
     "     confidence should drop, not stay high."),

    ("two_people_handoff", 15, "employee_unauthorized_giveaway",
     "NEEDS TWO PEOPLE (skip with 's' if alone). One hands an item\n"
     "     to the other with no register in view."),
]


def open_camera(index: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"\nCould not open camera {index}.")
        print("  - If macOS has not asked for camera access, run this from Terminal.app")
        print("    so it can prompt, then allow it.")
        print("  - If you denied it before: System Settings > Privacy & Security >")
        print("    Camera, and enable your terminal app.")
        print("  - Another camera index may work: --camera 1")
        return None
    for _ in range(5):  # let exposure settle
        cap.read()
    return cap


def record(cap, path, seconds, fps, preview=True):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        print(f"  could not open {path} for writing")
        return 0

    interval, start, written = 1.0 / fps, time.monotonic(), 0
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= seconds:
            break
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
        if preview:
            shown = frame.copy()
            left = seconds - elapsed
            cv2.putText(shown, f"REC  {left:4.1f}s", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            try:
                cv2.imshow("recording - q to abort", shown)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                preview = False
        target = start + written * interval
        now = time.monotonic()
        if target > now:
            time.sleep(target - now)

    writer.release()
    return written


def countdown(cap, seconds=3, preview=True):
    start = time.monotonic()
    while True:
        left = seconds - (time.monotonic() - start)
        if left <= 0:
            return
        ok, frame = cap.read()
        if not ok or not preview:
            time.sleep(min(0.1, max(0.0, left)))
            continue
        shown = frame.copy()
        cv2.putText(shown, str(int(left) + 1), (shown.shape[1] // 2 - 40, shown.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 220, 255), 8)
        try:
            cv2.imshow("recording - q to abort", shown)
            cv2.waitKey(1)
        except cv2.error:
            preview = False


def main():
    ap = argparse.ArgumentParser(description="Record staged test footage with known ground truth.")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--fps", type=int, default=15, help="frames per second (default 15, CCTV-ish)")
    ap.add_argument("--only", help="record a single scenario by name")
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    ap.add_argument("--no-preview", action="store_true", help="skip the preview window")
    args = ap.parse_args()

    if args.list:
        print("\nScenarios:\n")
        for name, secs, expected, how in SCENARIOS:
            print(f"  {name:22} {secs:>3}s  expects: {expected}")
        print()
        return

    todo = SCENARIOS
    if args.only:
        todo = [s for s in SCENARIOS if s[0] == args.only]
        if not todo:
            sys.exit(f"No scenario called {args.only!r}. Try --list.")

    os.makedirs(config.FOOTAGE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(GROUND_TRUTH), exist_ok=True)

    cap = open_camera(args.camera)
    if cap is None:
        sys.exit(1)

    truth = {}
    if os.path.exists(GROUND_TRUTH):
        try:
            truth = json.load(open(GROUND_TRUTH))
        except json.JSONDecodeError:
            pass

    preview = not args.no_preview
    print("\n" + "=" * 68)
    print("  Set the camera up HIGH, angled down, covering a shelf or table.")
    print("  Put a few small objects on it. Stand back so you are fully in frame.")
    print("=" * 68)

    try:
        for i, (name, secs, expected, how) in enumerate(todo, 1):
            path = os.path.join(config.FOOTAGE_DIR, f"{name}.mp4")
            print(f"\n[{i}/{len(todo)}] {name}  ({secs}s, expects {expected})")
            print(f"     {how}")
            if os.path.exists(path):
                print(f"     (already recorded - will overwrite)")
            ans = input("     Enter to record, 's' to skip, 'q' to stop: ").strip().lower()
            if ans == "q":
                break
            if ans == "s":
                continue

            countdown(cap, 3, preview)
            frames = record(cap, path, secs, args.fps, preview)
            if frames == 0:
                print("     nothing captured - skipped")
                continue
            truth[f"{name}.mp4"] = {
                "expected_category": expected,
                "expect_event": expected != "__no_event__",
                "instruction": " ".join(how.split()),
                "seconds": secs,
                "frames": frames,
                "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(GROUND_TRUTH, "w") as f:
                json.dump(truth, f, indent=2, sort_keys=True)
            print(f"     saved {path}  ({frames} frames)")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    print(f"\nGround truth: {GROUND_TRUTH}  ({len(truth)} clip(s))")
    print("\nNext:")
    print("  1. Tune the trigger for free:")
    print("     python -m pipeline.run_pipeline footage/browsing.mp4 --dry-run")
    print("  2. Then judge for real (costs about a cent per event):")
    print("     python -m pipeline.run_pipeline footage/browsing.mp4")


if __name__ == "__main__":
    main()
