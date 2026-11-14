"""
Normalise a recording into footage/ so the pipeline can read it.

Real footage arrives in whatever shape the camera, the app or the phone
produced. The commonest problem is rotation: a screen recording of a camera
app is portrait, so the camera's landscape view ends up on its side. YOLO is
trained on upright people and detects almost nothing in a sideways frame -
measured on one such clip, 1 person found across 8 sample frames rotated
wrong, 12 found rotated right.

Usage:
    python sample_clips/prep_footage.py <input> --name browsing
    python sample_clips/prep_footage.py <input> --name cafe --rotate ccw
    python sample_clips/prep_footage.py <input> --name cafe --detect-rotation

--detect-rotation tries each of the four rotations on sample frames and picks
whichever YOLO finds the most people in. Use it when you are unsure; it costs
a few seconds and no API calls.
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config  # noqa: E402

ROTATIONS = {
    "none": None,
    "cw": cv2.ROTATE_90_CLOCKWISE,
    "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}


def detect_rotation(path, samples=8):
    """Pick the rotation YOLO finds the most people in."""
    from ultralytics import YOLO
    model = YOLO(config.YOLO_MODEL)
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    counts = {k: 0 for k in ROTATIONS}
    for i in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / samples))
        ok, frame = cap.read()
        if not ok:
            continue
        for name, code in ROTATIONS.items():
            img = frame if code is None else cv2.rotate(frame, code)
            res = model.predict(img, conf=config.YOLO_CONF_THRESHOLD, verbose=False)[0]
            counts[name] += sum(1 for b in res.boxes if model.names[int(b.cls[0])] == "person")
    cap.release()
    best = max(counts, key=counts.get)
    print("  people found per rotation: " +
          ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])))
    if counts[best] == 0:
        print("  no people found in any rotation - leaving as-is")
        return "none"
    return best


def main():
    ap = argparse.ArgumentParser(description="Normalise footage into footage/.")
    ap.add_argument("input", help="source video")
    ap.add_argument("--name", required=True, help="output name (without extension)")
    ap.add_argument("--rotate", choices=list(ROTATIONS), default="none")
    ap.add_argument("--detect-rotation", action="store_true",
                    help="choose the rotation automatically")
    ap.add_argument("--fps", type=float, default=0,
                    help="output fps (0 keeps the source rate)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"No such file: {args.input}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"Could not decode {args.input}. Convert it first, e.g.\n"
                 f"  ffmpeg -i {args.input!r} -c:v libx264 out.mp4")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"source: {w}x{h} {src_fps:.0f}fps {total} frames {total/src_fps:.0f}s")

    rotate = args.rotate
    if args.detect_rotation:
        cap.release()
        rotate = detect_rotation(args.input)
        print(f"  chosen: {rotate}")
        cap = cv2.VideoCapture(args.input)

    code = ROTATIONS[rotate]
    out_w, out_h = (h, w) if code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE) else (w, h)
    fps = args.fps or src_fps

    os.makedirs(config.FOOTAGE_DIR, exist_ok=True)
    out_path = os.path.join(config.FOOTAGE_DIR, f"{args.name}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        sys.exit(f"Could not open {out_path} for writing")

    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if code is not None:
            frame = cv2.rotate(frame, code)
        writer.write(frame)
        written += 1
        if total and written % 1000 == 0:
            print(f"  {written}/{total} frames")

    cap.release()
    writer.release()
    print(f"wrote {out_path}  {out_w}x{out_h} {fps:.0f}fps {written} frames")
    print(f"\nNext (free):  python -m pipeline.run_pipeline footage/{args.name}.mp4 --dry-run")


if __name__ == "__main__":
    main()
