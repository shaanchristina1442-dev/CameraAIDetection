"""
Generate a short test clip with a quiet stretch, a burst of people moving
through frame, then quiet again - enough to exercise the trigger layer.

Usage:
    python sample_clips/make_synthetic_clip.py [output_path]

Uses the sample photos that ship with ultralytics (real people, so YOLO
actually detects them). This only tests that the pipeline runs; it is not
a substitute for real store footage when tuning thresholds.
"""
import os
import sys

import cv2
import numpy as np
from ultralytics.utils import ASSETS

W, H, FPS = 640, 480, 15
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "synthetic_demo.mp4")


def background():
    # flat "store floor" with a couple of shelves
    bg = np.full((H, W, 3), (190, 195, 200), np.uint8)
    for x in (40, 460):
        cv2.rectangle(bg, (x, 60), (x + 140, 420), (120, 100, 80), -1)
        for y in range(90, 420, 60):
            cv2.line(bg, (x, y), (x + 140, y), (80, 65, 50), 3)
    return bg


def main():
    people = cv2.resize(cv2.imread(str(ASSETS / "bus.jpg")), (W // 2, H))
    bg = background()
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    quiet, active = 3 * FPS, 4 * FPS
    for _ in range(quiet):
        writer.write(bg)
    for i in range(active):
        # slide the people across the frame left to right
        frame = bg.copy()
        x = int(-W // 2 + (W + W // 2) * i / active)
        x0, x1 = max(0, x), min(W, x + W // 2)
        if x1 > x0:
            frame[:, x0:x1] = people[:, x0 - x:x1 - x]
        writer.write(frame)
    for _ in range(quiet):
        writer.write(bg)

    writer.release()
    print(f"wrote {OUT} ({(2 * quiet + active) / FPS:.0f}s, people visible from 3s to 7s)")


if __name__ == "__main__":
    main()
