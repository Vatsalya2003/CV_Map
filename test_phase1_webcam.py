"""
test_phase1_webcam.py

Vatsalya Dabhi and Aditya Pandita
CS 5330 - Final Project

One-off Phase 1 sanity check script (not part of the final pipeline).
Grabs a few seconds of webcam frames, runs the detector on each one,
measures real FPS, and saves the last annotated frame to results/ so we
can visually confirm the boxes and labels look right. This file can be
deleted once Phase 1 is confirmed working.
"""

import sys
import time
import cv2

from detector import Detector
from utils import draw_boxes, draw_fps


def main(argv):
    duration_seconds = 5

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    detector = Detector()

    frame_count = 0
    last_frame = None
    start_time = time.time()

    while time.time() - start_time < duration_seconds:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.detect(frame)
        frame = draw_boxes(frame, detections)
        last_frame = frame
        frame_count += 1

    cap.release()

    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"Processed {frame_count} frames in {elapsed:.1f}s -> {fps:.2f} FPS")

    if last_frame is not None:
        draw_fps(last_frame, fps)
        out_path = "results/phase1_webcam_snapshot.jpg"
        cv2.imwrite(out_path, last_frame)
        print(f"Saved annotated snapshot to {out_path}")
    else:
        print("No frames captured, nothing to save.")


if __name__ == "__main__":
    main(sys.argv)
