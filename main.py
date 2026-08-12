"""
main.py

Vatsalya Dabhi and Aditya Pandita
CS 5330 - Final Project

This is the file that actually runs the whole pipeline. For each frame:
  1. Grab a frame from the camera (or a recorded video file).
  2. Run the detector to find obstacles.
  3. Work out a left/right guidance decision from those obstacles.
  4. Speak the decision out loud if it's new (or the cooldown passed).
  5. Draw a debug overlay (boxes, zone status, FPS) and show/save it.

Usage:
  python main.py                    -> use the live webcam (camera 0)
  python main.py path/to/video.mp4  -> run on a recorded video file instead

Add --save to also write the annotated video to results/, --no-window
to skip opening a live display window (useful for headless test runs),
and --log to write a per-frame CSV to results/ (frame number, decision,
zone statuses, and every detection) for Phase 3 evaluation.
"""

import sys
import os
import csv
import time

import cv2

from detector import Detector
from guidance import Guidance
from utils import draw_boxes, draw_fps
import zones


# If inference is too slow on a given machine, we don't have to run the
# detector on every single frame. PROCESS_EVERY_N = 1 means every frame,
# 2 means every other frame, etc. Detections from a skipped frame are
# just reused until the next processed frame.
PROCESS_EVERY_N = 1


def draw_zone_status(frame, decision, left_status, right_status):
    """
    Draw the current decision and each zone's status in the top-right
    area of the frame, and a vertical line showing the left/right split.
    """
    height, width = frame.shape[:2]
    midline = width // 2

    # Line down the middle so we can visually see the left/right split.
    cv2.line(frame, (midline, 0), (midline, height), (255, 255, 0), 1)

    cv2.putText(frame, f"decision: {decision}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"left: {left_status}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"right: {right_status}", (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame


def detections_to_string(detections):
    """
    Turn a list of Detection objects into one compact string for a CSV
    cell, e.g. "person:0.83:10,20,100,300;chair:0.55:400,200,600,470".
    Kept as one column instead of many so the CSV stays easy to read.
    """
    parts = []
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        parts.append(f"{d.label}:{d.conf:.2f}:{x1},{y1},{x2},{y2}")
    return ";".join(parts)


def run_pipeline(video_source, show_window=True, save_path=None, log_path=None):
    """
    Open the video source (webcam index or file path) and run the full
    detect -> decide -> speak -> display loop until the video ends or
    the user presses 'q'.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Could not open video source: {video_source}")
        return

    detector = Detector()
    guidance = Guidance()

    writer = None
    log_file = None
    log_writer = None
    if log_path:
        log_file = open(log_path, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["frame", "decision", "left_status", "right_status", "detections"])

    frame_count = 0
    last_detections = []
    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_height, frame_width = frame.shape[:2]

        # Only run the (expensive) detector every PROCESS_EVERY_N frames.
        # On skipped frames we just reuse the last known detections, so
        # the on-screen boxes don't disappear between detection frames.
        if frame_count % PROCESS_EVERY_N == 0:
            last_detections = detector.detect(frame)
        detections = last_detections

        decision, left_status, right_status = zones.decide(
            detections, frame_width, frame_height
        )
        guidance.speak_if_needed(decision)

        # Debug overlay: boxes, FPS, decision, zone split line.
        elapsed = time.time() - start_time
        fps = (frame_count + 1) / elapsed if elapsed > 0 else 0
        frame = draw_boxes(frame, detections)
        frame = draw_fps(frame, fps)
        frame = draw_zone_status(frame, decision, left_status, right_status)

        if save_path:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(save_path, fourcc, 20.0,
                                          (frame_width, frame_height))
            writer.write(frame)

        if log_writer:
            log_writer.writerow([
                frame_count, decision, left_status, right_status,
                detections_to_string(detections),
            ])

        if show_window:
            cv2.imshow("Obstacle Guidance (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_count += 1

    cap.release()
    if writer is not None:
        writer.release()
    if log_file is not None:
        log_file.close()
        print(f"Saved per-frame log to {log_path}")
    if show_window:
        cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"Processed {frame_count} frames in {total_time:.1f}s -> {avg_fps:.2f} FPS avg")


def main(argv):
    # Default to the live webcam (camera index 0).
    video_source = 0
    show_window = True
    save_path = None
    log_path = None

    args = argv[1:]
    for arg in list(args):
        if arg == "--no-window":
            show_window = False
            args.remove(arg)
        elif arg == "--save":
            save_path = "results/main_output.mp4"
            args.remove(arg)
        elif arg == "--log":
            args.remove(arg)
            log_path = "results/log_live.csv"

    if args:
        video_source = args[0]
        if log_path:
            # Name the log file after the video, e.g.
            # test_footage/hallway1.mp4 -> results/log_hallway1.csv
            clip_name = os.path.splitext(os.path.basename(video_source))[0]
            log_path = f"results/log_{clip_name}.csv"

    run_pipeline(video_source, show_window=show_window,
                 save_path=save_path, log_path=log_path)


if __name__ == "__main__":
    main(sys.argv)
