"""
extract_labeling_samples.py

Vatsalya Dabhi and Aditya Pandita
CS 5330 - Final Project

Phase 3 evaluation needs "ground truth" - what a human says is actually
in each frame, and what a human thinks the correct guidance decision is.
We can't label every single frame (there are hundreds per clip), so this
script samples every Nth frame from each log_<clip>.csv, saves that frame
as an image to look at, and builds a CSV template with the pipeline's
own prediction already filled in next to two empty columns for us to
fill in by hand: actual_objects and actual_decision.

Usage:
  python extract_labeling_samples.py
"""

import sys
import os
import csv

import cv2

# Every Nth frame gets pulled out for manual labeling. Around 2300 total
# frames across 3 clips, so every 15th frame gives us roughly 150
# frames to label by hand, which is enough for a rough precision/recall
# and decision-accuracy number without being an unreasonable amount of
# manual work.
SAMPLE_EVERY_N = 15

CLIPS = ["IMG_5077", "IMG_5078", "IMG_5079", "IMG_5089", "IMG_5091"]


def extract_for_clip(clip_name):
    """
    For one clip, read its log CSV (written by main.py --log), pick out
    every Nth logged row, save the matching video frame as a .jpg, and
    write a labeling template CSV with those rows plus two blank columns.
    """
    video_path = f"test_footage/{clip_name}.MOV"
    log_path = f"results/log_{clip_name}.csv"
    frames_dir = f"results/labeling_frames/{clip_name}"
    template_path = f"results/labeling_template_{clip_name}.csv"

    if not os.path.exists(log_path):
        print(f"Skipping {clip_name}, no log file found at {log_path}")
        return

    os.makedirs(frames_dir, exist_ok=True)

    # Read the frame numbers and predictions the pipeline already logged.
    with open(log_path, newline="") as f:
        rows = list(csv.DictReader(f))

    sampled_rows = [r for r in rows if int(r["frame"]) % SAMPLE_EVERY_N == 0]

    cap = cv2.VideoCapture(video_path)

    with open(template_path, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow([
            "clip", "frame", "image_file",
            "predicted_decision", "predicted_detections",
            "actual_objects", "actual_decision",
        ])

        for row in sampled_rows:
            frame_num = int(row["frame"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ok, frame = cap.read()
            if not ok:
                continue

            image_file = f"{clip_name}_frame{frame_num}.jpg"
            cv2.imwrite(os.path.join(frames_dir, image_file), frame)

            writer.writerow([
                clip_name, frame_num, image_file,
                row["decision"], row["detections"],
                "",  # actual_objects: fill in by hand, e.g. "person;chair"
                "",  # actual_decision: fill in by hand, e.g. "stop"
            ])

    cap.release()
    print(f"{clip_name}: sampled {len(sampled_rows)} frames -> {frames_dir}/ and {template_path}")


def main(argv):
    for clip_name in CLIPS:
        extract_for_clip(clip_name)

    print()
    print("Next step: open each results/labeling_frames/<clip>/ folder, look at the")
    print("images, and fill in the 'actual_objects' and 'actual_decision' columns")
    print("in the matching results/labeling_template_<clip>.csv file by hand.")


if __name__ == "__main__":
    main(sys.argv)
