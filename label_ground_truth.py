"""
label_ground_truth.py

Vatsalya Dabhi and Aditya Pandita
CS 5330 - Final Project

Phase 3 needs "ground truth" - what a human looking at the frame would
say is actually there, and what the correct guidance decision should
be. Labeling all ~157 sampled frames by hand was more than we had time
for given the deadline, so we hand-picked a smaller, representative set
of frames per clip (covering clear-path, obstacle-left, and stop cases)
and judged them visually using the contact sheets in
results/contact_sheets/. Only our 12 target obstacle classes are listed
in actual_objects, since anything outside that list is intentionally
ignored by the pipeline and shouldn't count against it.

This script writes those manual judgments into the matching rows of
results/labeling_template_<clip>.csv, leaving every other row blank
(evaluate.py only scores rows that have a label).
"""

import sys
import csv

# Manually judged ground truth, keyed by (clip, frame). Filled in by
# looking at results/contact_sheets/<clip>/sheet_*.jpg.
GROUND_TRUTH = {
    # IMG_5077: walking an open floor, then past a kitchen counter/oven.
    ("IMG_5077", 0):   ("chair", "path clear"),
    ("IMG_5077", 165): ("chair", "path clear"),
    ("IMG_5077", 300): ("chair", "path clear"),
    ("IMG_5077", 360): ("chair;dining table", "path clear"),
    ("IMG_5077", 405): ("chair", "obstacle left, move right"),
    ("IMG_5077", 465): ("oven;sink", "path clear"),
    ("IMG_5077", 495): ("oven", "path clear"),
    ("IMG_5077", 525): ("oven;sink", "path clear"),

    # IMG_5078: a chair (and briefly a small table) near a hallway wall,
    # later a kitchen counter/oven, then a stool near a couch. Includes
    # the cropped-chair miss case. NOTE: these were first labeled by
    # eyeballing resized contact-sheet thumbnails and several left/right
    # calls came out backwards (e.g. an oven that is objectively
    # right-of-center per its bounding box got labeled "obstacle left").
    # Relabeled by looking at the full-resolution frame for each one and
    # by cross-checking bbox center-x against the frame midline (1080/2).
    ("IMG_5078", 0):   ("chair", "path clear"),
    ("IMG_5078", 90):  ("chair", "path clear"),
    ("IMG_5078", 225): ("chair", "obstacle right, move left"),
    ("IMG_5078", 300): ("chair", "obstacle left, move right"),
    ("IMG_5078", 375): ("chair", "obstacle left, move right"),
    ("IMG_5078", 480): ("chair", "obstacle right, move left"),
    ("IMG_5078", 720): ("oven", "obstacle right, move left"),
    ("IMG_5078", 765): ("sink;oven", "obstacle right, move left"),
    ("IMG_5078", 855): ("chair", "obstacle left, move right"),
    ("IMG_5078", 885): ("chair", "obstacle left, move right"),

    # IMG_5079: a small table with a chair behind it, squarely blocking
    # the hallway ahead.
    ("IMG_5079", 180): ("chair;dining table", "stop"),
    ("IMG_5079", 210): ("chair;dining table", "stop"),
    ("IMG_5079", 240): ("chair;dining table", "stop"),
    ("IMG_5079", 270): ("chair;dining table", "stop"),
    ("IMG_5079", 300): ("chair;dining table", "stop"),
    ("IMG_5079", 360): ("chair;dining table", "stop"),
    ("IMG_5079", 405): ("chair", "stop"),
    ("IMG_5079", 450): ("chair", "stop"),
}


def apply_labels(clip_name):
    template_path = f"results/labeling_template_{clip_name}.csv"

    with open(template_path, newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else []

    labeled_count = 0
    for row in rows:
        key = (clip_name, int(row["frame"]))
        if key in GROUND_TRUTH:
            actual_objects, actual_decision = GROUND_TRUTH[key]
            row["actual_objects"] = actual_objects
            row["actual_decision"] = actual_decision
            labeled_count += 1

    with open(template_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{clip_name}: labeled {labeled_count} frames")


def main(argv):
    clips = sorted(set(clip for clip, _ in GROUND_TRUTH))
    for clip_name in clips:
        apply_labels(clip_name)


if __name__ == "__main__":
    main(sys.argv)
