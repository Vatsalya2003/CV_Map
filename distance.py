"""
distance.py

Vatsalya Dabhi and Aditya Pandita
CS 5330 - Final Project

We are NOT doing real depth estimation in this project (no meters, no
depth model). Instead we use a simple proxy: bigger box + lower in the
frame = probably closer to the camera. This is just a few lines of math,
not a model, and it is good enough to tell "close" from "far" for our
walking-guidance use case.
"""

import sys

# Thresholds on the closeness score that decide which bucket a detection
# falls into. These are starting values, tuned by hand while testing on
# real hallway/room footage in Phase 3.
FAR_THRESHOLD = 0.15
VERY_NEAR_THRESHOLD = 0.35


def closeness_score(bbox, frame_height):
    """
    Turn one bounding box into a single closeness score between 0 and 1.

    Two things make an object look "closer" in a normal eye-level camera
    shot:
      1. Its box takes up more of the frame's height (it's bigger).
      2. It sits lower in the frame (closer to the ground/camera).

    We combine these two 0-1 signals by multiplying them, so an object
    only gets a high score if it is both large AND low in the frame.
    """
    x1, y1, x2, y2 = bbox

    # Signal 1: how tall is the box compared to the whole frame?
    box_height = y2 - y1
    height_ratio = box_height / frame_height

    # Signal 2: how far down the frame is the bottom of the box?
    # y2 (box bottom) close to frame_height means it's low in the frame.
    position_weight = y2 / frame_height

    score = height_ratio * position_weight
    return score


def bucket(score):
    """
    Turn a closeness score into one of three simple labels.
    """
    if score < FAR_THRESHOLD:
        return "far"
    elif score < VERY_NEAR_THRESHOLD:
        return "near"
    else:
        return "very_near"


def main(argv):
    """
    Quick manual test with a made-up box and frame height, just to check
    the math produces sensible buckets.
    """
    frame_height = 480

    # A small box near the top of the frame -> should be "far".
    far_box = (100, 50, 150, 90)
    # A medium box around the middle of the frame -> should be "near".
    mid_box = (100, 200, 300, 320)
    # A large box filling the bottom of the frame -> should be "very_near".
    close_box = (50, 150, 550, 470)

    for name, box in [("far_box", far_box), ("mid_box", mid_box), ("close_box", close_box)]:
        score = closeness_score(box, frame_height)
        print(f"{name}: score={score:.3f} -> {bucket(score)}")


if __name__ == "__main__":
    main(sys.argv)
