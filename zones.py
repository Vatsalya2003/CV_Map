"""
zones.py

Vatsalya Dabhi and Aditya Pandita
CS 5330 - Final Project

This file decides what to tell the user based on WHERE obstacles are in
the frame, not just whether they exist. We split the frame into a left
half and a right half. If something "very near" is only on one side, we
tell the user to move to the other side. If it's near/very near on both
sides (or a box sits right on the middle line), we tell them to stop.
"""

import sys

from distance import closeness_score, bucket

# A box whose center is within this fraction of the frame width from the
# midline counts as being in BOTH zones, not just one. Without this, an
# obstacle that is really in the middle of the path could get assigned to
# only the left or only the right zone by chance, which would wrongly
# tell the user the other side is clear.
CENTER_MARGIN_FRACTION = 0.10


def get_zones_for_detection(bbox, frame_width):
    """
    Decide which zone(s) a single detection belongs to: "left", "right",
    or both if it's close enough to the midline.
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    midline = frame_width / 2
    margin = frame_width * CENTER_MARGIN_FRACTION

    zones = []
    if center_x < midline + margin:
        zones.append("left")
    if center_x > midline - margin:
        zones.append("right")

    return zones


def worst_bucket(buckets):
    """
    Given a list of bucket labels for one zone, return the closest one.
    "very_near" beats "near" beats "far" beats nothing at all.
    """
    if "very_near" in buckets:
        return "very_near"
    elif "near" in buckets:
        return "near"
    elif "far" in buckets:
        return "far"
    else:
        return "clear"


def decide(detections, frame_width, frame_height):
    """
    Main entry point: take a list of Detections plus the frame size, and
    return a plain-English guidance string.
    """
    left_buckets = []
    right_buckets = []

    for d in detections:
        score = closeness_score(d.bbox, frame_height)
        b = bucket(score)

        zones = get_zones_for_detection(d.bbox, frame_width)
        if "left" in zones:
            left_buckets.append(b)
        if "right" in zones:
            right_buckets.append(b)

    left_status = worst_bucket(left_buckets)
    right_status = worst_bucket(right_buckets)

    # Only "very_near" forces an action; "near" is informational only
    # (shown in the debug overlay/log, not acted on). We tried promoting
    # "near" to also block a zone during Phase 3 evaluation, since a
    # human would warn about a hallway chair before it's literally
    # very_near. That change did fix those hallway cases, but it broke
    # kitchen-appliance cases: box-height-ratio is a poor distance proxy
    # for physically large fixed objects (an oven looks "near" from its
    # height alone even when it's actually far across the room, since
    # real ovens are tall). Across our labeled evaluation frames, the
    # "near blocks too" version scored WORSE overall (13/26 correct)
    # than this simpler very_near-only version (15/26 correct), so we
    # kept this one. See NOTES.md for the full comparison.
    left_blocked = left_status == "very_near"
    right_blocked = right_status == "very_near"

    # Decision table, from the implementation plan:
    if not left_blocked and not right_blocked:
        decision = "path clear"
    elif left_blocked and not right_blocked:
        decision = "obstacle left, move right"
    elif right_blocked and not left_blocked:
        decision = "obstacle right, move left"
    else:
        decision = "stop"

    return decision, left_status, right_status


def main(argv):
    """
    Quick manual test using fake Detection-like objects, just to check the
    decision table produces the expected string for a few simple cases.
    """
    from detector import Detection

    frame_width, frame_height = 640, 480

    # Case 1: nothing in the frame -> path clear
    print(decide([], frame_width, frame_height))

    # Case 2: a big, low box on the left side -> obstacle left
    left_box = Detection((0, 200, 200, 470), "chair", 0.9)
    print(decide([left_box], frame_width, frame_height))

    # Case 3: a big, low box right on the midline -> stop
    center_box = Detection((280, 200, 360, 470), "person", 0.9)
    print(decide([center_box], frame_width, frame_height))


if __name__ == "__main__":
    main(sys.argv)
