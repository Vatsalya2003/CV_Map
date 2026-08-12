"""
utils.py

Vatsalya Dabhi and Aditya Pandita
CS 5330 - Final Project

Small helper functions shared by other files. Right now this is just the
debug-drawing function that draws boxes and labels on a frame so we can see
what the detector found while we are testing.
"""

import sys
import cv2


def draw_boxes(frame, detections):
    """
    Draw a rectangle and a text label for each detection on top of the
    frame. This does not change the original detections, it only draws on
    the image for us to look at while debugging.
    """
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d.bbox]

        # Green box around the detected object.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label text just above the box, e.g. "person 0.87"
        text = f"{d.label} {d.conf:.2f}"
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.putText(frame, text, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


def draw_fps(frame, fps):
    """
    Draw the current frames-per-second number in the top-left corner of
    the frame, so we can see how fast the pipeline is actually running.
    """
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame


def main(argv):
    """
    This file is just a helper module, there is nothing to run directly.
    """
    print("utils.py has no standalone behavior, import it from other files.")


if __name__ == "__main__":
    main(sys.argv)
