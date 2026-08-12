"""
detector.py

Vatsalya Dabhi and Aditya Pandita
CS 5330 - Final Project

This file loads a pretrained YOLOv8n model and uses it to find objects in a
video frame. We only care about a small set of "obstacle-like" classes from
COCO (person, chair, couch, etc.) since our project is about indoor
navigation, not general object detection. So after YOLO gives us every
detection it found, we throw away anything that isn't in our class list or
that has low confidence.
"""

import sys
import cv2
import torch
from ultralytics import YOLO


# The only classes we care about for this project. COCO does not have a
# "door" class, so we use a few large fixed objects (bench, refrigerator,
# oven, sink) as stand-ins for door-sized obstacles. toilet/microwave/
# laptop/vase/suitcase/book were added after Phase 3 testing to broaden
# indoor coverage, on request, since the original 12 missed some common
# household obstacles.
OBSTACLE_CLASSES = [
    "person", "chair", "couch", "dining table", "potted plant",
    "backpack", "bench", "refrigerator", "oven", "sink", "tv", "bed",
    "toilet", "microwave", "laptop", "vase", "suitcase", "book",
]

# Confidence threshold per class. Every class starts at the same value.
# This is a dictionary so that later, if one class (like "backpack") turns
# out to be noisy, we can raise its threshold without touching the others.
DEFAULT_CONF_THRESHOLD = 0.4


class Detection:
    """
    Simple container for one detected object in a frame.
    bbox is (x1, y1, x2, y2) in pixel coordinates.
    """

    def __init__(self, bbox, label, conf):
        self.bbox = bbox
        self.label = label
        self.conf = conf

    def __repr__(self):
        return f"Detection({self.label}, conf={self.conf:.2f}, bbox={self.bbox})"


class Detector:
    """
    Wraps a YOLOv8n model and returns only the obstacle classes we care
    about, above their confidence threshold.
    """

    def __init__(self, model_path="yolov8n.pt", classes=None, conf_thresholds=None):
        # Pick MPS (Apple GPU) if available, otherwise fall back to CPU.
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Load the pretrained YOLOv8n weights (trained on COCO's 80 classes).
        self.model = YOLO(model_path)

        # The list of class names we allow through.
        self.classes = classes if classes is not None else OBSTACLE_CLASSES

        # Per-class confidence threshold, defaulting everything to the same
        # value unless the caller overrides specific classes.
        self.conf_thresholds = {c: DEFAULT_CONF_THRESHOLD for c in self.classes}
        if conf_thresholds:
            self.conf_thresholds.update(conf_thresholds)

    def detect(self, frame):
        """
        Run YOLO on one frame and return a list of Detection objects,
        filtered down to our obstacle classes and confidence thresholds.
        """
        # verbose=False just stops YOLO from printing to the terminal every frame.
        results = self.model(frame, device=self.device, verbose=False)[0]

        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            label = self.model.names[class_id]
            conf = float(box.conf[0])

            # Skip anything that isn't one of our obstacle classes.
            if label not in self.classes:
                continue

            # Skip low-confidence detections for this class.
            if conf < self.conf_thresholds[label]:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection((x1, y1, x2, y2), label, conf))

        return detections


def main(argv):
    """
    Quick manual test: run the detector on a single image and print what
    it finds. Usage: python detector.py path/to/image.jpg
    """
    if len(argv) < 2:
        print("Usage: python detector.py <path_to_image>")
        return

    image_path = argv[1]
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read image: {image_path}")
        return

    detector = Detector()
    detections = detector.detect(frame)

    print(f"Found {len(detections)} obstacle(s):")
    for d in detections:
        print(f"  {d}")


if __name__ == "__main__":
    main(sys.argv)
