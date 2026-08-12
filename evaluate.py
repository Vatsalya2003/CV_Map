"""
evaluate.py

Vatsalya Dabhi and Aditya Pandita
CS 5330 - Final Project

Reads the labeling_template_<clip>.csv files (after label_ground_truth.py
has filled in the actual_objects/actual_decision columns for the frames
we hand-labeled) and computes two Phase 3 evaluation numbers for the
report:

  1. Detection precision/recall: did the detector find the same target
     classes a human sees in the frame?
  2. Decision accuracy: did the final guidance decision match what a
     human would reasonably say?

Rows with no ground truth (actual_decision left blank) are skipped, so
this only scores the frames we actually labeled.
"""

import sys
import csv
import glob


def parse_predicted_labels(detections_str):
    """
    Pull just the class names out of the pipeline's detections string,
    e.g. "chair:0.90:10,20,30,40;oven:0.55:1,2,3,4" -> {"chair", "oven"}.
    """
    labels = set()
    if not detections_str:
        return labels
    for part in detections_str.split(";"):
        if part:
            labels.add(part.split(":")[0])
    return labels


def parse_actual_objects(actual_objects_str):
    """
    "chair;dining table" -> {"chair", "dining table"}
    """
    if not actual_objects_str:
        return set()
    return set(x for x in actual_objects_str.split(";") if x)


def evaluate():
    template_files = sorted(glob.glob("results/labeling_template_*.csv"))

    total_tp = 0  # detected AND actually present
    total_fp = 0  # detected but NOT actually present
    total_fn = 0  # actually present but NOT detected

    decision_correct = 0
    decision_total = 0

    per_clip_results = []

    for path in template_files:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))

        labeled_rows = [r for r in rows if r["actual_decision"]]
        if not labeled_rows:
            continue

        clip_tp = clip_fp = clip_fn = 0
        clip_correct = 0

        for row in labeled_rows:
            predicted_labels = parse_predicted_labels(row["predicted_detections"])
            actual_labels = parse_actual_objects(row["actual_objects"])

            clip_tp += len(predicted_labels & actual_labels)
            clip_fp += len(predicted_labels - actual_labels)
            clip_fn += len(actual_labels - predicted_labels)

            if row["predicted_decision"].strip() == row["actual_decision"].strip():
                clip_correct += 1

        total_tp += clip_tp
        total_fp += clip_fp
        total_fn += clip_fn
        decision_correct += clip_correct
        decision_total += len(labeled_rows)

        clip_precision = clip_tp / (clip_tp + clip_fp) if (clip_tp + clip_fp) else 0
        clip_recall = clip_tp / (clip_tp + clip_fn) if (clip_tp + clip_fn) else 0
        clip_accuracy = clip_correct / len(labeled_rows)

        per_clip_results.append((path, len(labeled_rows), clip_precision, clip_recall, clip_accuracy))

    print("Per-clip results:")
    for path, n, precision, recall, accuracy in per_clip_results:
        print(f"  {path}: {n} labeled frames, "
              f"precision={precision:.2f}, recall={recall:.2f}, "
              f"decision_accuracy={accuracy:.2f}")

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    overall_accuracy = decision_correct / decision_total if decision_total else 0

    print()
    print("Overall:")
    print(f"  Labeled frames: {decision_total}")
    print(f"  Detection precision: {overall_precision:.2f}")
    print(f"  Detection recall: {overall_recall:.2f}")
    print(f"  Decision accuracy: {overall_accuracy:.2f}")


def main(argv):
    evaluate()


if __name__ == "__main__":
    main(sys.argv)
