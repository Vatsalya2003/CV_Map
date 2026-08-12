"""
build_contact_sheets.py

Vatsalya Dabhi and Aditya Pandita
CS 5330 - Final Project

One-off Phase 3 helper (not part of the final pipeline). Takes the
sampled labeling frames from extract_labeling_samples.py and arranges
them into grid "contact sheets" (like a photo contact sheet) with the
frame number printed on each tile. This makes it much faster to review
many sampled frames at once instead of opening one image at a time.
"""

import sys
import os
import csv
import math

import cv2
import numpy as np

TILE_WIDTH = 260
TILES_PER_ROW = 4
TILES_PER_SHEET = 12

CLIPS = ["IMG_5077", "IMG_5078", "IMG_5079"]


def build_sheets_for_clip(clip_name):
    template_path = f"results/labeling_template_{clip_name}.csv"
    frames_dir = f"results/labeling_frames/{clip_name}"
    sheets_dir = f"results/contact_sheets/{clip_name}"

    if not os.path.exists(template_path):
        print(f"Skipping {clip_name}, no template found.")
        return

    os.makedirs(sheets_dir, exist_ok=True)

    with open(template_path, newline="") as f:
        rows = list(csv.DictReader(f))

    num_sheets = math.ceil(len(rows) / TILES_PER_SHEET)

    for sheet_index in range(num_sheets):
        chunk = rows[sheet_index * TILES_PER_SHEET: (sheet_index + 1) * TILES_PER_SHEET]
        tiles = []

        for row in chunk:
            img_path = os.path.join(frames_dir, row["image_file"])
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Resize to a fixed small width, keeping aspect ratio.
            h, w = img.shape[:2]
            scale = TILE_WIDTH / w
            tile = cv2.resize(img, (TILE_WIDTH, int(h * scale)))

            # Stamp the frame number on the tile so we can match it back
            # to the CSV row when filling in the ground truth columns.
            cv2.putText(tile, f"frame {row['frame']}", (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            tiles.append(tile)

        if not tiles:
            continue

        tile_height = tiles[0].shape[0]
        num_rows = math.ceil(len(tiles) / TILES_PER_ROW)
        sheet = np.zeros((tile_height * num_rows, TILE_WIDTH * TILES_PER_ROW, 3), dtype=np.uint8)

        for i, tile in enumerate(tiles):
            row_i = i // TILES_PER_ROW
            col_i = i % TILES_PER_ROW
            y = row_i * tile_height
            x = col_i * TILE_WIDTH
            sheet[y:y + tile.shape[0], x:x + tile.shape[1]] = tile

        out_path = os.path.join(sheets_dir, f"sheet_{sheet_index}.jpg")
        cv2.imwrite(out_path, sheet)

    print(f"{clip_name}: wrote {num_sheets} contact sheet(s) to {sheets_dir}/")


def main(argv):
    for clip_name in CLIPS:
        build_sheets_for_clip(clip_name)


if __name__ == "__main__":
    main(sys.argv)
