# Implementation Plan — Obstacle Detection & Audio Guidance for the Visually Impaired

CS 5330 Final Project — Vatsalya Dabhi & Aditya Pandita — Due Aug 13

## 1. Decisions locked in

**Detection class subset (COCO, "Broad set"):**
`person, chair, couch, dining table, potted plant, backpack, bench, refrigerator, oven, sink, tv, bed`

Rationale: covers common indoor/hallway obstacles. COCO has no "door" class, so `bench`/`refrigerator`/`oven`/`sink` stand in as proxies for large fixed obstacles in a home/office setting. This list is filtered from YOLOv8n's full 80-class COCO output in `detector.py` — nothing else gets passed downstream.

**Zone split:** start with left/right halves (simplest, matches proposal). We will test thirds (left/center/right) in Phase 3 and switch only if halves prove too coarse on real footage.

**Closeness buckets:** 3 buckets — `far`, `near`, `very near` — driven by `(box_height / frame_height) * position_weight`, thresholds tuned empirically in Phase 2/3. Only `very_near` drives the zone decision table (stop/move) — `near` does not force an action yet, but is shown in the debug overlay and logged, so it's a real, visible signal rather than dead code. If Phase 3 testing shows the binary near/not-near split is too abrupt, `near` can be promoted to trigger a softer "caution" message.

**Performance target:** measure actual FPS during Phase 1's sanity check. `main.py` will support a `PROCESS_EVERY_N` frame-skip setting (default 1 = every frame) from the start, so if YOLOv8n inference doesn't keep up in real time on this hardware, we can drop to every 2nd/3rd frame without a refactor. Report FPS in the results section.

**Confidence threshold:** `detector.py` takes a per-class threshold dict (all classes default to 0.4). Global 0.4 is a starting point, not a final answer — some classes (potted plant, backpack) are noisier than others (person, chair) and likely need per-class tuning after Phase 3 testing.

## 2. Repo structure

```
CV_Final/
  main.py            # live camera loop, wires everything together
  detector.py         # YOLOv8n load + inference + class filtering
  distance.py         # bbox -> closeness bucket
  zones.py            # frame -> zone decision string
  guidance.py          # decision string -> TTS w/ debounce
  utils.py             # debug drawing helpers
  test_footage/        # recorded hallway/room videos
  results/             # plots, metrics, screenshots for report
  report/              # IEEE LaTeX source
  NOTES.md             # running log of LLM usage (for acknowledgement section)
  README.txt
  requirements.txt
```

## 3. Environment setup (Phase 0)

- Python 3.12 (Homebrew), venv in project dir
- `pip install ultralytics pyttsx3 opencv-python`
- Verify MPS available via `torch.backends.mps.is_available()`
- Download/cache YOLOv8n weights (`yolov8n.pt`) on first run via ultralytics
- Freeze installed versions into `requirements.txt` (`pip freeze > requirements.txt`) right after installs, so the environment is reproducible from day one

## 4. Phase 1 — Detection

`detector.py`:
- `Detector` class wraps `ultralytics.YOLO("yolov8n.pt")`
- `detect(frame) -> list[Detection]` where `Detection = {bbox, label, conf}`
- Filters to the 12-class subset above, applies a confidence threshold (start at 0.4)
- Runs on MPS device if available, else CPU

Sanity check: run on a test image and a short webcam clip, draw boxes via `utils.draw_boxes`, visually confirm labels/boxes look right before moving on.

## 5. Phase 2 — Distance proxy, zones, TTS

`distance.py`:
- `closeness_score(bbox, frame_height) -> float`
- `bucket(score) -> "far" | "near" | "very_near"`
- `position_weight`: obstacles in the lower half of the frame weighted higher (closer to camera in typical eye-level shot)

`zones.py`:
- Split frame width in half (left/right)
- Assign each detection's bbox center-x to a zone. Boxes whose center-x falls within a margin (starting at 10% of frame width) of the midline count toward **both** zones — this is the concrete rule for "center-straddling," so a box near the middle can't be mis-assigned entirely to one side
- Per-zone: take the worst-case (closest) bucket among its detections
- Decision table:
  - both zones clear -> "path clear"
  - very_near only left -> "obstacle left, move right"
  - very_near only right -> "obstacle right, move left"
  - very_near in both (including a margin box counted in both) -> "stop"

`guidance.py`:
- `Guidance` class: takes decision string, speaks via `pyttsx3` only if changed from last spoken decision OR cooldown (1.5s) elapsed
- Runs TTS in a background thread so it doesn't block the frame loop

`main.py`:
- Capture loop: read frame -> `detector.detect` -> `distance` per box -> `zones.decide` -> `guidance.speak_if_needed` -> draw debug overlay -> show window -> `q` to quit

## 6. Phase 3 — Real footage testing

- Record 3-5 short hallway/room walkthrough clips (varying obstacle density/lighting)
- Run pipeline on recorded footage, log every frame's decision + detections to `results/log_<clip>.csv`
- Two separate label passes on a subset of frames (needed independently — one feeds detection metrics, the other feeds decision metrics):
  1. **Objects actually present per frame** (ground truth for detection precision/recall)
  2. **Correct guidance decision per frame**, i.e. what a human would reasonably say (ground truth for decision-accuracy %)
- Compute: detection precision/recall against pass (1), % frame-decisions matching pass (2), qualitative failure notes (jitter, false positives, missed obstacles)
- Tune thresholds (confidence — possibly per-class, closeness buckets, cooldown) based on findings

## 7. Phase 4 — Report, video, optional iOS

- IEEE 2-column PDF (**hard limit: 8 pages**) — pipeline, design choices, evaluation results, limitations
- **Requires 3+ peer-reviewed references** (real CV venues/journals, not just arXiv preprints) — start collecting these during Phase 1-2, don't leave to the end
- <=15 min presentation video
- README.txt with run instructions
- iOS Vision/CoreML wrapper — only if Phases 1-3 are fully done with time remaining

## 8. Working agreement

- Build one file/phase at a time, pause for your confirmation before moving on
- Full working files each time, not snippets
- Vision/logic code gets inline comments explaining reasoning + chat explanation, so you can defend it to Prof. Maxwell
- `NOTES.md` tracks what was asked of the LLM at each step in plain language, for the acknowledgement section
