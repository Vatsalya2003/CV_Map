# Session Decision Log — 2026-08-11

Compressed record of every real decision made in this session, in order. For full narrative detail see `NOTES.md`; for the developer-facing explanation see `ONBOARDING.md`.

## Planning
- Locked detection class subset: **broad set** — person, chair, couch, dining table, potted plant, backpack, bench, refrigerator, oven, sink, tv, bed (later expanded, see below).
- Zone split: left/right halves (not thirds), with a **10%-of-frame-width center margin** so a box near the midline counts toward both zones — this is what makes "stop" trigger correctly.
- Closeness: 3 buckets (far/near/very_near) from `box_height_ratio * position_weight`; only `very_near` drives the decision, `near` is informational only.
- Added explicit scope guards after a second review pass: FPS/frame-skip fallback, per-class confidence thresholds (structure only, not yet tuned), two-pass labeling requirement (objects-present vs. correct-decision), IEEE report page/reference requirements.
- **Decided against adding C++** to the project despite the course covering it — no rubric requirement, too risky with ~2 days left. Stuck to Python end-to-end.

## Phase 0 – Environment
- Python 3.12 venv (not the system 3.14), `ultralytics` + `pyttsx3` + `opencv-python` installed, MPS confirmed available, `yolov8n.pt` downloaded, `requirements.txt` frozen, `.gitignore` added.

## Phase 1 – Detection
- `detector.py` + `utils.py` built and sanity-checked on real webcam footage (~10.9 FPS on MPS, correct person detection with real bbox).

## Phase 2 – Distance / Zones / Guidance
- `distance.py`, `zones.py`, `guidance.py` built and unit-sanity-checked.
- **Real bug found and fixed**: `pyttsx3`'s `runAndWait()` isn't thread-safe; spawning a new thread per spoken sentence crashed with "run loop already started" on fast decision changes. Fixed with one persistent worker thread + a `queue.Queue`.

## main.py — full pipeline
- Wired everything into `main.py` with webcam/file input, `--save`, `--no-window`, `--log` flags.
- Verified end-to-end on a recorded clip (~18.75 FPS, correct "stop" on a person straddling the frame midline).

## Phase 3 – Real footage evaluation
- Recorded 3 real iPhone clips (open floor, chair-on-one-side, table-blocking-path). Confirmed OpenCV auto-corrects iPhone portrait orientation — no manual rotation fix needed.
- Built the full evaluation toolchain: `extract_labeling_samples.py` (sample every 15th frame + build labeling CSV), `build_contact_sheets.py` (grid multiple frames per image for fast review), `label_ground_truth.py` (hand-judged ground truth for a 26-frame representative subset, not all ~2300 frames — a deliberate scope call given the deadline), `evaluate.py` (precision/recall/decision-accuracy).
- **Labeling-quality bug caught and fixed**: first labeling pass (done from resized thumbnails) had real left/right errors — e.g. an oven objectively right-of-center by its bbox got called "obstacle left." Fixed by cross-checking bbox center-x against the frame midline mathematically and re-viewing full-resolution frames, not just thumbnails.
- **Threshold experiment, tried and reverted**: promoted `near` to also block a zone (not just `very_near`). Fixed real hallway-chair misses but broke kitchen-appliance cases (box-height is a poor distance proxy for physically large fixed objects like ovens). Measured net accuracy both ways across all 26 labeled frames: near-blocks-too = 13/26 (50%), very_near-only = 15/26 (58%). **Kept very_near-only** since it was actually better in aggregate; documented the comparison directly in `zones.py`.
- **Final evaluation numbers**: detection precision 0.82, recall 0.89, decision accuracy 0.58 (per-clip: 0.88 / 0.40 / 0.50).
- Known real failure case: a chair cropped at the frame edge was completely missed by YOLO (genuine false negative, not a threshold issue).

## Phone demo (web app, not native iOS)
- Explicitly chose a **Flask + browser web app** over a native iOS app, given ~2 days left before the deadline (native iOS was scoped as 1-2+ days of Swift/CoreML work, too risky).
- Built `web_app.py` (reuses `detector.py`/`zones.py` unchanged, one `/process` JSON endpoint) and `templates/index.html` (phone camera capture, canvas overlay, browser `SpeechSynthesis` with the same debounce logic as `guidance.py`).
- **Three real bugs hit and fixed, in order**:
  1. Flask's `ssl_context="adhoc"` self-signed cert has no SAN field → iOS Safari silently loads `about:blank` (desktop Safari tolerates it, iOS doesn't).
  2. Generating a proper SAN cert with openssl still didn't fully resolve it → switched to plain HTTP locally + **ngrok tunnel** for a real trusted cert (also removes the "same WiFi" requirement).
  3. Page loaded but no audio → iOS Safari requires `speechSynthesis.speak()` to be called synchronously inside a real user-gesture handler at least once before any later async-triggered `speak()` calls will produce sound. Fixed by speaking an empty utterance directly inside the Start button's click handler.
- Added a visible on-screen "🔊 speaking: ..." indicator to make future audio issues easier to diagnose (code-not-calling-speak vs. device-muted).

## Class list expansion
- After phone testing, deliberately expanded `OBSTACLE_CLASSES` (not silently): added toilet, microwave, laptop, vase, suitcase, book.
- Clarified and documented as a **known, accepted limitation** (not a bug to fix): a plain featureless wall can't be detected by an object detector — would require depth/segmentation, which is explicitly out of scope.

## Git / handoff
- Initialized git, committed everything (211 files, including test footage and annotated demo videos — chosen deliberately over excluding large media, since this is a one-time course submission repo).
- Pushed to `https://github.com/Vatsalya2003/CV_Map.git` on `main`.
- Wrote `ONBOARDING.md` (full developer handoff guide) and generated a Claude Code share link for teammate Aditya Pandita.

## Still open (next session)
- IEEE report (≤8 pages, needs 3+ peer-reviewed references) — not started.
- Presentation video (≤15 min) — not started.
- `README.txt` — not started.
- Native iOS app — explicitly not started, stretch goal only if time remains after the above.
