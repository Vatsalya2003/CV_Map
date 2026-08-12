# Developer Guide — Obstacle Detection & Audio Guidance for the Visually Impaired

CS 5330 (Pattern Recognition and Computer Vision), Northeastern University, Summer 2026 — Final Project, due Aug 13.
Team: Vatsalya Dabhi (dabhi.va@northeastern.edu) and Aditya Pandita (pandita.a@northeastern.edu).

This document is written so a teammate (and their AI assistant) can pick up this project cold, with zero prior context, and immediately understand what exists, why it was built this way, what's proven to work, and what's still open. Read this before touching any code.

## 1. What this project actually is

A scoped-down assistive-navigation system. Not SLAM, not real depth estimation, not turn-by-turn navigation — explicitly out of scope by design. The pipeline is:

1. Capture a video frame (webcam, phone, or recorded clip)
2. Run YOLOv8n (pretrained on COCO) to detect a fixed subset of "obstacle-like" object classes
3. Turn each detected box into a simple "closeness" score using box height and vertical position (a proxy, not real depth)
4. Split the frame into left/right zones and decide: "path clear" / "obstacle left, move right" / "obstacle right, move left" / "stop"
5. Speak the decision out loud (debounced so it doesn't repeat every frame)

Everything is Python. There is no C++ in this project (explicitly decided against, despite the course covering C++, because it wasn't required and there wasn't time to safely add it).

## 2. Repo layout and what each file does

```
main.py                    - the real entry point. Wires detector -> distance/zones -> guidance -> debug overlay into one loop.
detector.py                - YOLOv8n wrapper. Loads yolov8n.pt, filters detections to OBSTACLE_CLASSES, applies per-class confidence thresholds.
distance.py                - bbox -> closeness score -> far/near/very_near bucket. Pure math, no model.
zones.py                   - frame -> left/right zone assignment (with a center margin) -> decision string.
guidance.py                - decision string -> spoken audio via pyttsx3, with debounce (change-or-cooldown) so it doesn't spam.
utils.py                   - debug drawing helpers (draw_boxes, draw_fps).

web_app.py + templates/index.html   - phone demo: Flask server reusing detector.py/zones.py as-is, phone opens a page in
                                       Safari that streams its camera to the server and speaks the decision back. See
                                       section 6 below, there are real gotchas here.

extract_labeling_samples.py - Phase 3 tool: samples every Nth frame from a logged run, saves the image, builds a CSV
                               template with the pipeline's own prediction + blank ground-truth columns.
build_contact_sheets.py     - Phase 3 tool: arranges sampled frames into labeled image grids for fast manual review.
label_ground_truth.py       - Contains the actual hand-judged ground truth (see GROUND_TRUTH dict) and writes it into
                               the labeling_template_<clip>.csv files.
evaluate.py                 - Reads the labeled templates, computes detection precision/recall and decision accuracy.

test_phase1_webcam.py       - one-off Phase 1 sanity script, not part of the real pipeline, safe to delete before submission.

IMPLEMENTATION_PLAN.md      - the original phased plan, with locked-in decisions (class subset, zone split, thresholds).
NOTES.md                    - chronological log of every real decision, bug, and finding, in plain language. READ THIS.
                               It has more context than this file in some places, especially around debugging specifics.

test_footage/                - 3 real iPhone clips used for Phase 3 evaluation (IMG_5077/5078/5079, see section 5).
results/                     - all Phase 3 outputs: per-frame logs, labeling templates, labeled frame images, contact
                               sheets, annotated demo videos, evaluation snapshots.
```

## 3. Environment setup

```
cd CV_Final
python3.12 -m venv venv          # must be 3.12 specifically, matches course convention
source venv/bin/activate
pip install -r requirements.txt
```

MPS (Apple GPU) acceleration is used automatically if available (`detector.py` checks `torch.backends.mps.is_available()`). `yolov8n.pt` downloads automatically on first run via `ultralytics` and is gitignored (`*.pt`).

## 4. Running things

```
python main.py                              # live webcam, shows a window, press q to quit
python main.py test_footage/IMG_5079.MOV     # run on a recorded clip instead of webcam
python main.py <video> --no-window           # headless (no display window) - use for any non-interactive/SSH run
python main.py <video> --save                # also save annotated output to results/main_output.mp4
python main.py <video> --log                 # write a per-frame CSV to results/log_<clip>.csv
```

Phase 3 evaluation pipeline (in order):
```
python extract_labeling_samples.py   # needs results/log_<clip>.csv to already exist for each clip in CLIPS
python build_contact_sheets.py       # optional, just for faster visual review
python label_ground_truth.py         # writes the GROUND_TRUTH dict's judgments into the templates
python evaluate.py                   # prints precision/recall/decision accuracy
```

## 5. Key decisions and WHY (don't re-litigate these without reading the reasoning first)

- **Detection classes** (`detector.py OBSTACLE_CLASSES`): person, chair, couch, dining table, potted plant, backpack, bench, refrigerator, oven, sink, tv, bed, toilet, microwave, laptop, vase, suitcase, book. COCO has no "door" class, so large fixed objects stand in for door-sized obstacles. The last six (toilet/microwave/laptop/vase/suitcase/book) were added after initial phone testing, on request, to broaden indoor coverage — this was a deliberate, discussed decision, not something to just keep expanding casually.
- **Zone split**: left/right halves, with a margin (10% of frame width) around the midline so a box near the center counts toward BOTH zones — see `get_zones_for_detection` in `zones.py`. This is what makes "stop" trigger correctly when something is dead ahead.
- **Closeness buckets**: far / near / very_near from `distance.py`. Only `very_near` currently drives the decision (near is informational only, shown in overlay/logs but doesn't force an action). **This was tested and deliberately reverted** — we tried making `near` also block a zone, and it fixed some real hallway-chair misses but broke kitchen-appliance cases badly (large fixed objects like ovens read as "near" from far away just because they're physically tall, since the closeness proxy is only `box_height_ratio * position_weight` with no real distance information). Net accuracy across 26 hand-labeled evaluation frames was WORSE with near-blocks-too (13/26) than with very_near-only (15/26), so very_near-only is what's in the code now. Full comparison is written directly in `zones.py`'s `decide()` docstring/comments AND in `NOTES.md`.
- **Confidence threshold**: single global 0.4 default in `detector.py`, structured as a per-class dict so it CAN be tuned per-class later, but hasn't needed to be yet.

## 6. The phone web app — real gotchas, read before touching this

`web_app.py` + `templates/index.html` let you run the demo from a phone browser instead of the desktop webcam, by reusing `detector.py`/`zones.py` unchanged. Three real problems were hit and fixed here, in order:

1. **iOS Safari rejects self-signed certs with no SAN.** Flask's `ssl_context="adhoc"` makes a cert with no Subject Alternative Name field. Desktop Safari tolerates this; iOS Safari silently loads `about:blank` with zero warning. Generating a proper SAN cert with openssl STILL didn't fully resolve it reliably.
2. **Fix that actually worked**: stopped doing HTTPS termination locally at all. Flask now runs plain HTTP (`app.run(host="0.0.0.0", port=5001)`, no ssl_context), and `ngrok http 5001` fronts it with a real, publicly-trusted certificate. This also removes the "phone must be on the same WiFi" requirement, since ngrok tunnels over the internet.
3. **No audio even after the page loaded correctly.** iOS Safari requires `speechSynthesis.speak()` to be called synchronously inside a real user-gesture handler (a click) at least once before ANY later `speak()` call (e.g. from an async fetch response) will produce sound — otherwise it fails completely silently. Fixed in `index.html`'s Start button click handler by speaking one empty utterance first, synchronously, before anything else happens.

If you're demoing this again: `ngrok http 5001` gives you a fresh random URL every time you restart it (unless you have a reserved ngrok domain), so re-check the URL each time, don't assume the old one still works.

## 7. Phase 3 evaluation — what the numbers mean and their real limits

Recorded 3 real iPhone clips (`test_footage/IMG_5077/5078/5079.MOV`, all portrait 1080x1920): an open-floor walk, a chair-close-on-one-side walk, and a table-blocking-the-hallway walk. Ran the full pipeline with `--log` on each, then hand-labeled a representative 26-frame subset (not all ~2300 frames — see `NOTES.md` for why that scope call was made given the deadline) with real ground truth for both "what objects are actually there" and "what decision a human would give."

**Final numbers**: detection precision 0.82, recall 0.89, decision accuracy 0.58 (uneven per clip: 0.88 / 0.40 / 0.50).

Two real findings worth knowing before you touch thresholds again:
- A chair cropped at the top frame edge was completely missed by YOLO (a genuine false negative, not a threshold issue) — worth citing as a limitation.
- The labeling process itself had a real bug the first pass through: judging left/right from small resized thumbnail contact sheets produced backwards calls in several cases (e.g. an oven that's objectively right-of-center by its bounding box got called "obstacle left"). Caught by cross-checking bbox center-x against the frame midline mathematically, not just eyeballing images. If you add more labeled frames later, verify left/right against bbox math, not just a quick visual read of a small thumbnail.

## 8. What's NOT done / explicit non-goals

- No real depth estimation, no SLAM — intentionally out of scope, don't add it even if it seems like it'd fix the "can't detect a plain wall" limitation (it would, but it's explicitly excluded from this project's scope).
- No native iOS app. The phone demo is a web app (Flask + ngrok), not Swift/CoreML. A real iOS app was scoped as a stretch goal and deliberately NOT started, given the deadline — if picked up later, start from scratch reading Apple's Vision framework + CoreML docs, this codebase has nothing iOS-native in it.
- Report (IEEE format, ≤8 pages, needs 3+ peer-reviewed references), presentation video (≤15 min), and README.txt are NOT yet written as of this document's creation. That's the actual remaining graded work.

## 9. Git

Remote: `https://github.com/Vatsalya2003/CV_Map.git`, branch `main`. Repo includes the raw test footage and annotated output videos committed directly (not Git LFS) — GitHub will warn about one file being just over its 50MB soft limit, that's expected and not an error.

## 10. If you (or your Claude Code) get stuck

Read `NOTES.md` first — it's a chronological, plain-language log of literally every real decision, bug, and finding across the whole project, written specifically so this kind of handoff works. It has more granular reasoning than this file does in several places (especially the debugging narratives). `IMPLEMENTATION_PLAN.md` has the original phased plan if you want the "why did we build it in this order" context.
