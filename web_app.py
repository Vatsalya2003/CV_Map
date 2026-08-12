"""
web_app.py

Vatsalya Dabhi and Aditya Pandita
CS 5330 - Final Project

This is the "app on your phone" stretch goal, done the fast way: instead
of writing a native iOS app (Swift + CoreML + Xcode), we run a small
Flask web server on the Mac that reuses our existing detector.py,
distance.py, and zones.py exactly as they are. The phone just opens a
web page in Safari, which:
  1. Turns on the phone's camera (via the browser, not any app install).
  2. Sends a frame to this server every fraction of a second.
  3. Gets back a guidance decision (e.g. "obstacle left, move right").
  4. Speaks it out loud using the browser's own text-to-speech.

The phone and the Mac need to be on the same WiFi network. We serve
over HTTPS with a self-signed certificate because iOS Safari refuses to
grant camera access over plain HTTP to anything other than localhost.
"""

import sys

from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2

from detector import Detector
import zones

app = Flask(__name__)
detector = Detector()


@app.route("/")
def index():
    """Serve the single page the phone loads in its browser."""
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_frame():
    """
    Receive one JPEG frame from the phone, run it through the same
    detect -> distance/zones pipeline as main.py, and return the
    guidance decision plus the detection boxes as JSON so the page can
    draw them on top of the video.
    """
    file_bytes = request.files["frame"].read()
    np_array = np.frombuffer(file_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "could not decode frame"}), 400

    frame_height, frame_width = frame.shape[:2]
    detections = detector.detect(frame)
    decision, left_status, right_status = zones.decide(detections, frame_width, frame_height)

    boxes = [
        {
            "label": d.label,
            "conf": d.conf,
            "x1": d.bbox[0], "y1": d.bbox[1],
            "x2": d.bbox[2], "y2": d.bbox[3],
        }
        for d in detections
    ]

    return jsonify({
        "decision": decision,
        "left_status": left_status,
        "right_status": right_status,
        "boxes": boxes,
        "frame_width": frame_width,
        "frame_height": frame_height,
    })


def main(argv):
    # We used to terminate HTTPS here ourselves with a self-signed
    # certificate, since iOS Safari won't grant camera access to a
    # plain http:// page. That worked in desktop Safari but iOS Safari
    # silently refused it (loading about:blank, no warning at all) even
    # after adding a proper SAN entry - iOS is stricter about
    # self-signed certs than macOS. Simplest reliable fix: run plain
    # HTTP locally and put ngrok in front of it (`ngrok http 5001`).
    # ngrok terminates HTTPS with a real, publicly-trusted certificate,
    # so the phone sees a normal secure page with no warnings at all,
    # and it works over any network, not just the same WiFi.
    app.run(host="0.0.0.0", port=5001, debug=False)


if __name__ == "__main__":
    main(sys.argv)
