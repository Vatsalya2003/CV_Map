"""
guidance.py

Vatsalya Dabhi and Aditya Pandita
CS 5330 - Final Project

This file turns a decision string (like "obstacle left, move right") into
spoken audio using pyttsx3. We do NOT want to speak every single frame,
that would spam the user with the same sentence dozens of times a
second. So we only speak when the decision changes, or when a cooldown
timer has passed since we last spoke (in case the same warning needs to
be repeated after a while).
"""

import sys
import time
import threading
import queue

import pyttsx3

# Minimum seconds between spoken messages, even if the decision hasn't
# changed. This lets an ongoing "stop" warning repeat occasionally
# instead of only being spoken once and then going silent.
COOLDOWN_SECONDS = 1.5


class Guidance:
    """
    Wraps a pyttsx3 TTS engine and only speaks when needed (decision
    changed, or cooldown has passed).

    pyttsx3's engine.runAndWait() is not safe to call from more than one
    thread at a time (it errors out with "run loop already started" if a
    second call comes in before the first finishes). So instead of
    spawning a new thread per sentence, we run ONE background worker
    thread that pulls sentences off a queue and speaks them one at a
    time, in order.
    """

    def __init__(self, cooldown_seconds=COOLDOWN_SECONDS):
        self.engine = pyttsx3.init()
        self.cooldown_seconds = cooldown_seconds
        self.last_decision = None
        self.last_spoken_time = 0.0

        self._queue = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def speak_if_needed(self, decision):
        """
        Decide whether to speak this decision out loud, based on whether
        it changed since last time, or whether enough time has passed.
        """
        now = time.time()
        decision_changed = decision != self.last_decision
        cooldown_passed = (now - self.last_spoken_time) >= self.cooldown_seconds

        if decision_changed or cooldown_passed:
            self.last_decision = decision
            self.last_spoken_time = now
            self._queue.put(decision)

    def _worker_loop(self):
        """
        Runs forever on the background thread, speaking one sentence at
        a time so the video loop never has to wait for speech to finish.
        """
        while True:
            text = self._queue.get()
            self.engine.say(text)
            self.engine.runAndWait()


def main(argv):
    """
    Quick manual test: speak a couple of decisions, showing that a repeat
    decision within the cooldown window does NOT get spoken again, but a
    changed decision does.
    """
    guidance = Guidance(cooldown_seconds=2.0)

    print("Speaking: path clear")
    guidance.speak_if_needed("path clear")

    time.sleep(0.5)
    print("Same decision within cooldown, should NOT speak again")
    guidance.speak_if_needed("path clear")

    time.sleep(0.5)
    print("Speaking: stop")
    guidance.speak_if_needed("stop")

    # Give the background thread time to finish talking before exiting.
    time.sleep(3)


if __name__ == "__main__":
    main(sys.argv)
