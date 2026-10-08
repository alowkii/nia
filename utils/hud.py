"""The assistant's line to NIA's window (nia.py): its state, what was heard and said, and each tool call go
out; typed commands, mic presses and live setting changes come back in.

nia.py starts the assistant with NIA_HUD="<port>:<key>" and listens there. Started any other way (python main.py,
the tests) there's no window, and every call here does nothing.
"""
import os
import threading
from multiprocessing.connection import Client

import numpy as np

FPS = 60


def envelope(pcm, rate, fps=FPS):
    """Loudness of each 1/fps-second slice of the audio, 0-1 relative to its loudest slice"""
    pcm = np.asarray(pcm, dtype=np.float32)
    hop = max(1, rate // fps)
    frames = len(pcm) // hop
    if not frames:
        return []
    rms = np.sqrt(np.mean(pcm[:frames * hop].reshape(frames, hop) ** 2, axis=1))
    return np.round(rms / (rms.max() or 1), 3).tolist()


class Hud:
    def __init__(self):
        self.conn = None
        self.lock = threading.Lock()  # sends come from the voice loop, the mic thread and the agent

    def connect(self, on_command, address=None):
        """Join nia.py's window; on_command gets each message from it on a background thread"""
        address = address or os.getenv("NIA_HUD")
        if not address:
            return
        port, key = address.split(":")
        self.conn = Client(("127.0.0.1", int(port)), authkey=bytes.fromhex(key))

        def listen():
            try:
                while True:
                    on_command(self.conn.recv())
            except (EOFError, OSError):  # the window's script is gone: nobody to work for
                on_command({"type": "quit"})
        threading.Thread(target=listen, daemon=True).start()

    def send(self, **message):
        if self.conn:
            try:
                with self.lock:
                    self.conn.send(message)
            except OSError:
                self.conn = None

    def show(self, state):
        """booting | asleep | awake | thinking"""
        self.send(state=state)

    def speak(self, pcm, rate):
        """A sentence is about to play: the window follows its loudness, starting now"""
        self.send(state="speaking", envelope=envelope(pcm, rate), fps=FPS)


hud = Hud()  # one per process
