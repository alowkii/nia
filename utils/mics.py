"""Microphones, by name: names survive reboots and replugging, device numbers don't.

Windows lists each microphone once per audio system (MME, DirectSound, WASAPI, WDM-KS). NIA shows and opens the
DirectSound ones: full names (MME cuts them at 31 characters) and, like MME, they take Moonshine's 16 kHz as is.

Linux (PipeWire) has one audio system, but PortAudio only sees it as a single "pipewire" device, which records from
the default microphone. So NIA lists the sound server's own sources by name, records through that device, and then
moves her recording onto the chosen source (route) - as a mixer app does when you drag a stream to another input.
"""
import json
import os
import subprocess
import threading
import time

import sounddevice as sd

from utils.system import WINDOWS

DEFAULT = ""  # the mic the system uses by default
STAND_INS = ("Primary Sound Capture Driver", "Microsoft Sound Mapper - Input")  # aliases for the default


def _pactl(*args):
    try:
        done = subprocess.run(["pactl", "-f", "json", *args], capture_output=True, text=True, timeout=5)
        return json.loads(done.stdout or "[]")
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def _sources():
    """Linux: the sound server's microphones as (name, description) - not the monitors that record what's playing"""
    return [(s["name"], s.get("description") or s["name"]) for s in _pactl("list", "sources")
            if not s["name"].endswith(".monitor") and (s.get("properties") or {}).get("device.class") != "monitor"]


def _pipewire_device():
    """Linux: PortAudio's device for the sound server (PipeWire's, else ALSA's default), or None"""
    devices = {d["name"]: i for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0}
    return devices.get("pipewire", devices.get("default"))


def _inputs():
    if not WINDOWS:
        device = _pipewire_device()
        return [(device, description) for _, description in _sources()] if device is not None else []
    apis = [api["name"] for api in sd.query_hostapis()]
    preferred = "Windows DirectSound" if "Windows DirectSound" in apis else apis[sd.default.hostapi]
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0 and apis[d["hostapi"]] == preferred and d["name"] not in STAND_INS]


def names(refresh=False):
    """The microphones there are now; refresh=True notices ones plugged in since this process started"""
    if refresh:  # PortAudio reads the device list once, when it starts
        sd._terminate()
        sd._initialize()
    return [name for _, name in _inputs()]


def index(name):
    """The device to open for a microphone name - None (the default) for DEFAULT or one that's gone"""
    return next((i for i, n in _inputs() if n == name), None) if name else None


def route(name, wait=3.0):
    """Linux: this process's recording, moved onto the microphone named - in the background, since the stream
    appears in the sound server a moment after it's opened. Nothing to do on Windows, or for the default"""
    if WINDOWS or not name:
        return
    source = next((n for n, description in _sources() if description == name), None)
    if not source:
        return

    def move():
        end = time.time() + wait
        while time.time() < end:
            mine = {c["index"] for c in _pactl("list", "clients")
                    if (c.get("properties") or {}).get("application.process.id") == str(os.getpid())}
            streams = [s["index"] for s in _pactl("list", "source-outputs") if s.get("client") in mine]
            if streams:
                for stream in streams:
                    subprocess.run(["pactl", "move-source-output", str(stream), source], capture_output=True)
                return
            time.sleep(0.2)
    threading.Thread(target=move, daemon=True).start()
