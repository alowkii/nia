"""Each app's volume on Linux (PipeWire or PulseAudio, through pactl), in the shape of pycaw's Windows mixer: sessions
with a ProcessId, an Identifier that stays the same across restarts, and a SimpleAudioVolume to get and set - so
the code that turns other apps down while NIA speaks is the same on both.

A stream's own properties may not say which process it is (NIA's voice, through PipeWire's ALSA plugin, doesn't), but
its client always does, so the process is taken from there. Notification sounds (media.role "event") are Linux's
counterpart to Windows' system sounds, which pycaw lists as process 0."""
import json
import subprocess
from types import SimpleNamespace

FULL = 65536  # PulseAudio's 100%


def _pactl(*args):
    done = subprocess.run(["pactl", "-f", "json", *args], capture_output=True, text=True, timeout=5)
    return json.loads(done.stdout or "[]")


class StreamVolume:
    """One playing stream's volume, 0-1 of its own full volume, as pycaw's SimpleAudioVolume"""

    def __init__(self, index, value):
        self.index, self.value = index, value

    def GetMasterVolume(self):
        return self.value

    def SetMasterVolume(self, value, context=None):
        subprocess.run(["pactl", "set-sink-input-volume", str(self.index), str(round(value * FULL))],
                       capture_output=True, timeout=5)
        self.value = value


class AudioUtilities:
    @staticmethod
    def GetAllSessions():
        """Every stream playing now. No pactl, or no sound server: none - she just doesn't turn anything down"""
        try:
            clients = {c["index"]: c.get("properties", {}) for c in _pactl("list", "clients")}
            streams = _pactl("list", "sink-inputs")
        except (OSError, subprocess.SubprocessError, ValueError):
            return []
        sessions = []
        for stream in streams:
            props = {**clients.get(stream.get("client"), {}), **stream.get("properties", {})}
            channels = [c["value"] for c in (stream.get("volume") or {}).values()] or [FULL]
            event = props.get("media.role") == "event"
            pid = int(clients.get(stream.get("client"), {}).get("application.process.id") or -1)
            sessions.append(SimpleNamespace(
                ProcessId=0 if event else pid,
                Identifier=props.get("application.process.binary") or props.get("application.name") or "",
                SimpleAudioVolume=StreamVolume(stream["index"], max(channels) / FULL)))
        return sessions
