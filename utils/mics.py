"""Microphones, by name: names survive reboots and replugging, device numbers don't.

Windows lists each microphone once per audio system (MME, DirectSound, WASAPI, WDM-KS). NIA shows and opens the
DirectSound ones: full names (MME cuts them at 31 characters) and, like MME, they take Moonshine's 16 kHz as is.
"""
import sounddevice as sd

DEFAULT = ""  # the mic Windows uses by default
STAND_INS = ("Primary Sound Capture Driver", "Microsoft Sound Mapper - Input")  # aliases for the default


def _inputs():
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
