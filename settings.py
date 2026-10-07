"""NIA settings: defaults live here, your changes in settings.json (written by ui.py)"""
import json
from pathlib import Path

PATH = Path(__file__).resolve().parent / "settings.json"

DEFAULTS = {
    # LLM server (llama-server + Bonsai, GPU) - restart the server to apply
    "bonsai_dir": r"D:\bonsai",
    "model_file": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "port": 8081,  # 8080 is taken by httpd on this machine
    "context": 16384,  # tokens; ~7.5 of 8 GB VRAM on an RTX 4060 with the display. The agent harness alone is ~3.6K
    # 8-bit context cache: measured 478 MB less VRAM at 16K with the same speed (34.5 vs 34.6 tok/s)
    "cache_8bit": True,
    "thinking": False,  # model card suggests temperature 1.0, top_p 0.95 with thinking on
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "presence_penalty": 1.5,
    # Assistant (CPU) - restart the assistant to apply
    # ponytail: Kokoro takes ~1 s to start speaking on CPU; "piper_en_US-lessac-medium" ~0.2 s but more robotic
    "voice": "kokoro_af_heart",
    "voice_volume": 1.0,  # NIA's own voice, 0.1-1 (1 = full; more would clip); also set by voice ("speak quieter")
    "stt_model": "medium",  # Moonshine streaming: tiny / small / medium (tiny rarely hears "Nia")
    "wake_phrase": "hey nia",
    # How closely the start of what you say must match the wake phrase, 0-1. Lower wakes more easily:
    # 0.8 takes "Hey Nia" and rejects "Hey, nice..." (0.77), but "Hey Mia" (0.83) still wakes her
    "wake_threshold": 0.8,
    "greeting": "Yes, sir?",
    "duck_level": 0.3,  # other apps' volume while NIA speaks, as a fraction of their own; 1 = no ducking
    "session_timeout": 60,
    # Decision model for PC commands that aren't on the read-only allowlist (see agent/approval.py): run
    # without asking if every hazard scores below the threshold. Empty model name = always ask
    "approval_model": "openthai-cpu",
    "approval_threshold": 0.5,
    # What "play some music" picks from, at random, when no song is named (comma-separated playlist searches)
    "music_moods": "lofi beats, chill hits, feel good pop, indie favourites, throwback hits",  # seconds of no exchange before going back to wake-word listening
}


def load():
    """Defaults, overridden by whatever settings.json holds"""
    try:
        saved = json.loads(PATH.read_text())
    except FileNotFoundError:
        saved = {}
    return {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}


def save(settings):
    PATH.write_text(json.dumps(settings, indent=2))
