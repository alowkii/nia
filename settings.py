"""NIA settings: defaults live here, your changes in settings.json (written by ui.py)"""
import json
from pathlib import Path

PATH = Path(__file__).resolve().parent / "settings.json"

DEFAULTS = {
    # LLM server (llama-server + Bonsai, GPU) - restart the server to apply
    "bonsai_dir": r"D:\bonsai",
    "model_file": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "port": 8081,  # 8080 is taken by httpd on this machine
    "context": 8192,  # tokens; more needs VRAM the display is already using
    "thinking": False,  # model card suggests temperature 1.0, top_p 0.95 with thinking on
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "presence_penalty": 1.5,
    # Assistant (CPU) - restart the assistant to apply
    # ponytail: Kokoro takes ~1 s to start speaking on CPU; "piper_en_US-lessac-medium" ~0.2 s but more robotic
    "voice": "kokoro_af_heart",
    "stt_model": "medium",  # Moonshine streaming: tiny / small / medium
    "wake_sensitivity": 0.5,  # Porcupine, 0-1: higher wakes more easily but false-triggers more
    "greeting": "Yes, sir?",
    "session_timeout": 60,  # seconds of no exchange before going back to wake-word listening
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
