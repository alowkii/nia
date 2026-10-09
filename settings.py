"""NIA settings: defaults live here, your changes in settings.json (written by the settings panel in nia.py)"""
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # .env, for everything that imports settings

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
    "voice": "kokoro_bm_george",  # British male, for the JARVIS manner; "kokoro_af_heart" was the American default
    "voice_speed": 1.2,  # 1 = the voice's natural pace; George at 1.2 says the same in ~17% less time
    "ready_chime": True,  # a soft rising chime each time NIA is ready - after starting, and after a restart
    "voice_volume": 1.0,  # NIA's own voice, 0.1-1 (1 = full; more would clip); also set by voice ("speak quieter")
    "mic_device": "",  # the microphone NIA listens through, by name; "" = Windows' default (switchable live)
    "stt_model": "medium",  # Moonshine streaming: tiny / small / medium (tiny rarely hears "Nia")
    "wake_phrase": "hey nia",
    # How closely the start of what you say must match the wake phrase, 0-1. Lower wakes more easily:
    # 0.8 takes "Hey Nia" and rejects "Hey, nice..." (0.77), but "Hey Mia" (0.83) still wakes her
    "wake_threshold": 0.8,
    # Said on waking: one picked at random ("|" between them), mixed with a few for the time of day
    "greeting": "At your service, sir. | You rang, sir? | Standing by, sir. | Listening, sir. | "
                "How may I be of use, sir? | Ready when you are, sir. | I'm all ears, sir. Figuratively speaking.",
    "max_spoken_words": 40,  # a longer answer is cut at a sentence and she offers "Shall I go on, sir?"
    "duck_level": 0.3,  # other apps' volume while NIA speaks, as a fraction of their own; 1 = no ducking
    "session_timeout": 60,  # seconds of no exchange before going back to wake-word listening
    # Long-term memory (agent/memory.py): Ollama embedding model ("" = no memory), how many memories each turn
    # may recall, and how alike they must be (0-1; EmbeddingGemma: relevant >= 0.29, off-topic <= 0.16)
    "embedding_model": "embeddinggemma",
    "memory_results": 4,
    "memory_min_similarity": 0.22,
    # Only this many recent exchanges are sent to the LLM; older ones come back through memory if relevant
    "history_turns": 4,
    # Decision model for PC commands that aren't on the read-only allowlist (see agent/approval.py): run
    # without asking if every hazard scores below the threshold. Empty model name = always ask
    "approval_model": "openthai-cpu",
    "approval_threshold": 0.5,
    # What "play some music" picks from, at random, when no song is named (comma-separated playlist searches)
    "music_moods": "lofi beats, chill hits, feel good pop, indie favourites, throwback hits",
    # Ask Claude (agent/claude.py): the model Claude Code answers with; "" = its own default
    "claude_model": "sonnet",
    # How hard Claude works on each question: low answered in ~17-24 s, medium/high check more sources but are slower
    "claude_effort": "low",
    # The window (nia.py) - these apply at once
    "wake_word": True,  # off: only the mic button (or typing) wakes her
    "push_to_talk": False,  # on: she listens while the mic button is held
    "spoken_replies": True,  # off: replies are shown in the window, not spoken
    "hud_theme": "cyan",  # cyan | gold | ice | mint | violet | red
}

THEMES = ("cyan", "gold", "ice", "mint", "violet", "red")
# Allowed ranges for numbers (inclusive)
RANGES = {"wake_threshold": (0, 1), "duck_level": (0, 1), "approval_threshold": (0, 1),
          "memory_min_similarity": (0, 1), "memory_results": (0, 20), "history_turns": (1, 50),
          "max_spoken_words": (10, 500), "voice_speed": (0.5, 2), "voice_volume": (0.1, 1),
          "session_timeout": (5, 3600), "port": (1, 65535), "context": (2048, 262144)}


def parse(raw):
    """Values from a form (strings, or JSON) as each default's type, checked: (values, None) or (None, error).
    Keys not given keep their current value."""
    values = load()
    for key, value in raw.items():
        if key not in DEFAULTS:
            return None, f"Unknown setting {key!r}"
        kind = type(DEFAULTS[key])
        try:
            if kind is bool:
                values[key] = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "on", "yes")
            else:
                values[key] = kind(str(value).strip()) if kind is not str else str(value)
        except ValueError:
            return None, f"{key} must be a {'whole ' if kind is int else ''}number, got {value!r}"
    for key, (low, high) in RANGES.items():
        if not low <= values[key] <= high:
            return None, f"{key} must be between {low} and {high}"
    if not values["wake_phrase"].strip():
        return None, "Wake phrase can't be empty"
    if not any(m.strip() for m in values["music_moods"].split(",")):
        return None, "Give at least one random music pick, e.g. lofi beats"
    if values["claude_effort"] not in ("low", "medium", "high", "xhigh", "max"):
        return None, "Claude effort must be low, medium, high, xhigh or max"
    if values["hud_theme"] not in THEMES:
        return None, f"Theme must be one of {', '.join(THEMES)}"
    return values, None


def load():
    """Defaults, overridden by whatever settings.json holds"""
    try:
        saved = json.loads(PATH.read_text())
    except FileNotFoundError:
        saved = {}
    return {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}


def save(settings):
    PATH.write_text(json.dumps(settings, indent=2))
