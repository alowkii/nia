"""Checks the voice helpers: the fuzzy wake-phrase matcher on real Moonshine transcripts,
the cleanup of replies before they're spoken, and NIA's own voice volume. Run: python test_voice.py"""
import tempfile
from pathlib import Path

import time
from types import SimpleNamespace

import settings
from agent.chat import CANCEL
from voice_assistant.voice_assistant import WakeWordDetector, speakable, stop_request, wake_command


def wake(text, threshold=0.8):
    return wake_command(text, "hey nia", threshold)


# Woken, with whatever follows the phrase as the command
assert wake("Hey Nia.") == ""
assert wake(" Hey Nia.") == ""
assert wake("Hey, Nia!") == ""
assert wake("Hey Nia play some loafy beats.") == "play some loafy beats."
assert wake("Hey Nia, what time is it?") == "what time is it?"
assert wake("Yeah. Hey Nia what time is it?") == "what time is it?"  # stray leading word
# Mid-line, as when music or talk runs into the phrase
assert wake("I just can't stop loving you Hey Nia, pause the music.") == "pause the music."

# Not woken
assert wake("Hey nice to meet you.") is None  # 0.77, just under the default
assert wake("Here is the news.") is None
assert wake("Hey, man.") is None
assert wake("") is None
assert wake("Tell me about California.") is None  # 'nia' inside a word

# The threshold is the knob: "Hey Mia" scores 0.83
assert wake("Hey Mia, how are you?") == "how are you?"
assert wake("Hey Mia, how are you?", threshold=0.9) is None

# Spoken text: no symbol may reach the voice, which reads non-ASCII as the letter "L"
assert speakable("I hit a hiccup, sir — the token expired.") == "I hit a hiccup, sir, the token expired."
assert speakable("Done, sir — volume’s at 50% now…") == "Done, sir, volume's at 50 percent now..."
assert speakable("Playing “Halo” by Beyoncé 🎵") == 'Playing "Halo" by Beyonce'
assert speakable("Some R&B, then AC/DC → enjoy!") == "Some R and B, then AC DC enjoy!"
assert speakable("Plain text stays as it is.") == "Plain text stays as it is."
assert speakable("Wait—what?") == "Wait, what?"
assert all(ord(c) < 128 for c in speakable("Café ☕ — naïve “quotes” ‘and’ © 2026 ✓"))
# URLs: only the site's name is said, never the address
assert speakable("I'll run the command: start https://www.youtube.com. Should I go ahead?") == \
    "I'll run the command: start the youtube link. Should I go ahead?"
assert speakable('start "https://www.youtube.com/results?search_query=shadman"') == 'start "the youtube link"'
assert speakable("See www.google.co.in/maps or https://en.wikipedia.org/wiki/Nia") == \
    "See the google link or the wikipedia link"
assert speakable("Open github.com") == "Open github.com"  # a bare name with no www or https stays

# Interrupting: what counts as "stop" while NIA is busy
def stop(text):
    return stop_request(text, "hey nia", 0.8)

assert stop("Stop.") == (True, "")
assert stop("Nia, stop!") == (True, "")
assert stop("Hey Nia.") == (True, "")
assert stop("Hey Nia, stop") == (True, "")
assert stop("Hey Nia, pause the music.") == (True, "pause the music.")
assert stop("Playing lofi beats for you, sir.") == (False, "")  # her own voice leaking in
assert stop("I can stop the music or switch it") == (False, "")  # "stop" too far in to be an order


class StubVoice:
    def __init__(self):
        self.stopped = False
    def stop(self):
        self.stopped = True

def heard(nia, text, duration=1.0):
    nia.on_line(SimpleNamespace(text=text, duration=duration, last_transcription_latency_ms=100))

nia = WakeWordDetector.__new__(WakeWordDetector)  # skip mic, models and LLM
nia.settings = {**settings.DEFAULTS}
nia.lines, nia.tts = __import__("queue").Queue(), StubVoice()

with nia.working():
    heard(nia, "Sure, playing some lofi beats now")  # her own voice: ignored, nothing stopped
    assert nia.lines.empty() and not CANCEL.is_set() and not nia.tts.stopped
    heard(nia, "Hey Nia, pause the music")  # stops her, and the command comes next
    assert CANCEL.is_set() and nia.tts.stopped
    assert nia.lines.get_nowait().text == "pause the music"
assert not nia.busy

heard(nia, "the tail of her last sentence", duration=2.0)  # began while she was talking
assert nia.lines.empty()
nia.busy_ended = time.time() - 5
heard(nia, "what time is it")  # well after she finished: a normal command
assert nia.lines.get_nowait().text == "what time is it"
with nia.working():
    assert not CANCEL.is_set(), "each turn starts uncancelled"

# Voice volume: applied to the TTS, clamped to 0.1-1 (more clips), saved for the next start
class StubTTS:
    def volume(self, level):
        self.level = level

with tempfile.TemporaryDirectory() as tmp:
    settings.PATH = Path(tmp) / "settings.json"  # never touch the real settings.json
    nia = WakeWordDetector.__new__(WakeWordDetector)  # skip mic, models and LLM
    nia.settings, nia.tts = settings.load(), StubTTS()
    set_volume, change_volume = nia._voice_tools()
    assert set_volume.invoke({"percent": 50}) == "Your voice volume is now 50%"
    assert nia.tts.level == 0.5 and settings.load()["voice_volume"] == 0.5
    assert change_volume.invoke({"step": 25}) == "Your voice volume is now 75%"
    assert nia.tts.level == 0.75 and settings.load()["voice_volume"] == 0.75
    assert "maximum" in change_volume.invoke({"step": 100})  # capped, and says so
    assert nia.tts.level == 1.0
    change_volume.invoke({"step": -500})
    assert nia.tts.level == 0.1  # never silent

print("ok")
