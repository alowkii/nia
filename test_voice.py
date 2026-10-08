"""Checks the voice helpers: the fuzzy wake-phrase matcher on real Moonshine transcripts,
the cleanup of replies before they're spoken, and NIA's own voice volume. Run: python test_voice.py"""
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import settings
from agent.chat import CANCEL
from utils.hud import envelope, hud
from voice_assistant.voice_assistant import (State, WakeWordDetector, first_part, fragment, name_only, pick_greeting,
                                             plain, speakable, stop_request, voice_language, wake_command)


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

# Replies from a real session: markdown lists became 94 s of speech
tools_reply = """Here's a rundown of what I've got on deck, sir:

**Music (Spotify)**
- Play a specific track, album, or playlist by name
- Pause / resume playback

**Your PC (Windows)**
1. List files and folders
2. Run shell commands via `cmd.exe`"""
assert plain(tools_reply) == ("Here's a rundown of what I've got on deck, sir: Music (Spotify). Play a specific track, "
                              "album, or playlist by name. Pause / resume playback. Your PC (Windows). List files and "
                              "folders. Run shell commands via cmd.exe.")
now, rest = first_part(tools_reply, 12)
assert now == "Here's a rundown of what I've got on deck, sir: Music (Spotify)." and rest.startswith("Play a specific")
assert first_part("Done, sir.", 40) == ("Done, sir.", "")  # short replies are untouched
long_sentence = "word " * 60 + "end."
assert first_part(long_sentence, 40)[0] == long_sentence.strip(), "always at least one whole sentence"

# Her name on its own, as Moonshine hears it: an attention call, not "play a song called Near"
for text in ["Near.", "Nia", "Henia.", "Hey Nia.", "So near.", "Mia?"]:
    assert name_only(text), text
for text in ["Near my heart", "play near by", "Nia, play lofi", "What's near me?"]:
    assert not name_only(text), text
# Scraps from clipped recordings are asked about, not acted on ("Jo." became running jo.exe)
for text in ["Jo.", "Uh.", "Hm"]:
    assert fragment(text), text
for text in ["Pause.", "Next", "Yes.", "Skip", "Thunderstruck", "Play lofi"]:
    assert not fragment(text), text

# Greetings: the setting's lines plus ones for the hour, never the same twice running
said = [None]
for _ in range(50):
    said.append(pick_greeting("At your service, sir. | You rang, sir?", 2, said[-1]))
    assert said[-1] != said[-2]
assert {"At your service, sir.", "You rang, sir?", "Still up, sir?"} <= set(said)  # 2 am: late-night lines too
assert not {"Good morning, sir.", "Good evening, sir."} & set(said)
assert "Good morning, sir." in {pick_greeting("", 9) for _ in range(50)}
assert pick_greeting("", 23).startswith(("Still", "Burning", "At this hour"))  # only the hour's lines if none set

# British voices load with British pronunciation, or Moonshine refuses them
assert [voice_language(v) for v in ("kokoro_bm_george", "piper_en_GB-alan-medium", "kokoro_af_heart")] == \
    ["en_gb", "en_gb", "en_us"]

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
nia.interrupted = __import__("threading").Event()
nia.state = State.LISTENING

with nia.working():
    heard(nia, "Sure, playing some lofi beats now")  # her own voice: ignored, nothing stopped
    assert nia.lines.empty() and not CANCEL.is_set() and not nia.interrupted.is_set()
    heard(nia, "Hey Nia, pause the music")  # stops her, and the command comes next
    assert CANCEL.is_set() and nia.interrupted.is_set()
    # ...but only by flagging it: stopping the speaker on the mic's thread crashed the process
    assert not nia.tts.stopped
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

# The window: a spoken sentence becomes a loudness curve it follows, 60 values a second
rate = 24000
loud_then_quiet = np.concatenate([np.sin(np.arange(rate) / 3) * 0.8, np.sin(np.arange(rate) / 3) * 0.1])
curve = envelope(loud_then_quiet, rate)
assert len(curve) == 120 and max(curve) == 1.0 and all(0 <= v <= 1 for v in curve)
assert min(curve[:60]) > 0.9 and max(curve[60:]) < 0.2
assert envelope([], rate) == []

# The voice loop driven from the window, as nia.py does it: over an authenticated local connection, typed
# text and mic presses come in and her states and words go out. Mic, voice and agent are stand-ins.
import secrets
import threading
from multiprocessing.connection import Listener

key = secrets.token_bytes(16)
listener = Listener(("127.0.0.1", 0), authkey=key)
accepted = []
threading.Thread(target=lambda: accepted.append(listener.accept()), daemon=True).start()

class StubAgent:
    pending = None
    def respond(self, text):
        return f"Done: {text}, sir."

nia = WakeWordDetector.__new__(WakeWordDetector)  # skip mic, models and LLM
nia.settings = {**settings.DEFAULTS, "spoken_replies": False, "wake_word": False}  # shown, never played
nia.lines, nia.interrupted = __import__("queue").Queue(), threading.Event()
nia.state, nia.deadline, nia.busy, nia.busy_ended = State.LISTENING, 0.0, False, 0.0
nia.rest, nia.greeted, nia.held, nia.quitting = "", None, False, False
nia.mic, nia.tts, nia.assistant = SimpleNamespace(start=lambda: None, close=lambda: None), \
    SimpleNamespace(close=lambda: None), StubAgent()
hud.connect(on_command=nia.command, address=f"{listener.address[1]}:{key.hex()}")
while not accepted:
    time.sleep(0.01)
window = accepted[0]
loop = threading.Thread(target=nia.run, daemon=True)
loop.start()

def events_until(**want):
    """Messages from the voice loop up to the first that matches want"""
    seen = []
    while window.poll(5):
        seen.append(window.recv())
        if all(seen[-1].get(k) == v for k, v in want.items()):
            return seen
    raise AssertionError(f"never got {want}; got {seen}")

assert events_until(state="asleep")  # ready, waiting
nia.on_line(SimpleNamespace(text="Hey Nia, what time is it?", duration=1.0, last_transcription_latency_ms=90))
time.sleep(0.3)
assert not window.poll(0.2), "with the wake word off, speech alone doesn't wake her"
window.send({"type": "text", "text": "pause the music"})
seen = events_until(state="awake")
assert {"you": "pause the music"} in seen and {"state": "thinking"} in seen
assert {"nia": "Done: pause the music, sir."} in seen, "the reply shows in the window even when not spoken"
window.send({"type": "text", "text": "Hey Nia"})  # typed her name: the instant greeting, not 15 s of Bonsai
seen = events_until(state="awake")
said = [m["nia"] for m in seen if "nia" in m]
assert said and "Done" not in said[0] and not any("you" in m for m in seen)
window.send({"type": "sleep"})
assert events_until(state="asleep")
window.send({"type": "wake"})
assert events_until(state="awake")
window.send({"type": "settings", "values": {"push_to_talk": True}})  # applies at once
window.send({"type": "ptt_up"})  # released with nothing said: back to sleep once the last words are in
started = time.time()
assert events_until(state="asleep") and 2 < time.time() - started < 4
window.send({"type": "quit"})
loop.join(5)
assert not loop.is_alive(), "quit from the window stops the loop"
hud.conn = None

print("ok")
