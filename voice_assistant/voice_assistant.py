import os
import random
import re
import subprocess
import sys
import time
import queue
import threading
import unicodedata
from contextlib import contextmanager
from datetime import datetime
from difflib import SequenceMatcher
from types import SimpleNamespace
from urllib.parse import urlsplit
import numpy as np
import sounddevice as sd
from langchain_core.tools import tool
from moonshine_voice import MicTranscriber, ModelArch, TextToSpeech
from pycaw.pycaw import AudioUtilities

import settings
import agent.chat as agent_chat
from agent import claude
from agent.chat import CANCEL, AssistantModel, ensure_llm_server
from utils import mics
from utils.hud import hud

# Set logging
from utils.logger import logging
logger = logging.getLogger(__name__)

STT_MODELS = {"tiny": ModelArch.TINY_STREAMING, "small": ModelArch.SMALL_STREAMING, "medium": ModelArch.MEDIUM_STREAMING}


# Greetings for the time of day, from each starting hour; mixed in with the greeting setting's own
TIME_GREETINGS = [
    (5, ["Good morning, sir.", "Morning, sir. What's first on the list?", "Good morning, sir. Systems are up, "
                                                                             "coffee remains your department."]),
    (12, ["Good afternoon, sir.", "Afternoon, sir. What can I do for you?"]),
    (17, ["Good evening, sir.", "Evening, sir. What are we doing tonight?"]),
    (22, ["Still up, sir?", "Burning the midnight oil, sir?", "At this hour, sir? Very well, I'm listening."]),
]


# Called again soon after talking, she answers like someone already in the room - "Good evening" every time
# sounded like a bot. A real greeting only after this long without a word
RECENT = 30 * 60
ACKS = ["Sir?", "Yes, sir?", "Go ahead, sir.", "I'm listening, sir."]


def pick_greeting(greetings, hour, last=None, recent=False):
    """A random greeting - one of the setting's ("|" between them) or one for the hour - never the last one said;
    just a short "Sir?" if they were talking a moment ago"""
    by_hour = next((lines for start, lines in reversed(TIME_GREETINGS) if hour >= start), TIME_GREETINGS[-1][1])
    pool = ACKS if recent else [g.strip() for g in greetings.split("|") if g.strip()] + by_hour
    return random.choice([g for g in pool if g != last] or pool)


CHIME_NOTES = ((0.0, 440.0, 0.9), (0.16, 659.25, 1.0))  # (start s, Hz, level): A4 rising a fifth to E5
CHIME_PAD = (220.0, 329.63, 554.37)  # A3, E4, C#5: an A major chord swelling softly underneath
CHIME_LENGTH = 1.6


def chime(rate=24000, peak=0.3):
    """The sound of NIA becoming ready, ~1.6 s: two warm felt-mallet notes rising a fifth over a soft chord, in a
    little room. Rounded 12 ms attacks (no click), low-mid pitches (nothing shrill). Made here, not a file"""
    t = np.arange(int(rate * CHIME_LENGTH)) / rate
    ease_in = lambda x, length: 0.5 - 0.5 * np.cos(np.pi * np.clip(x / length, 0, 1))
    sound = np.zeros(len(t))
    for start, freq, level in CHIME_NOTES:
        local = np.clip(t - start, 0, None)
        tone = (np.sin(2 * np.pi * freq * local) + 0.2 * np.sin(2 * np.pi * freq * 1.0025 * local)) / 1.2  # warmth
        tone = tone * np.exp(-local / 0.55) + 0.18 * np.sin(4 * np.pi * freq * local) * np.exp(-local / 0.10)
        sound += level * tone * ease_in(local, 0.012) * (t >= start)
    swell = ease_in(t, 0.18) * np.exp(-np.clip(t - 0.18, 0, None) / 0.55)
    sound += 0.10 * swell * sum(np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * f * 1.004 * t) for f in CHIME_PAD)
    # The room: the sound convolved with a dark, decaying noise burst, mixed in quietly
    tail = np.arange(int(rate * 1.05))
    burst = np.convolve(np.random.default_rng(7).standard_normal(len(tail)) * np.exp(-tail / (rate * 0.35)),
                        np.ones(24) / 24, "same")
    size = len(sound) + len(burst)
    wet = np.fft.irfft(np.fft.rfft(sound, size) * np.fft.rfft(burst, size))[:len(sound)]
    sound = 0.78 * sound + 0.22 * wet / np.abs(wet).max() * np.abs(sound).max()
    sound *= np.clip((CHIME_LENGTH - t) / 0.25, 0, 1)  # a soft tail-out
    return (sound / np.abs(sound).max() * peak).astype(np.float32)


def play_chime(volume=1.0, rate=24000):
    """Play the chime without waiting for it to finish"""
    sd.play(chime(rate) * volume, rate)


def voice_language(voice):
    """British voices (kokoro_bm_george, piper_en_GB-alan-medium) need British pronunciation to load"""
    return "en_gb" if re.search(r"^kokoro_b[fm]_|en_GB", voice) else "en_us"


# How Moonshine has actually heard the wake phrase (spaces and punctuation dropped): taken as exact matches,
# so "Heineia" (0.77) wakes her without lowering the threshold for everything - "Hey, nice..." also scores 0.77.
# Only ones seen in the logs
SOUNDALIKES = {"heynia": {"heineia", "heinear", "henear", "henia", "hania", "heania", "heaenea", "tenia"}}
# A line that starts with her name is addressed to her, as with a person: "Nia, what's playing?" (it scored 0.67
# against "Hey Nia" and was ignored). Not "near" or "Mia" - ordinary words that start ordinary sentences
ADDRESSED = {"nia", "nea", "neah", "niya"}


def wake_match(text, phrase):
    """How closely the best spot in text matches the wake phrase (0-1), and what follows that spot.

    Fuzzy, because the recogniser hears "Hey Nia" as "Hey, Nia." or "Heania" - so the phrase is
    compared, spaces and punctuation dropped, to short word windows. Every position is tried:
    with music or talk in the room the phrase often lands mid-line, not at the start.
    """
    words = list(re.finditer(r"[a-z']+", text.lower()))
    if words and words[0].group() in ADDRESSED:
        return 1.0, text[words[0].end():].lstrip(" ,.!?;:-")
    target = "".join(phrase.lower().split())
    size = len(phrase.split())
    best, end = 0.0, 0
    for start in range(len(words)):
        for n in range(1, size + 2):  # the phrase may come out as fewer or more words
            window = "".join(w.group() for w in words[start:start + n])
            score = 1.0 if window in SOUNDALIKES.get(target, ()) else SequenceMatcher(None, window, target).ratio()
            if score > best:
                best, end = score, min(start + n, len(words))
    if not words:
        return 0.0, ""
    return best, text[words[end - 1].end():].lstrip(" ,.!?;:-")


def stop_request(text, phrase, threshold):
    """For an utterance heard while NIA is busy: (should she stop, the command that came with it).
    "Stop" / "Nia, stop" stop her; "Hey Nia" stops her too, and "Hey Nia, pause the music" also
    hands back "pause the music". Only near the start of the line, so her own voice leaking into
    the mic mid-sentence doesn't count."""
    words = re.findall(r"[a-z']+", text.lower())
    if "stop" in words[:2]:  # "Stop", "Nia, stop", "Okay, stop" - but not her own "I can stop the music"
        return True, ""
    if wake_match(" ".join(words[:4]), phrase)[0] >= threshold:
        rest = wake_match(text, phrase)[1]
        return True, "" if re.fullmatch(r"\W*stop\W*", rest.lower()) else rest
    return False, ""


def wake_command(text, phrase, threshold):
    """Whatever follows the wake phrase ('' for just the phrase), or None if it isn't there."""
    score, rest = wake_match(text, phrase)
    return rest if score >= threshold else None


# Symbols the voice gets wrong, mapped to what should be said
SPOKEN = {"—": ", ", "–": ", ", "…": "...", "‘": "'", "’": "'", "“": '"', "”": '"',
          "%": " percent", "&": " and ", "/": " "}


def site_name(match):
    """'https://www.youtube.com/results?q=x' -> 'the youtube link'. Reading out a whole URL takes
    seconds, and the TTS garbles even "youtube.com", so only the site's name is said"""
    url = match.group(0)
    host = urlsplit(url if "://" in url else f"http://{url}").hostname or ""
    labels = host.removeprefix("www.").split(".")
    # google.co.in -> google: skip a second-level suffix like co / com / org
    name = labels[-3] if len(labels) > 2 and labels[-2] in ("co", "com", "org", "net", "gov", "ac") else labels[-2 if len(labels) > 1 else 0]
    return f"the {name} link"


def plain(text):
    """Markdown to plain spoken sentences: the LLM sometimes answers with bullet lists and bold despite the
    prompt. List items and lines become sentences; markers, bold and code ticks go"""
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.M)  # headings
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", text, flags=re.M)  # bullets and numbered items
    text = re.sub(r"\*\*|__|`", "", text)  # bold, code
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(line if line[-1] in ".!?:;," else line + "." for line in lines)


SHORT_REST = 35  # words: a leftover this short is said, not offered


def first_part(text, max_words, short_rest=SHORT_REST):
    """(what to say now, the rest): whole sentences up to about max_words, at least one. A long answer
    took 94 s to speak in one session; the rest is offered instead of read out - but only when there's
    real length left. Offering "Shall I go on?" for one more sentence cut ordinary answers in half and
    hid her own closing questions"""
    sentences = re.split(r"(?<=[.!?])\s+", plain(text))
    now, words = [], 0
    for sentence in sentences:
        if now and words + len(sentence.split()) > max_words:
            break
        now.append(sentence)
        words += len(sentence.split())
    rest = " ".join(sentences[len(now):])
    if len(rest.split()) <= short_rest:
        return " ".join(sentences), ""
    return " ".join(now), rest


# The chirpy sign-offs the prompt forbids but the 1-bit model still adds now and then ("Anything else you'd like
# to know?") - JARVIS doesn't ask
SIGN_OFF = re.compile(r"\b(anything else|let me know if|anything (specific|more|next))\b", re.I)


def drop_filler(text):
    """text without a closing "anything else?"-type sentence - unless that's all there is"""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) > 1 and SIGN_OFF.search(sentences[-1]):
        sentences.pop()
    return " ".join(sentences)


MORE = re.compile(r"\b(yes|yeah|yep|sure|go on|continue|keep going|tell me more|more|please do|ok(ay)?)\b")
NOT_MORE = re.compile(r"\b(no|nope|stop|enough|that's (it|fine)|never mind)\b")
# How Moonshine hears the name on its own - "Near." was taken as a command and played a song called Near
NAME_SOUNDS = {"nia", "near", "nea", "neah", "niya", "mia", "henia", "hania", "heania", "heynia"}
FILLER = {"hey", "hi", "so", "oh", "ok", "okay", "um", "uh"}
# Short words that are real commands; any other single word of 3 letters or fewer ("Jo.") is a fragment
SHORT_COMMANDS = {"yes", "no", "yep", "nah", "ok", "hi", "go", "up", "off", "on", "mute", "play", "next", "skip",
                  "back", "stop", "more", "sure", "nope"}


def name_only(text):
    """Just her name (or how it's misheard) - an attention call, not a command"""
    words = re.findall(r"[a-z']+", text.lower())
    return bool(words) and set(words) <= NAME_SOUNDS | FILLER and bool(set(words) & NAME_SOUNDS)


def fragment(text):
    """A lone scrap like "Jo." - from a clipped recording, not a request worth guessing at"""
    words = re.findall(r"[a-z']+", text.lower())
    return len(words) == 1 and len(words[0]) <= 3 and words[0] not in SHORT_COMMANDS


HESITATIONS = {"uh", "uhh", "um", "umm", "hm", "hmm", "mm", "er", "erm", "ah", "eh"}


def hesitation(text):
    """Only "uh", "um", "hmm"... - someone gathering their thoughts, which deserves silence, not a reply"""
    words = re.findall(r"[a-z']+", text.lower())
    return bool(words) and all(word in HESITATIONS for word in words)


def cut_off(text):
    """Moonshine ends a line at a pause and marks one that stopped mid-thought with "..." """
    return text.rstrip().endswith(("...", "…"))


def speakable(text):
    """Text as it should be heard. URLs become "the youtube link". The TTS reads any non-ASCII
    character as the letter "L" (dashes, emoji, curly quotes) and skips % and &, so those become
    words or pauses, accents are dropped (Beyoncé -> Beyonce) and anything left that isn't ASCII goes."""
    # A URL never ends in punctuation - "...youtube.com." keeps its full stop for the sentence
    text = re.sub(r"\b(?:https?://|www\.)[^\s\"'<>]*[^\s\"'<>.,!?;:)]", site_name, text)
    for symbol, spoken in SPOKEN.items():
        text = text.replace(symbol, spoken)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)  # "sir , the" -> "sir, the"
    return re.sub(r"([,.!?;:])(?:\s*,)+", r"\1", text).strip()  # "sir,, the" -> "sir, the"


# ponytail: original volumes live only in memory - force-stopping NIA mid-sentence leaves other apps ducked
# in the Windows mixer; persist them to a file and restore at startup if that bites
def duck(level):
    """Turn every other app's volume down to level x its own (Windows mixer); returns a function that restores them"""
    saved = []
    for session in AudioUtilities.GetAllSessions():
        if session.ProcessId in (0, os.getpid()):  # system sounds, and NIA's own voice
            continue
        try:
            volume = session.SimpleAudioVolume
            saved.append((volume, volume.GetMasterVolume()))
            volume.SetMasterVolume(saved[-1][1] * level, None)
        except Exception:  # the app closed between listing and ducking
            pass

    def restore():
        for volume, original in saved:
            try:
                volume.SetMasterVolume(original, None)
            except Exception:
                pass
    return restore


class WakeWordDetector:
    def __init__(self):
        self.awake = False  # after the wake phrase: listening for a command
        self.settings = s = settings.load()
        self.deadline = 0.0  # when being awake times out
        self.busy = False  # thinking or talking: only a stop phrase gets through
        self.busy_ended = 0.0
        self.interrupted = threading.Event()  # "stop" heard: speak() cuts the voice off
        self.rest = ""  # the unspoken part of a long answer, said if the user asks her to go on
        self.greeted = None  # the last greeting, so the next one differs
        self.last_talk = 0.0  # when she last spoke with them: a greeting only after a while
        self.voice = threading.Lock()  # one sentence at a time, whichever thread speaks
        self.held = False  # the window's mic button is held down (push-to-talk)
        self.quitting = False
        self.restart_after = None  # "assistant", "all" or "off", set by restart_myself during a turn

        # Everything the voice loop acts on arrives here: lines heard (on_line), and from the window
        # typed text, mic presses and "quit" (command) - so all state changes happen on the loop's thread
        self.lines = queue.Queue()
        hud.connect(on_command=self.command)  # the window shows her booting while the models load
        hud.show("booting")

        # Speech to text (Moonshine streaming, CPU), always on: it hears the wake phrase, the
        # commands, and "stop" while she's busy. Its VAD ends each utterance; see on_line.
        logger.info(f"Loading Moonshine speech-to-text ({s['stt_model']})...")
        self.mic = (MicTranscriber().model_arch(STT_MODELS[s["stt_model"]]).device(self.mic_index())
                    .on_line(self.on_line).on_text(self.on_text).load())
        self.mic.set_keyterms([s["wake_phrase"].split()[-1].title()])  # bias towards the name, e.g. "Nia"

        # Text to speech (CPU)
        logger.info(f"Loading text-to-speech ({s['voice']})...")
        self.tts = TextToSpeech().language(voice_language(s["voice"])).voice(s["voice"]).volume(s["voice_volume"]).load()

        # Assistant model (Bonsai on llama-server, GPU)
        logger.info("Starting the LLM server if needed...")
        ensure_llm_server()
        self.assistant = AssistantModel(extra_tools=self._voice_tools())
        # Claude takes 15-40 s, a shell command or search can take a while: say so once, don't go silent
        agent_chat.ANNOUNCE = lambda: self.speak("One moment, sir.")
        claude.ANNOUNCE = agent_chat.announce
        # A change built on a branch in the background: she says how it went, unprompted, on the loop's thread
        claude.ON_DONE = lambda text: self.lines.put(SimpleNamespace(
            control="announce", text=text, duration=0, last_transcription_latency_ms=0))
        self.assistant.warm_up()
        logger.info("Assistant ready!")

    def set_voice_volume(self, level):
        """NIA's own voice gain, clamped to 0.1-1 and saved so it survives restarts.
        The TTS already peaks at full scale at 1.0, so more would only clip and distort."""
        level = round(min(max(level, 0.1), 1.0), 2)
        self.settings["voice_volume"] = level
        self.tts.volume(level)
        saved = settings.load()
        saved["voice_volume"] = level
        settings.save(saved)
        logger.info(f"Voice volume set to {level:.0%}")
        if level == 1.0:
            return "Your voice is at 100%, its maximum - for louder, the user can turn up the Windows volume"
        return f"Your voice volume is now {level:.0%}"

    def _voice_tools(self):
        @tool
        def set_voice_volume(percent: int) -> str:
            """Set how loud YOUR OWN voice is, in percent (10-100, 100 is the maximum), e.g. "talk at 50 percent".
            Not for music - use the Spotify volume tools for that"""
            return self.set_voice_volume(percent / 100)

        @tool
        def change_voice_volume(step: int) -> str:
            """Make YOUR OWN voice louder (positive step) or quieter (negative step), in percent.
            For "speak up", "talk quieter", "you're too loud". Use 25 or -25 unless told an amount.
            Not for music - use the Spotify volume tools for that"""
            return self.set_voice_volume(self.settings["voice_volume"] + step / 100)

        @tool
        def restart_myself(whole_system: bool = True) -> str:
            """Restart, right after this reply. When the user asks you to restart or reboot: always the whole
            system - your window, the language model, speech and agent, all reloaded so any change to your code
            takes effect (~40 s). Only when you restart on your own because something of yours is stuck (you keep
            mishearing, a tool keeps failing): whole_system=False restarts just your speech and agent (~10 s).
            The conversation starts fresh; memory stays"""
            self.restart_after = "all" if whole_system else "assistant"
            return ("Restarting right after this reply - it's happening, so don't ask whether to. Tell the user in "
                    f"a few words: back in about {40 if whole_system else 10} seconds")

        @tool
        def shut_down_myself() -> str:
            """Switch yourself - the NIA system - off completely, right after this reply: your window, the language
            model, speech and agent. For "shut down", "shut down the system", "turn off", "power down", "switch off"
            or "quit": these always mean you, never the PC. Only when the computer is named ("shut down the computer
            system", "...the PC", "...the laptop") is it the machine: that's the shell's shutdown /s /t 0 instead.
            Not a restart (that's restart_myself), and not "stop" (that only stops what you're doing). They start
            you again by running nia.py"""
            self.restart_after = "off"
            return "Shutting down right after this reply - it's happening, so don't ask whether to. Say goodbye briefly"

        return [set_voice_volume, change_voice_volume, restart_myself, shut_down_myself]

    def restart(self):
        """Hand over to a fresh copy of herself - or, for "off", switch off: nia.py's window carries either out
        for all of NIA; run on her own, she starts a new copy (unless switching off) and quits"""
        what, self.restart_after = self.restart_after, None
        logger.info("Shutting down, as asked" if what == "off" else f"Restarting myself ({what})")
        if hud.conn:
            hud.send(restart=what)
        else:
            if what != "off":
                subprocess.Popen([sys.executable, *sys.argv], creationflags=subprocess.CREATE_NEW_CONSOLE)
            self.lines.put(None)

    def on_line(self, line):
        """Each finished utterance, on Moonshine's thread. While NIA is busy, only a stop phrase gets
        through - it cuts her off and cancels the turn - so she never answers her own voice. A line
        that began while she was busy is dropped too, even if it ends after: that's her voice's tail."""
        s = self.settings
        if self.busy:
            stop, command = stop_request(line.text, s["wake_phrase"], s["wake_threshold"])
            if stop:
                logger.info(f"Interrupted by: {line.text!r}")
                # Only flags here: this is Moonshine's mic thread, and stopping the speaker from it
                # corrupted the heap and killed the process. speak() stops it on the main thread.
                CANCEL.set()
                self.interrupted.set()
                if command:  # "Hey Nia, pause the music" - do that next
                    self.lines.put(SimpleNamespace(text=command, duration=line.duration,
                                                   last_transcription_latency_ms=line.last_transcription_latency_ms))
            return
        # ponytail: estimates when the line began from its length plus ~0.5 s of end-of-speech silence
        if time.time() - line.duration - 0.5 < self.busy_ended - 0.3:
            return
        self.lines.put(line)

    def on_text(self, text):
        """Words arriving while the user is still talking: the awake window shows them as they come"""
        if self.awake and not self.busy:
            hud.send(partial=text)

    def command(self, message):
        """From the window, on its connection's thread: {"type": text | wake | sleep | ptt_down | ptt_up | quit},
        or {"type": "settings", "values": {...}} for settings that apply at once"""
        kind = message["type"]
        if kind == "settings":
            self.settings.update(message["values"])  # the loop reads self.settings each time
            if "mic_device" in message["values"]:  # switched on the loop's thread, like everything else
                self.lines.put(SimpleNamespace(control="mic", text="", duration=0, last_transcription_latency_ms=0))
            return
        if kind in ("sleep", "quit") and self.busy:  # stop her now, not when she's done
            self.quitting = kind == "quit"
            CANCEL.set()
            self.interrupted.set()
        self.lines.put(None if kind == "quit" else
                       SimpleNamespace(control=kind, text=message.get("text", ""), duration=0,
                                       last_transcription_latency_ms=0))

    def finish_sentence(self, text, wait=2.0):
        """A line cut off at a pause ("I mean, could you...") waits up to `wait` seconds for the rest, and the
        two are joined - so she answers the whole sentence instead of asking you to finish it"""
        while cut_off(text):
            try:
                more = self.lines.get(timeout=wait)
            except queue.Empty:
                break  # nothing followed: take it as it is
            if more is None or getattr(more, "control", None):  # quit, or a press in the window: not words
                self.lines.put(more)
                break
            logger.info(f"Joined a line cut off at a pause: {text!r} + {more.text.strip()!r}")
            text = text.rstrip(" .…") + " " + more.text.strip()
        return text

    def mic_index(self):
        """The input device for the chosen microphone, telling the window which one is in use"""
        name = self.settings["mic_device"]
        index = mics.index(name)
        if name and index is None:
            logger.warning(f"Microphone {name!r} isn't connected - listening through Windows' default instead")
        logger.info(f"Listening through {name if index is not None else 'the default microphone'}")
        hud.send(mic=name if index is not None else mics.DEFAULT)
        return index

    def switch_mic(self):
        """Reopen the microphone on the newly chosen device, without restarting anything else"""
        self.mic.stop()
        # ponytail: Moonshine has no public way to reopen its input stream; this relies on its _sd_stream
        if self.mic._sd_stream is not None:
            self.mic._sd_stream.close()
            self.mic._sd_stream = None
        self.mic.device(self.mic_index()).start()

    def idle(self):
        hud.show("awake" if self.awake else "asleep")

    def stay_awake(self):
        """After an exchange: listen for a follow-up until the session times out - or, with push-to-talk
        released, only for the words still being transcribed"""
        released = self.settings["push_to_talk"] and not self.held
        self.deadline = time.time() + (2.5 if released else self.settings["session_timeout"])

    @contextmanager
    def working(self):
        """NIA is thinking or talking: the mic stays live, but on_line only listens for "stop"."""
        CANCEL.clear()
        self.interrupted.clear()
        self.busy = True
        hud.show("thinking")
        try:
            yield
        finally:
            self.busy = False
            self.busy_ended = time.time()
            self.idle()

    def run(self):
        s = self.settings
        self.mic.start()
        logger.info(f"Listening for {s['wake_phrase']!r}...")
        self.idle()
        if s["ready_chime"]:  # she's ready: a soft chime, so you know without looking
            play_chime(s["voice_volume"])

        try:
            while True:
                awake = self.awake
                try:
                    line = self.lines.get(timeout=max(0, self.deadline - time.time()) if awake else None)
                except queue.Empty:
                    logger.info(f"No command - listening for {s['wake_phrase']!r} again.")
                    self.awake = False
                    self.idle()
                    continue
                if line is None:  # the window closed or asked her to stop
                    logger.info("Stopping (asked to quit)")
                    break

                control = getattr(line, "control", None)  # from the window's mic button
                if control == "announce":  # news from a background job
                    with self.working():
                        self.speak(line.text)
                    continue
                if control and control != "text":
                    logger.info(f"From the window: {control}")
                if control in ("wake", "ptt_down"):
                    self.held = control == "ptt_down"
                    self.awake = True
                    self.deadline = time.time() + s["session_timeout"]
                    self.idle()
                    continue
                if control == "ptt_up":
                    self.held = False
                    self.stay_awake()
                    continue
                if control == "mic":
                    self.switch_mic()
                    continue
                if control == "sleep":
                    self.awake, self.rest = False, ""
                    self.idle()
                    continue
                typed = control == "text"

                text = line.text.strip()
                if not typed:
                    text = self.finish_sentence(text)
                if not text:
                    continue
                if not typed and hesitation(text):  # "Uh," - you're still thinking, not asking
                    logger.info(f"Ignored hesitation: {text!r}")
                    continue

                if typed:  # typed into the window: always a command, no wake phrase needed
                    self.awake = True
                else:
                    # The phrase also counts mid-session, where people repeat it out of habit
                    score, rest = wake_match(text, s["wake_phrase"])
                    command = rest if score >= s["wake_threshold"] and s["wake_word"] else None
                    if command is None and not awake:
                        # Logged so a missed wake shows what was heard and how close it came
                        logger.info(f"Ignored {text!r} (wake match {score:.2f}, needs {s['wake_threshold']})")
                        continue
                    if command is not None:
                        if not awake:
                            logger.info(f"Wake phrase heard: {text!r}")
                            self.awake = True
                        if not command:
                            with self.working():
                                self.greet()
                            self.stay_awake()
                            continue
                        text = command  # "Hey Nia, play lofi" in one breath

                if self.rest:  # she offered to go on with a long answer
                    # Only a short reply - "Okay, play Thunderstruck" is a new request, not "go on"
                    if len(text.split()) <= 4 and MORE.search(text.lower()) and not NOT_MORE.search(text.lower()):
                        with self.working():
                            self.say(self.rest)
                        self.stay_awake()
                        continue
                    self.rest = ""
                if name_only(text) or (not typed and fragment(text) and not self.assistant.pending):
                    logger.info(f"Not a request: {text!r}")
                    with self.working():
                        if name_only(text):
                            self.greet()
                        else:
                            self.speak(f"Sorry sir, I only caught '{text.strip(' .,!?')}'. What do you need?")
                    self.stay_awake()
                    continue

                self.handle_command(text, line)
                # Only a real exchange keeps the session open - silence must time out
                self.stay_awake()

        except KeyboardInterrupt:
            logger.info("Stopping (Ctrl+C)")
        finally:
            self.cleanup()

    def handle_command(self, text, line):
        if getattr(line, "control", None) == "text":
            logger.info(f"Typed: {text!r}")
        else:
            logger.info(f"Heard: {text!r} (transcribed {line.duration:.1f}s of speech, "
                        f"{line.last_transcription_latency_ms} ms after you stopped)")
        hud.send(you=text)
        with self.working():
            started = time.perf_counter()
            try:
                reply = self.assistant.respond(text)
            except Exception:  # one bad turn mustn't kill the assistant
                logger.exception("Agent error")
                reply = "Sorry sir, something went wrong on my end. The details are in the log."
            if CANCEL.is_set():  # "stop" while she was thinking: drop whatever she was going to say
                logger.info(f"Stopped by the user after {time.perf_counter() - started:.1f}s")
                reply = "" if self.quitting else "Okay."
                self.interrupted.clear()  # the stop is handled - don't cut off the "Okay." too
            logger.info(f"Reply: {reply!r} (LLM {time.perf_counter() - started:.1f}s)")
            if reply:
                started = time.perf_counter()
                self.say(reply)
                logger.info(f"Spoke in {time.perf_counter() - started:.1f}s")
        self.last_talk = time.time()
        if self.restart_after:  # asked for during the turn: only now, once she's said so
            self.restart()

    def say(self, reply):
        """Speak a reply in plain sentences, at most about max_spoken_words of it; the rest waits for "go on" """
        now, self.rest = first_part(drop_filler(reply), self.settings["max_spoken_words"])
        self.speak(now + (" Shall I go on, sir?" if self.rest else ""))

    def greet(self):
        recent = time.time() - self.last_talk < RECENT
        self.greeted = pick_greeting(self.settings["greeting"], datetime.now().hour, self.greeted, recent)
        self.speak(self.greeted)
        self.last_talk = time.time()

    def speak(self, text):
        """Speak text over ducked app audio. Call inside working(), so "stop" can cut it off.

        Not tts.say()/stop(): stopping while Moonshine is mid-synthesis crashed the process
        (heap corruption, segfault). Instead one helper thread synthesizes a sentence at a time -
        never interrupted, never two native calls at once - while this thread plays them with
        sounddevice and, on "stop", just stops playback. The helper finishes its current sentence
        and quits, so a stop in the first second waits for that sentence (under ~1 s)."""
        with self.voice:  # "One moment" from a slow step's timer and the reply never talk over each other
            self._speak(text)

    def _speak(self, text):
        hud.send(nia=text)  # the window shows every reply, spoken or not
        if not self.settings["spoken_replies"]:
            logger.info(f"Shown, not spoken: {text}")
            return
        logger.info(f"Speaking: {text}")
        sentences =[s for s in re.split(r"(?<=[.!?])\s+", speakable(text)) if s.strip()]
        ready = queue.Queue()

        def synthesize():  # the next sentence is made while the current one plays
            try:
                for sentence in sentences:
                    if self.interrupted.is_set():
                        break
                    ready.put(self.tts.synthesize(sentence, speed=self.settings["voice_speed"],
                                                  volume=self.settings["voice_volume"]))
            except Exception as e:
                logger.error(f"TTS error: {e}")
            finally:
                ready.put(None)

        restore = duck(self.settings["duck_level"]) if self.settings["duck_level"] < 1 else (lambda: None)
        maker = threading.Thread(target=synthesize, daemon=True)
        maker.start()
        try:
            while (speech := ready.get()) is not None:
                pcm, rate = speech
                hud.speak(pcm, rate)
                sd.play(np.asarray(pcm, dtype=np.float32), rate)
                ends = time.time() + len(pcm) / rate
                while time.time() < ends and not self.interrupted.is_set():
                    time.sleep(0.03)
                if self.interrupted.is_set():
                    sd.stop()
                    hud.show("thinking")  # the window stops mouthing words she no longer says
                    break
                sd.wait()  # the last few milliseconds still in the buffer, so endings aren't clipped
        except Exception as e:
            logger.error(f"TTS error: {e}")
        finally:
            restore()  # even after an error or a "stop", so other apps aren't left quiet
            maker.join(timeout=5)  # never leave a synthesis running into the next one

    def cleanup(self):
        self.mic.close()
        self.tts.close()
