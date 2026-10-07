import os
import re
import sys
import time
import enum
import queue
import threading
import unicodedata
from contextlib import contextmanager
from difflib import SequenceMatcher
from types import SimpleNamespace
from urllib.parse import urlsplit
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from langchain_core.tools import tool
from moonshine_voice import MicTranscriber, ModelArch, TextToSpeech
from pycaw.pycaw import AudioUtilities

# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import settings
from agent.chat import CANCEL, AssistantModel, ensure_llm_server

# Set logging
from utils.logger import logging
logger = logging.getLogger(__name__)

STT_MODELS = {"tiny": ModelArch.TINY_STREAMING, "small": ModelArch.SMALL_STREAMING, "medium": ModelArch.MEDIUM_STREAMING}


def wake_match(text, phrase):
    """How closely the best spot in text matches the wake phrase (0-1), and what follows that spot.

    Fuzzy, because the recogniser hears "Hey Nia" as "Hey, Nia." or "Heania" - so the phrase is
    compared, spaces and punctuation dropped, to short word windows. Every position is tried:
    with music or talk in the room the phrase often lands mid-line, not at the start.
    """
    words = list(re.finditer(r"[a-z']+", text.lower()))
    target = "".join(phrase.lower().split())
    size = len(phrase.split())
    best, end = 0.0, 0
    for start in range(len(words)):
        for n in range(1, size + 2):  # the phrase may come out as fewer or more words
            window = "".join(w.group() for w in words[start:start + n])
            score = SequenceMatcher(None, window, target).ratio()
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


class State(enum.Enum):
    LISTENING = 1
    COMMAND_MODE = 2


class WakeWordDetector:
    def __init__(self):
        self.state = State.LISTENING
        self.settings = s = settings.load()
        self.deadline = 0.0  # when COMMAND_MODE times out
        self.busy = False  # thinking or talking: only a stop phrase gets through
        self.busy_ended = 0.0
        self.interrupted = threading.Event()  # "stop" heard: speak() cuts the voice off

        # Speech to text (Moonshine streaming, CPU), always on: it hears the wake phrase, the
        # commands, and "stop" while she's busy. Its VAD ends each utterance; see on_line.
        logger.info(f"Loading Moonshine speech-to-text ({s['stt_model']})...")
        self.lines = queue.Queue()
        self.mic = (MicTranscriber().model_arch(STT_MODELS[s["stt_model"]])
                    .on_line(self.on_line).load())
        self.mic.set_keyterms([s["wake_phrase"].split()[-1].title()])  # bias towards the name, e.g. "Nia"

        # Text to speech (CPU)
        logger.info(f"Loading text-to-speech ({s['voice']})...")
        self.tts = TextToSpeech().language("en_us").voice(s["voice"]).volume(s["voice_volume"]).load()

        # Assistant model (Bonsai on llama-server, GPU)
        logger.info("Starting the LLM server if needed...")
        ensure_llm_server()
        self.assistant = AssistantModel(extra_tools=self._voice_tools())
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

        return [set_voice_volume, change_voice_volume]

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

    @contextmanager
    def working(self):
        """NIA is thinking or talking: the mic stays live, but on_line only listens for "stop"."""
        CANCEL.clear()
        self.interrupted.clear()
        self.busy = True
        try:
            yield
        finally:
            self.busy = False
            self.busy_ended = time.time()

    def run(self):
        s = self.settings
        self.mic.start()
        logger.info(f"Listening for {s['wake_phrase']!r}...")

        try:
            while True:
                awake = self.state == State.COMMAND_MODE
                try:
                    line = self.lines.get(timeout=max(0, self.deadline - time.time()) if awake else None)
                except queue.Empty:
                    logger.info(f"No command for {s['session_timeout']}s - listening for {s['wake_phrase']!r} again.")
                    self.state = State.LISTENING
                    continue

                text = line.text.strip()
                if not text:
                    continue

                # The phrase also counts mid-session, where people repeat it out of habit
                score, rest = wake_match(text, s["wake_phrase"])
                command = rest if score >= s["wake_threshold"] else None
                if command is None and not awake:
                    # Logged so a missed wake shows what was heard and how close it came
                    logger.info(f"Ignored {text!r} (wake match {score:.2f}, needs {s['wake_threshold']})")
                    continue
                if command is not None:
                    if not awake:
                        logger.info(f"Wake phrase heard: {text!r}")
                        self.state = State.COMMAND_MODE
                    if not command:
                        with self.working():
                            self.speak(s["greeting"])
                        self.deadline = time.time() + s["session_timeout"]
                        continue
                    text = command  # "Hey Nia, play lofi" in one breath

                self.handle_command(text, line)
                # Only a real exchange keeps the session open - silence must time out
                self.deadline = time.time() + s["session_timeout"]

        except KeyboardInterrupt:
            logger.info("Stopping (Ctrl+C)")
        finally:
            self.cleanup()

    def handle_command(self, text, line):
        logger.info(f"Heard: {text!r} (transcribed {line.duration:.1f}s of speech, "
                    f"{line.last_transcription_latency_ms} ms after you stopped)")
        with self.working():
            started = time.perf_counter()
            try:
                reply = self.assistant.respond(text)
            except Exception:  # one bad turn mustn't kill the assistant
                logger.exception("Agent error")
                reply = "Sorry sir, something went wrong on my end. The details are in the log."
            if CANCEL.is_set():  # "stop" while she was thinking: drop whatever she was going to say
                logger.info(f"Stopped by the user after {time.perf_counter() - started:.1f}s")
                reply = "Okay."
                self.interrupted.clear()  # the stop is handled - don't cut off the "Okay." too
            logger.info(f"Reply: {reply!r} (LLM {time.perf_counter() - started:.1f}s)")
            if reply:
                started = time.perf_counter()
                self.speak(reply)
                logger.info(f"Spoke in {time.perf_counter() - started:.1f}s")

    def speak(self, text):
        """Speak text over ducked app audio. Call inside working(), so "stop" can cut it off.

        Not tts.say()/stop(): stopping while Moonshine is mid-synthesis crashed the process
        (heap corruption, segfault). Instead one helper thread synthesizes a sentence at a time -
        never interrupted, never two native calls at once - while this thread plays them with
        sounddevice and, on "stop", just stops playback. The helper finishes its current sentence
        and quits, so a stop in the first second waits for that sentence (under ~1 s)."""
        logger.info(f"Speaking: {text}")
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", speakable(text)) if s.strip()]
        ready = queue.Queue()

        def synthesize():  # the next sentence is made while the current one plays
            try:
                for sentence in sentences:
                    if self.interrupted.is_set():
                        break
                    ready.put(self.tts.synthesize(sentence, volume=self.settings["voice_volume"]))
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
                sd.play(np.asarray(pcm, dtype=np.float32), rate)
                ends = time.time() + len(pcm) / rate
                while time.time() < ends and not self.interrupted.is_set():
                    time.sleep(0.03)
                if self.interrupted.is_set():
                    sd.stop()
                    break
                sd.wait()  # the last few milliseconds still in the buffer, so endings aren't clipped
        except Exception as e:
            logger.error(f"TTS error: {e}")
        finally:
            restore()  # even after an error or a "stop", so other apps aren't left quiet
            maker.join(timeout=5)  # never leave a synthesis running into the next one

    # TODO: This is creating too much latency
    # def valid_command_from_your_voice(self):
    #     encoder = VoiceEncoder()
    #     reference_embedding = np.load("voice_samples/my_voice_embedding.npy")

    #     wav_new = preprocess_wav("temp/command.wav")
    #     embedding_new = encoder.embed_utterance(wav_new)

    #     similarity = 1 - cosine(reference_embedding, embedding_new)
    #     if similarity > 0.75:
    #         return True
    #     else:
    #         return False

    def cleanup(self):
        self.mic.close()
        self.tts.close()
