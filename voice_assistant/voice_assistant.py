import os
import sys
import time
import enum
import queue
import pvporcupine
import pyaudio
import struct
from dotenv import load_dotenv
from moonshine_voice import MicTranscriber, TextToSpeech

# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.chat import AssistantModel, ensure_llm_server

# Set logging
from utils.logger import logging
logger = logging.getLogger(__name__)

# ponytail: Kokoro takes ~1 s to start speaking on CPU; "piper_en_US-lessac-medium" takes ~0.2 s but sounds more robotic
VOICE = "kokoro_af_heart"
GREETING = "Yes, sir?"
SESSION_TIMEOUT = 60  # seconds of no exchange before going back to wake-word listening


class State(enum.Enum):
    LISTENING = 1
    COMMAND_MODE = 2


class WakeWordDetector:
    def __init__(self):
        self.state = State.LISTENING

        # Speech to text (Moonshine medium streaming, CPU). Its own mic stream and VAD
        # end each utterance; completed lines land in this queue.
        logger.info("Loading Moonshine speech-to-text...")
        self.lines = queue.Queue()
        self.mic = MicTranscriber().on_line(lambda line: self.lines.put(line.text)).load()

        # Text to speech (CPU)
        logger.info(f"Loading text-to-speech ({VOICE})...")
        self.tts = TextToSpeech().language("en_us").voice(VOICE).load()

        # Assistant model (Bonsai on llama-server, GPU)
        logger.info("Starting the LLM server if needed...")
        ensure_llm_server()
        self.assistant = AssistantModel()
        logger.info("Assistant ready!")

        # Porcupine access key
        self.access_key = os.getenv('PICOVOICE_ACCESS_KEY')
        if not self.access_key:
            raise ValueError("PICOVOICE_ACCESS_KEY not found in environment variables")

        # Porcupine init
        self.porcupine = pvporcupine.create(
            access_key=self.access_key,
            keyword_paths=["wake_word/Hey-Nia_en_windows_v3_0_0.ppn"],
            sensitivities=[0.5]
        )

        # Audio init
        self.pa = pyaudio.PyAudio()
        self.audio_stream = self.pa.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )

        logger.info(f"Porcupine version: {self.porcupine.version}")
        logger.info(f"Frame length: {self.porcupine.frame_length}")
        logger.info(f"Sample rate: {self.porcupine.sample_rate}Hz")

    def run(self):
        logger.info("State Machine started. Listening for wake words...")

        try:
            while True:
                if self.state == State.LISTENING:
                    self.handle_listening_state()

                elif self.state == State.COMMAND_MODE:
                    self.handle_command_mode_state()

        except KeyboardInterrupt:
            logger.error("\nStopping...")
        finally:
            self.cleanup()

    def speak(self, text):
        """Speak text, with the mic muted so NIA doesn't transcribe herself."""
        logger.info(f"Speaking: {text}")
        self.mic.mute(True)
        try:
            self.tts.say(text)
            self.tts.wait()
        except Exception as e:
            logger.error(f"TTS error: {e}")
        finally:
            self.mic.mute(False)

    # --------------------------
    # STATE: LISTENING
    # --------------------------
    def handle_listening_state(self):
        pcm = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
        pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)

        keyword_index = self.porcupine.process(pcm)

        if keyword_index >= 0:
            logger.info("Wake word detected!")
            self.state = State.COMMAND_MODE

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

    # --------------------------
    # STATE: COMMAND MODE
    # --------------------------
    def handle_command_mode_state(self):
        # Porcupine pauses while Moonshine owns the conversation, so its buffer
        # doesn't fill with stale audio (or NIA's own voice)
        self.audio_stream.stop_stream()

        self.speak(GREETING)
        self.mic.start()
        logger.info("Listening for voice command...")

        deadline = time.time() + SESSION_TIMEOUT
        while (remaining := deadline - time.time()) > 0:
            try:
                text = self.lines.get(timeout=remaining).strip()
            except queue.Empty:
                break
            if not text:
                continue

            logger.info(f"Recognized command: {text}")
            self.mic.mute(True)  # ignore the room while thinking
            reply = self.assistant.respond(text)
            logger.info(f"Assistant response: {reply}")
            if reply:
                self.speak(reply)
            self.mic.mute(False)

            # Only a real exchange keeps the session open - silence must time out
            deadline = time.time() + SESSION_TIMEOUT

        self.mic.stop()
        while not self.lines.empty():  # drop anything said as the session closed
            self.lines.get_nowait()

        logger.info("Returning to wake-word listening mode.")
        self.audio_stream.start_stream()
        self.state = State.LISTENING

    def cleanup(self):
        self.mic.close()
        self.tts.close()
        if self.audio_stream:
            self.audio_stream.close()
        if self.pa:
            self.pa.terminate()
        if self.porcupine:
            self.porcupine.delete()
