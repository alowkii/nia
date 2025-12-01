import os
import sys
import time
import enum
import pvporcupine
import pyaudio
import struct
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from dotenv import load_dotenv
import whisper
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
from TTS.api import TTS
import numpy as np
import sounddevice as sd
import torch
import threading

# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.chat import AssistantModel
from agent.prompts.feedback import feedback_prompt

# Set logging
from utils.logger import logging
logger = logging.getLogger(__name__)

class State(enum.Enum):
    LISTENING = 1
    WAKE_DETECTED = 2
    COMMAND_MODE = 3


class WakeWordDetector:
    def __init__(self):
        self.state = State.LISTENING

        # Clear the GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Whisper model (local)
        logger.info("Loading Whisper model locally (tiny)...")
        self.whisper_model = whisper.load_model("tiny", device="cuda")

        # TTS model (VITS - fast inference)
        logger.info("Loading VITS TTS model...")
        self._use_gpu = torch.cuda.is_available()
        self.tts = TTS("tts_models/en/ljspeech/vits", gpu=self._use_gpu)
        self.tts_rate = self.tts.synthesizer.tts_config.audio["sample_rate"]
        logger.info(f"VITS TTS ready! (GPU: {self._use_gpu})")

        # Assistant model (Ollama)
        logger.info("Initializing Assistant Model...")
        self.assistant = AssistantModel(model="gemma3:4b")
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

                elif self.state == State.WAKE_DETECTED:
                    self.handle_wake_detected_state()

                elif self.state == State.COMMAND_MODE:
                    self.handle_command_mode_state()

        except KeyboardInterrupt:
            logger.error("\nStopping...")
        finally:
            self.cleanup()

    import threading

    def speak(self, text, blocking=True):
        """Convert text to speech and play it."""
        def _speak():
            try:
                logger.info(f"Speaking: {text}")
                
                # PAUSE audio recording while speaking
                stream_was_active = False
                if self.audio_stream and self.audio_stream.is_active():
                    stream_was_active = True
                    self.audio_stream.stop_stream()
                
                wav = np.array(self.tts.tts(text))
                sd.play(wav, self.tts_rate)
                sd.wait()
                
                # RESUME audio recording after speaking
                if self.audio_stream and stream_was_active:
                    self.audio_stream.start_stream()
                    time.sleep(0.3)  # Small delay before flushing
                    self.flush_audio_buffer()  # Clear any buffered audio
                    
            except Exception as e:
                logger.error(f"TTS error: {e}")
        
        if blocking:
            _speak()
        else:
            threading.Thread(target=_speak, daemon=True).start()

    # --------------------------
    # STATE: LISTENING
    # --------------------------
    def handle_listening_state(self):
        pcm = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
        pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)

        keyword_index = self.porcupine.process(pcm)

        if keyword_index >= 0:
            logger.info("Wake word detected!")
            self.state = State.WAKE_DETECTED

    # --------------------------
    # STATE: WAKE DETECTED
    # --------------------------
    def handle_wake_detected_state(self):
        time.sleep(1.0)
        self.flush_audio_buffer()

        logger.info("Entering command mode...")
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
        # TODO: Creating too much latency
        # if not self.valid_command_from_your_voice():
        #     # Terminate if the voice is not recognized.
        #     logger.info("Person Invalid! Returning to wake-word listening mode.")
        #     self.state = State.LISTENING

        # First interaction after wake word
        response = self.assistant.chat("Hey Nia!")
        self.speak(response.get("text_reply"), blocking=True)
        
        time.sleep(0.5)             # Small delay after speech completes
        self.flush_audio_buffer()   # Clear buffer

        start_time = time.time()
        while time.time() - start_time < 60:  # 1 minute

            logger.info("Listening for voice command...")

            audio = self.record_audio()

            # Save to file
            wavfile.write("temp/command.wav", 16000, audio)

            # Transcribe using Whisper
            logger.info("Transcribing with Whisper...")
            result = self.whisper_model.transcribe("temp/command.wav",
                                                    fp16=True,  # Use half precision
                                                    language="en",  # Skip language detection
                                                    beam_size=1,  # Faster but slightly less accurate
                                                    best_of=1
                                                )

            text = result["text"].strip()
            logger.info(f"Recognized command: {text}")

            if text:  # Only if we got actual text
                logger.info(f"Sending to assistant: {text}")
                response = self.assistant.chat(text)
                action_type = response["action_type"]
                result_txt = response["text_reply"]
                if action_type == "feedback":
                    response = self.assistant.chat(feedback_prompt.format(response_text=result_txt))
                logger.info(f"Assistant JSON response: {response}")
                response = response.get("text_reply")
                logger.info(f"Assistant response: {response}")
                self.speak(response, blocking=True)
            
            # Reset the timeout after each interaction
            start_time = time.time()

        # After command, return to listening
        logger.info("Returning to wake-word listening mode.")
        self.state = State.LISTENING

    # --------------------------
    # AUDIO RECORDING
    # --------------------------
    def record_audio(self, sample_rate=16000):
        """
        Improved voice recording with:
        - auto-adjusted silence threshold
        - reliable speech detection
        - max duration fallback
        """

        logger.info("Speak now...")

        CHUNK = 1024
        MAX_DURATION = 10  # seconds max record
        MIN_SPEECH_DURATION = 0.3
        SILENCE_DURATION = 1.0

        # TEMP: measure ambient noise for first 0.3 seconds
        logger.info("Calibrating noise floor...")
        noise_samples = []

        with sd.InputStream(samplerate=sample_rate, channels=1) as stream:
            start_time = time.time()
            while time.time() - start_time < 0.1:
                frame, _ = stream.read(CHUNK)
                frame = frame.flatten()
                noise_samples.append(frame)

            noise = np.concatenate(noise_samples)
            noise_rms = np.sqrt(np.mean(noise ** 2))
            silence_threshold = max(noise_rms * 2.5, 0.0008)

            logger.info(f"Noise RMS={noise_rms:.6f}, using silence_threshold={silence_threshold:.6f}")

            # Begin actual recording
            frames = []
            silence_start = None
            record_start = time.time()
            speech_started = False

            while True:
                frame, _ = stream.read(CHUNK)
                frame = frame.flatten()
                frames.append(frame)

                rms = np.sqrt(np.mean(frame ** 2))

                # Detect speech start
                if rms > silence_threshold:
                    speech_started = True
                    silence_start = None

                # If speech already started, detect end
                if speech_started:
                    if rms < silence_threshold:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start >= SILENCE_DURATION:
                            logger.info("Silence detected, stopping.")
                            break

                # Max recording safety
                if time.time() - record_start > MAX_DURATION:
                    logger.info("Max recording time reached. Stopping.")
                    break

            audio = np.concatenate(frames)

        return audio.astype(np.float32)

    # --------------------------
    # UTILITIES
    # --------------------------
    def flush_audio_buffer(self):
        """Flush audio buffer only if stream is active"""
        if self.audio_stream and self.audio_stream.is_active():
            for _ in range(5):
                try:
                    self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                except OSError as e:
                    logger.warning(f"Error flushing buffer: {e}")
                    break

    def cleanup(self):
        if self.audio_stream:
            self.audio_stream.close()
        if self.pa:
            self.pa.terminate()
        if self.porcupine:
            self.porcupine.delete()
