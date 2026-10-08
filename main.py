"""The assistant on its own: mic, speech and agent, with no window. nia.py runs this hidden behind NIA's window;
run it directly (python main.py) to see its log in the console."""
from voice_assistant.voice_assistant import WakeWordDetector, logger

if __name__ == "__main__":
    try:
        WakeWordDetector().run()
    except Exception:
        logger.exception("NIA crashed")
        raise SystemExit(1)
