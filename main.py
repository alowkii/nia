import os
import sys

from voice_assistant.voice_assistant import WakeWordDetector, logger

if __name__ == "__main__":
    try:
        detector = WakeWordDetector()
        detector.run()
    except Exception:
        logger.exception("NIA crashed")
        if os.getenv("NIA_PAUSE_ON_CRASH") and sys.stdin:  # set by ui.py, so its console window stays open to read
            input("\nNIA crashed - the error is above and in logs/nia.log. Press Enter to close.")
        raise SystemExit(1)
