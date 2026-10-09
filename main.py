"""The assistant on its own: mic, speech and agent, with no window. nia.py runs this hidden behind NIA's window;
run it directly (python main.py) to see its log in the console."""
import logging
import os

from voice_assistant.voice_assistant import WakeWordDetector, logger

if __name__ == "__main__":
    code = 0
    try:
        WakeWordDetector().run()
    except Exception:
        logger.exception("NIA crashed")
        code = 1
    logging.shutdown()
    os._exit(code)  # now, not when every library thread is done: one kept her running 10+ minutes after NIA closed
