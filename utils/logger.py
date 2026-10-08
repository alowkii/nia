import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Absolute, so the log lands in the same place whatever the working directory
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# One file per kind of process: the tests' fake turns must never land in NIA's real log, and the window (nia.py)
# runs alongside the assistant - two processes rotating one file fails on Windows
_script = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else ""
LOG_FILE = LOG_DIR / ("window.log" if _script == "nia" else "test.log" if _script.startswith("test_") else "nia.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")],
    force=True
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# One line per LLM request is noise; the agent logs each turn itself
for name in ("httpx", "httpx2"):  # the openai client logs through httpx2
    logging.getLogger(name).setLevel(logging.WARNING)


# Crashes go to the log too, not just to a console window that may already be gone
def _log_crash(exc_type, exc, tb):
    if issubclass(exc_type, KeyboardInterrupt):
        return sys.__excepthook__(exc_type, exc, tb)
    logging.getLogger("crash").critical("Uncaught exception", exc_info=(exc_type, exc, tb))


sys.excepthook = _log_crash
threading.excepthook = lambda args: _log_crash(args.exc_type, args.exc_value, args.exc_traceback)
