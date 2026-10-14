"""What differs between Windows and Linux, in one place: how a program is named, started hidden or apart from NIA,
and which Python runs a console script. Everything else asks here instead of checking the OS itself."""
import os
import subprocess
import sys
from pathlib import Path

WINDOWS = os.name == "nt"
EXE = ".exe" if WINDOWS else ""  # llama-server.exe, claude.exe - no suffix on Linux
SHELL = "Windows cmd.exe" if WINDOWS else "Linux bash"

# Started with no console window of its own (Windows); on Linux nothing opens one anyway
HIDDEN = {"creationflags": subprocess.CREATE_NO_WINDOW} if WINDOWS else {}
# Started in a console of its own (Windows), or in its own session on Linux - either way, not ended with ours
NEW_CONSOLE = {"creationflags": subprocess.CREATE_NEW_CONSOLE} if WINDOWS else {"start_new_session": True}
# An app opened for the user, which must outlive NIA and never write into her console
DETACHED = (HIDDEN if WINDOWS else {"start_new_session": True}) | {
    "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}


def console_python():
    """The Python that runs a script with a console - pythonw (Windows) has no stdout for a child to log to"""
    return str(Path(sys.executable).with_name("python.exe")) if WINDOWS else sys.executable
