r"""Opening the PC's installed apps the way Windows itself finds them - never by guessing folders. VLC sat in
C:\Program Files (x86)\VideoLAN\VLC, yet NIA tried "...\VLC media player" (its Start-menu name, not its folder),
`where vlc.exe` (which only searches PATH) and nine more guesses, and ran out of steps. Windows keeps both answers:
the App Paths registry key that the Run box uses, and the Start menu's shortcuts."""
import os
from pathlib import Path

from langchain_core.tools import tool

from .approval import windows_path

START_MENUS = [Path(os.getenv("ProgramData", r"C:\ProgramData")) / "Microsoft" / "Windows" / "Start Menu" / "Programs",
               Path(os.getenv("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"]


def registered(name):
    """The program Windows has registered under name (vlc -> ...\VLC\vlc.exe) - what the Run box uses - or None"""
    import winreg
    key = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{name}.exe"
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, key) as k:
                program = Path(os.path.expandvars(winreg.QueryValueEx(k, "")[0].strip('"')))
        except OSError:
            continue
        if program.is_file():
            return program
    return None


def in_start_menu(name):
    """The program behind the Start-menu shortcut that best matches name ("VLC" -> "VLC media player"), or None"""
    import pythoncom
    from win32com.shell import shell
    wanted = name.lower()
    shortcuts = [lnk for menu in START_MENUS if menu.is_dir()
                 for lnk in menu.rglob("*.lnk") if wanted in lnk.stem.lower()]
    for lnk in sorted(shortcuts, key=lambda lnk: len(lnk.stem)):  # "VLC media player" before "...reset preferences"
        pythoncom.CoInitialize()
        link = pythoncom.CoCreateInstance(shell.CLSID_ShellLink, None, pythoncom.CLSCTX_INPROC_SERVER,
                                          shell.IID_IShellLink)
        link.QueryInterface(pythoncom.IID_IPersistFile).Load(str(lnk))
        program = Path(link.GetPath(0)[0])
        if program.suffix.lower() == ".exe" and program.is_file():
            return program
    return None


def find_app(name):
    """An installed app's program: registered first, then the Start menu"""
    name = name.strip().removesuffix(".exe")
    words = name.lower().split()
    return (registered(name) or (registered(words[0]) if words else None)) or in_start_menu(name)


@tool
def open_app(app: str, file: str = "") -> str:
    """Open an installed app - "open VLC", "open Notepad" - optionally with a file to open in it: "play this movie
    in VLC". app is its name as the user says it. file is the file's full path (a file-tool path like /d/x.mp4 is
    fine). No approval needed. This finds apps the way Windows does - never search folders for an app's program"""
    program = find_app(app)
    if program is None:
        return f"{app} doesn't seem to be installed: it isn't in the Start menu or among Windows' registered apps"
    path = windows_path(file) if file else None
    if path and not path.is_file():
        return f"Not opened: there's no file at {path}"
    try:
        os.startfile(program, arguments=f'"{path}"' if path else "")
    except OSError as e:  # e.g. a Store app Windows won't start from its program file
        return f"Found {program.stem} at {program}, but Windows wouldn't start it ({e.strerror or e})"
    return f"Opened {path.name} in {program.stem}" if path else f"Opened {program.stem}"


TOOLS = [open_app]
