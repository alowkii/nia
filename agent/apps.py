r"""Opening the PC's installed apps the way the system itself finds them - never by guessing folders. VLC sat in
C:\Program Files (x86)\VideoLAN\VLC, yet NIA tried "...\VLC media player" (its Start-menu name, not its folder),
`where vlc.exe` (which only searches PATH) and nine more guesses, and ran out of steps. Windows keeps both answers:
the App Paths registry key that the Run box uses, and the Start menu's shortcuts. Linux keeps them in the desktop
entries (.desktop files) that app launchers list, and on PATH."""
import configparser
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from langchain_core.tools import tool

from utils.system import DETACHED, WINDOWS
from .approval import pc_path

START_MENUS = [Path(os.getenv("ProgramData", r"C:\ProgramData")) / "Microsoft" / "Windows" / "Start Menu" / "Programs",
               Path(os.getenv("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"]


def registered(name):
    r"""The program Windows has registered under name (vlc -> ...\VLC\vlc.exe) - what the Run box uses - or None"""
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


# Linux: where desktop entries live, most personal first - an entry in your own folder overrides the system's
def application_dirs():
    data_home = Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    data_dirs = (os.getenv("XDG_DATA_DIRS") or "/usr/local/share:/usr/share").split(":")
    flatpaks = [data_home / "flatpak" / "exports" / "share", Path("/var/lib/flatpak/exports/share")]
    return [Path(d) / "applications" for d in [data_home, *data_dirs, *flatpaks] if d]


def desktop_entries():
    """Each app the launcher lists, as {"file", "name", "exec", "terminal"}; the first entry of each file name wins"""
    entries, seen = [], set()
    for folder in application_dirs():
        if not folder.is_dir():
            continue
        for file in sorted(folder.rglob("*.desktop")):
            if file.name in seen:
                continue
            seen.add(file.name)
            parser = configparser.ConfigParser(interpolation=None, strict=False)
            try:
                parser.read(file, encoding="utf-8")
                entry = parser["Desktop Entry"]
            except (configparser.Error, KeyError, UnicodeDecodeError):
                continue
            if entry.get("Type") != "Application" or entry.get("Hidden") == "true" or not entry.get("Exec") \
                    or entry.get("NoDisplay") == "true":
                continue
            entries.append({"file": file, "name": entry.get("Name", file.stem), "exec": entry["Exec"],
                            "terminal": entry.get("Terminal") == "true"})
    return entries


def entry_program(entry):
    """The program a desktop entry runs, as a full path - /usr/bin/vlc for "vlc --started-from-file %U" - or None"""
    try:
        words = shlex.split(entry["exec"])
    except ValueError:
        return None
    if words[:1] == ["env"]:  # env VAR=x program ...
        words = [w for w in words[1:] if "=" not in w]
    found = shutil.which(words[0]) if words else None
    return Path(found).resolve() if found else None


def in_desktop_entries(name):
    """The desktop entry that best matches name ("VLC" -> VLC media player, "code" -> Visual Studio Code), or None"""
    wanted = name.lower()
    entries = desktop_entries()

    def program_name(e):
        program = entry_program(e)
        return program.name.lower() if program else ""
    for matches in (lambda e: e["name"].lower() == wanted, lambda e: e["file"].stem.lower().split(".")[-1] == wanted,
                    lambda e: program_name(e) == wanted, lambda e: wanted in e["name"].lower()):
        found = [e for e in entries if matches(e)]
        if found:
            return min(found, key=lambda e: len(e["name"]))["file"]  # "VLC media player" before longer ones
    return None


def find_app(name):
    """An installed app: Windows - its program, registered first, then the Start menu; Linux - its desktop entry,
    then a program of that name on PATH"""
    name = name.strip().removesuffix(".exe")
    words = name.lower().split()
    if not WINDOWS:
        program = shutil.which(words[0]) if len(words) == 1 else None
        return in_desktop_entries(name) or (Path(program) if program else None)
    return (registered(name) or (registered(words[0]) if words else None)) or in_start_menu(name)


def app_name(program):
    """What to call an app found by find_app: its launcher name on Linux (VLC media player), else its program's"""
    if program.suffix == ".desktop":
        return next((e["name"] for e in desktop_entries() if e["file"] == program), program.stem)
    return program.stem


def command_line(program, path=None):
    """How Linux starts an app, with a file: a desktop entry's Exec line, its field codes filled in (%f, %U: the file;
    %c: its name; the rest dropped) - in a terminal for a terminal app (btop) - or a program on PATH with the file"""
    if program.suffix != ".desktop":
        return [str(program)] + ([str(path)] if path else [])
    entry = next(e for e in desktop_entries() if e["file"] == program)
    words, given = [], False
    for word in shlex.split(entry["exec"]):
        if code := re.search(r"%[fFuU]", word):  # "%U" on its own, or inside a word: Spotify's "--uri=%u"
            given = True
            if path:
                words.append(word.replace(code.group(0), str(path)))
            continue
        word = re.sub(r"%[ikdDnNvm]", "", word.replace("%c", entry["name"]).replace("%%", "\0")).replace("\0", "%")
        if word:
            words.append(word)
    words += [str(path)] if path and not given else []
    terminal = shutil.which("xdg-terminal-exec")
    return [terminal, *words] if entry["terminal"] and terminal else words


def launch(program, path=None):
    """Start an app (with a file), apart from NIA - it stays open when she stops, and never holds her turn"""
    if WINDOWS:
        os.startfile(program, arguments=f'"{path}"' if path else "")
    else:
        subprocess.Popen(command_line(program, path), cwd=Path.home(), **DETACHED)


@tool
def open_app(app: str, file: str = "") -> str:
    """Open an installed app - "open VLC", "open Notepad" - or play a file in one: "play this movie in VLC", "open
    the song in Groove". app is its name as the user says it; file is the file's full path (a file-tool path like
    /d/x.mp4 is fine) - find the file first, then call this once. No approval needed. It finds the app the way
    the system does, so never search folders or run `where` or `which` for an app's program"""
    program = find_app(app)
    if program is None:
        return f"{app} doesn't seem to be installed: " + (
            "it isn't in the Start menu or among Windows' registered apps" if WINDOWS else
            "it isn't among the apps in the launcher, nor a program on PATH")
    path = pc_path(file) if file else None
    if path and not path.is_file():
        return f"Not opened: there's no file at {path}"
    try:
        launch(program, path)
    except OSError as e:  # e.g. a Store app Windows won't start from its program file
        return f"Found {app_name(program)} at {program}, but it wouldn't start ({e.strerror or e})"
    return f"Opened {path.name} in {app_name(program)}" if path else f"Opened {app_name(program)}"


def open_with_default(path):
    """A file in whatever app the PC opens that kind of file with - for a video when VLC isn't installed"""
    if WINDOWS:
        os.startfile(path)
    else:
        subprocess.Popen(["xdg-open", str(path)], **DETACHED)


# "Pick one for tonight" sometimes started the movie, and "play the movie" then opened a second copy
REOPEN_AFTER = 120  # seconds
last_video = {"path": None, "when": 0.0}


@tool
def play_video(file: str, player: str = "VLC") -> str:
    """Play a movie or video file from the PC - "play a movie", "play the movie", "watch The Founder" - in a video
    player: VLC unless the user names another (the PC's own video player if VLC isn't installed). file is the video's
    full path (a file-tool path like /d/x.mp4 is fine) - find it first, e.g. with glob. No approval needed. For
    YouTube use play_youtube; for music, Spotify"""
    path = pc_path(file)
    if path == last_video["path"] and time.time() - last_video["when"] < REOPEN_AFTER:
        return f"{path.name} is already playing - opened a moment ago. Don't open it again"
    if player == "VLC" and path.is_file() and find_app(player) is None:  # VLC was only the default, not asked for
        try:
            open_with_default(path)
        except OSError as e:
            return f"VLC isn't installed, and the PC's own video player wouldn't start ({e.strerror or e})"
        result = f"Opened {path.name} in the PC's video player (VLC isn't installed)"
    else:
        result = open_app.invoke({"app": player, "file": file})
    if result.startswith("Opened"):
        last_video.update(path=path, when=time.time())
    return result


# play_video by its own name: "play a movie" didn't lead Bonsai to open_app - "there's no video player tool available
# to me" - the way "play on YouTube" leads to play_youtube
TOOLS = [play_video, open_app]
