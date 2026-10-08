"""Which PC commands NIA may run without asking. Three layers, cheapest first:

1. Risks (code): anything that deletes, closes programs, shuts down, installs, formats, touches the
   registry, presses keys in another window or runs as administrator always asks. No model can
   override this.
2. Read-only allowlist (code, instant): one known read command, piped only into filters like
   findstr, with nothing chained or redirected, runs.
3. The decision model (OpenThai-SystemOne on Ollama, CPU): six narrow hazard questions; if every one
   is below the threshold it runs, otherwise NIA asks. If Ollama doesn't answer, she asks.

New folders and new files in your own folders don't ask; overwriting, editing and deleting always do (see
creatable). See eval_approval.py for how the model layer was chosen.
"""
import json
import logging
import os
import re
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# Said out loud whatever the stated intent, so a loose description can't hide a dangerous command
RISKS = [
    (r"\b(del|erase|rm|rmdir|rd|remove-item)\b", "it deletes files"),
    (r"\b(format|diskpart|clear-disk)\b", "it formats or wipes a disk"),
    (r"\b(shutdown|restart-computer|stop-computer|logoff)\b", "it shuts down or restarts the PC"),
    (r"\b(taskkill|stop-process|kill)\b", "it closes programs"),
    (r"\b(reg|regedit|set-itemproperty|new-itemproperty)\b|hk(lm|cu):", "it changes the registry"),
    (r"\b(invoke-webrequest|iwr|curl|wget|bitsadmin)\b.*\|\s*(iex|invoke-expression)", "it downloads and runs code"),
    # She has web_search and read_page for the web; a shell command fetching from it ran unasked once (curl, 0.03)
    (r"\b(invoke-webrequest|iwr|invoke-restmethod|irm|curl|wget|bitsadmin|certutil|start-bitstransfer)\b",
     "it reaches the internet"),
    # ask_claude runs Claude Code safely; from the shell it could be given any tools at all
    (r"\bclaude(\.exe)?\b", "it runs Claude Code"),
    (r"\b(winget|choco|msiexec|scoop)\b|\b(pip|npm|pnpm|yarn)\s+(install|uninstall|add|remove)\b", "it installs or removes software"),
    (r"sendkeys|keybd_event|sendinput", "it presses keys in another window"),
    (r"-verb\s+runas|\brunas\b", "it runs something as administrator"),
]


def risks(action):
    command = str(action["args"].get("command", "")) if action["name"] == "execute" else ""
    return [note for pattern, note in RISKS if re.search(pattern, command, re.I)]


# Layer 2: commands that can only read. Anything chained, redirected or escaped is out
READ_ONLY = {"tasklist", "dir", "ipconfig", "systeminfo", "type", "more", "where", "whoami", "hostname", "ver",
             "netstat", "nslookup", "ping", "tree", "getmac", "vol", "findstr", "find"}
FILTERS = {"findstr", "find", "sort", "more"}
GIT_READ = {"status", "log", "diff", "show"}
CMD_UNSAFE = re.compile(r"[&;<>^`]|\|\|")
POWERSHELL = re.compile(r'powershell(?:\.exe)?\s+(?:-NoProfile\s+)?-(?:Command|c)\s+"(.+)"', re.I | re.S)
PS_UNSAFE = re.compile(r"[;&<>`]|\$\(|@\(|-ComObject|\binvoke|\biex\b|\bstart-|\[", re.I)
PS_READ = re.compile(r"(get|test|select|where|sort|format|measure|group|resolve)-\w+|out-string", re.I)


def read_only(command):
    """True only for a command that can do nothing but read and print"""
    command = command.strip()
    if ps := POWERSHELL.fullmatch(command):
        inner = ps.group(1)
        stages = [stage.strip() for stage in inner.split("|")]
        return not PS_UNSAFE.search(inner) and all(s and PS_READ.fullmatch(s.split()[0]) for s in stages)
    if CMD_UNSAFE.search(command):
        return False
    stages = [stage.split() for stage in command.split("|")]
    if not all(stages):
        return False
    first = stages[0][0].lower().removesuffix(".exe")
    if first == "git":
        words = stages[0][1:]
        if words[:1] == ["-C"]:
            words = words[2:]
        if not words or words[0] not in GIT_READ:
            return False
    elif first not in READ_ONLY:
        return False
    return all(stage[0].lower().removesuffix(".exe") in FILTERS for stage in stages[1:])


# Layer 3: one narrow yes/no per hazard, with examples of what does NOT count - this setup separated
# harmless from risky commands best (eval_approval.py); broad "is it read-only?" questions did not
READING = "only reads, lists, searches or prints information (such as dir, type, findstr, tasklist, Get-Process)"


def _noul(question, yes, no):
    return {"type": "noul", "instructions": question, "criteria": {"true": yes, "false": no}}


HAZARDS = {
    "writes_files": _noul("Does running `command` create, write, overwrite, rename, copy or move a file or folder?",
                          "It changes the files on disk, e.g. echo x > f.txt, copy, move, mkdir, Set-Content",
                          f"Files stay exactly as they were: the command {READING}"),
    "deletes": _noul("Does running `command` delete or erase anything?",
                     "It removes files, folders, data or settings, e.g. del, rm, Remove-Item, rmdir",
                     f"Nothing is deleted: the command {READING}"),
    "opens_or_starts": _noul("Does running `command` open a window or launch an app, website, file or process?",
                             "It opens or starts something visible or running, e.g. start, notepad, explorer, "
                             "Start-Process, a URL",
                             f"Nothing opens or starts: the command {READING} in the console"),
    "stops_programs": _noul("Does running `command` close, stop, kill, restart or shut down a program, service or the "
                            "computer?",
                            "It ends or restarts something, e.g. taskkill, Stop-Process, shutdown, Restart-Service",
                            f"Everything keeps running as before: the command {READING}"),
    "installs": _noul("Does running `command` install, update, uninstall or download software or files?",
                      "e.g. winget install, pip install, choco, Invoke-WebRequest, curl -o",
                      f"Nothing is installed or downloaded: the command {READING}"),
    "runs_code": _noul("Does `command` run a script or program file, or code passed as text, that could do anything?",
                       "It runs e.g. python x.py, node app.js, a .bat or .ps1 file, or powershell code that acts on "
                       "the system",
                       f"It only uses built-in commands that look things up: the command {READING}, or finds where "
                       "a program is"),
}


def hazards(command, model, timeout=30):
    """{hazard: probability} from the decision model, or None if it can't be reached"""
    body = {"model": model, "state": {"command": command, "shell": "Windows cmd.exe"}, "questions": HAZARDS}
    request = urllib.request.Request("http://127.0.0.1:11434/v1/systemone", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    try:
        answers = json.load(urllib.request.urlopen(request, timeout=timeout))["answers"]
        return {name: answers[name]["noul"] for name in HAZARDS}
    except (OSError, KeyError, ValueError) as e:
        logger.warning(f"Decision model unavailable ({e!r}) - asking instead")
        return None


# Making a new folder or a new file in your own folders is too small to ask about ("I'll run mkdir. Should I go
# ahead?", "I'll save coin_flip.py?" - each needed a yes). Overwriting, editing and deleting still ask, and so does
# anything in system folders, AppData (the Startup folder runs whatever lands in it), hidden dot-folders, or NIA's
# own code - changes to herself go through a reviewed branch
NIA_ROOT = Path(__file__).resolve().parent.parent
OFF_LIMITS = re.compile(r"\\(windows|program files( \(x86\))?|programdata|appdata|\$recycle\.bin|"
                        r"system volume information)(\\|$)|\\\.", re.I)
MKDIR = re.compile(r'\s*(?:mkdir|md)\s+("[^"]+"|[^\s"&|<>;^]+)(?:\s+2>&1)?(?:\s*&&\s*echo\s+[\w .-]*)?\s*', re.I)


def windows_path(path):
    """A file tool's path (/c/Users/x, or relative to the home folder) or a Windows one, as a full Windows path"""
    text = str(path).strip().strip('"')
    if drive := re.match(r"^/([a-zA-Z])(?:/|$)(.*)", text):
        return Path(os.path.abspath(f"{drive.group(1).upper()}:\\{drive.group(2)}"))
    if re.match(r"^[a-zA-Z]:[\\/]", text):
        return Path(os.path.abspath(text))
    return Path(os.path.abspath(Path.home() / text.lstrip("/\\")))


def creatable(path):
    """Whether a new folder or file may appear at path without asking: your own folders, or another drive -
    never the system, AppData, a dot-folder or NIA's own code"""
    target = windows_path(path)
    if OFF_LIMITS.search(str(target)):
        return False
    for own in (NIA_ROOT, NIA_ROOT.parent / "nia-changes"):
        if target == own or own in target.parents:
            return False
    on_system_drive = target.drive.upper() == Path.home().drive.upper()
    return not on_system_drive or Path.home() in target.parents


def decide(action, model, threshold):
    """("run" or "ask", why) for one action awaiting approval"""
    if action["name"] == "write_file":
        path = action["args"].get("file_path", "")
        if creatable(path) and not windows_path(path).exists():
            return "run", "a new file in your folders"
        return "ask", "it overwrites a file, or it's outside your folders"
    if action["name"] != "execute":
        return "ask", "edits, deletions and changes to NIA always ask"
    command = str(action["args"].get("command", ""))
    if notes := risks(action):
        return "ask", "; ".join(notes)
    if read_only(command):
        return "run", "read-only command"
    if (folder := MKDIR.fullmatch(command)) and creatable(folder.group(1)):
        return "run", "a new folder in your folders"
    if not model:
        return "ask", "no decision model set"
    scores = hazards(command, model)
    if scores is None:
        return "ask", "decision model unavailable"
    worst = max(scores, key=scores.get)
    summary = ", ".join(f"{name} {p:.2f}" for name, p in scores.items())
    verdict = "run" if scores[worst] < threshold else "ask"
    return verdict, f"model: worst {worst} {scores[worst]:.2f} vs threshold {threshold} ({summary})"
