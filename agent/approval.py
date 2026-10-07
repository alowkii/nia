"""Which PC commands NIA may run without asking. Three layers, cheapest first:

1. Risks (code): anything that deletes, closes programs, shuts down, installs, formats, touches the
   registry, presses keys in another window or runs as administrator always asks. No model can
   override this.
2. Read-only allowlist (code, instant): one known read command, piped only into filters like
   findstr, with nothing chained or redirected, runs.
3. The decision model (OpenThai-SystemOne on Ollama, CPU): six narrow hazard questions; if every one
   is below the threshold it runs, otherwise NIA asks. If Ollama doesn't answer, she asks.

File writes, edits and deletes always ask. See eval_approval.py for how the model layer was chosen.
"""
import json
import logging
import re
import urllib.request

logger = logging.getLogger(__name__)

# Said out loud whatever the stated intent, so a loose description can't hide a dangerous command
RISKS = [
    (r"\b(del|erase|rm|rmdir|rd|remove-item)\b", "it deletes files"),
    (r"\b(format|diskpart|clear-disk)\b", "it formats or wipes a disk"),
    (r"\b(shutdown|restart-computer|stop-computer|logoff)\b", "it shuts down or restarts the PC"),
    (r"\b(taskkill|stop-process|kill)\b", "it closes programs"),
    (r"\b(reg|regedit|set-itemproperty|new-itemproperty)\b|hk(lm|cu):", "it changes the registry"),
    (r"\b(invoke-webrequest|iwr|curl|wget|bitsadmin)\b.*\|\s*(iex|invoke-expression)", "it downloads and runs code"),
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


def decide(action, model, threshold):
    """("run" or "ask", why) for one action awaiting approval"""
    if action["name"] != "execute":
        return "ask", "file changes always ask"
    command = str(action["args"].get("command", ""))
    if notes := risks(action):
        return "ask", "; ".join(notes)
    if read_only(command):
        return "run", "read-only command"
    if not model:
        return "ask", "no decision model set"
    scores = hazards(command, model)
    if scores is None:
        return "ask", "decision model unavailable"
    worst = max(scores, key=scores.get)
    summary = ", ".join(f"{name} {p:.2f}" for name, p in scores.items())
    verdict = "run" if scores[worst] < threshold else "ask"
    return verdict, f"model: worst {worst} {scores[worst]:.2f} vs threshold {threshold} ({summary})"
