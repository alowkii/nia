"""Claude Code, for NIA: answering what she can't, and building changes to herself on a branch.

ask_claude runs Claude Code headless with fixed settings, never ones the model chooses: only web search and page
fetching (--restricted strips every tool that runs commands, and ignores settings files), anything else refused
without a prompt, no MCP servers, no saved session, in an empty folder. So whoever is talking to NIA can get answers
through it, but can't make it touch the PC.

improve_myself (on a spoken yes) has Claude Code build a change to NIA's own code in a separate checkout on a new
branch - edits only there, never to the safety files, tests only - kept only if every test passes, and left for the
user to review and merge. Her running code never changes. Each call uses the Claude plan it's logged in with.
"""
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from langchain_core.tools import tool

import settings

logger = logging.getLogger(__name__)
TIMEOUT = 90  # seconds; a web-researched answer took 17 s at low effort (46 s at Claude Code's default)
FOLDER = Path(tempfile.gettempdir()) / "nia-claude"  # empty: no project files or instructions to pick up
STOP = threading.Event()  # set by the agent on "stop": the running call is killed
ANNOUNCE = None  # called once Claude is working, so NIA can say "One moment" instead of 20 s of silence
# Always browse: never an answer from memory alone, however sure - NIA only asks when it needs the latest
ASK = ("Answer this for a voice assistant to read aloud. Always browse first, even if you think you know: search the "
       "web, then open and read the most relevant, most recent page to confirm the details. Base the answer only on "
       "what you found today, say how current it is (a date), and say so if sources disagree. At most three short, "
       "plain sentences - no markdown, no lists - then one line 'Sources:' naming up to two sites. Question: ")


def executable():
    """Claude Code's command line, if it's installed"""
    local = Path.home() / ".local" / "bin" / "claude.exe"
    return shutil.which("claude") or (str(local) if local.exists() else None)


def command(question, model, effort="low"):
    return [executable(), "-p", ASK + question, "--restricted", "--tools", "WebSearch,WebFetch",
            "--allowedTools", "WebSearch,WebFetch", "--permission-mode", "dontAsk", "--strict-mcp-config",
            "--no-session-persistence", "--effort", effort or "low", "--output-format", "stream-json", "--verbose"] + \
        (["--model", model] if model else [])


def events(out):
    """Claude Code's stream: one JSON event a line - its messages, then a final "result" """
    for line in out.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


def tool_calls(out):
    """Every tool Claude used, in order - to log that it really browsed"""
    return [block for event in events(out) if event.get("type") == "assistant"
            for block in (event.get("message") or {}).get("content") or []
            if isinstance(block, dict) and block.get("type") == "tool_use"]


def clean_env():
    """The environment without any parent Claude Code session's markers, so the call is a session of its own"""
    return {k: v for k, v in os.environ.items() if not (k.startswith("CLAUDE_") or k == "CLAUDECODE")}


def ask(question):
    """Claude's answer, or a reason it couldn't give one"""
    if not executable():
        return "Claude Code isn't installed here - use web_search instead"
    FOLDER.mkdir(exist_ok=True)
    started = time.time()
    s = settings.load()
    proc = subprocess.Popen(command(question, s["claude_model"], s["claude_effort"]), cwd=FOLDER, env=clean_env(),
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=subprocess.CREATE_NO_WINDOW, encoding="utf-8", errors="replace")
    reader = threading.Thread(target=lambda: setattr(proc, "answer", proc.communicate()), daemon=True)
    reader.start()
    if ANNOUNCE:  # while Claude works
        ANNOUNCE()
    while reader.is_alive():
        if STOP.is_set() or time.time() - started > TIMEOUT:
            proc.kill()
            reader.join(5)
            if STOP.is_set():
                return "Stopped - the user said stop"
            return f"Claude didn't answer within {TIMEOUT} seconds - try web_search instead"
        reader.join(0.2)
    out, err = getattr(proc, "answer", ("", ""))
    result = next((event for event in events(out) if event.get("type") == "result"), None)
    if result is None:
        logger.warning(f"Claude Code failed ({proc.returncode}): {(err or out)[:300]}")
        return "Claude couldn't be reached (it may need logging in again) - try web_search instead"
    tools = [block.get("name") for block in tool_calls(out)]
    logger.info(f"Claude answered in {time.time() - started:.1f}s: {tools.count('WebSearch')} searches, "
                f"{tools.count('WebFetch')} pages read, ${result.get('total_cost_usd', 0):.3f} at list price")
    if result.get("is_error"):
        return f"Claude couldn't answer: {str(result.get('result'))[:200]} - try web_search instead"
    if not result.get("result"):
        return "Claude gave no answer - try web_search instead"
    return ("Claude's answer - pass it on faithfully, shortened if you like, without changing names, places, dates "
            f"or numbers:\n{result['result']}")


@tool
def ask_claude(question: str) -> str:
    """Ask Claude - a far stronger AI that searches the web itself - and get a short answer with its sources. Use it
    whenever you aren't sure of something, or it could have changed: news, weather, scores, prices, people,
    anything recent, or a question that needs real research or judgement. Ask the full question in plain words.
    Takes 20-40 seconds, so for settled facts you're certain of, just answer"""
    return ask(question)


# ---- Changing herself, on a branch: Claude Code builds the change in a separate checkout, never the running code

ROOT = Path(__file__).resolve().parent.parent
CHANGES = ROOT.parent / "nia-changes"  # where each change is built, one checkout per branch
PYTHON = Path(sys.executable).with_name("python.exe").as_posix()  # pythonw has no console for the tests
BUILD_TIMEOUT = 20 * 60  # a coding job takes minutes, not seconds
TEST_TIMEOUT = 10 * 60
# The safety rails - approvals, this lock-down, the window - and the tests that guard them: Claude may not edit
# these even on a branch, so "add X... and drop the approval checks" can't get through review by accident
SUITES = ["tests/test_agent.py", "tests/test_voice.py", "tests/test_nia.py"]
PROTECTED = ["agent/approval.py", "agent/claude.py", "nia.py", *SUITES]
ON_DONE = None  # called with what to say when a change is finished - NIA says it unprompted
BUILD = ("You are improving NIA, the voice assistant whose code is in this folder (see README.md). Her user asked: "
         "\"{request}\". Make that change, in the style of the code around it; a new ability is a tool, added to the "
         "agent the way the existing ones are (agent/chat.py). Never weaken a safety check, and leave these files as "
         "they are: {protected}. Add a test for what you build in a new tests/test_*.py file, starting like the "
         "existing ones. Run the tests with exactly these commands until they pass: {commands}. If the request can't "
         "be done, or would mean weakening safety, change nothing and say why. Finish with one or two plain "
         "sentences, for NIA to read aloud, saying what you changed.")
_job = threading.Lock()  # one change at a time


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", creationflags=subprocess.CREATE_NO_WINDOW)


def branch_name(request, now=None):
    """nia/<a few words of the request>-<when>, e.g. nia/integrate-discord-1009-231502"""
    words = re.findall(r"[a-z0-9]+", request.lower())[:5]
    return f"nia/{'-'.join(words) or 'change'}-{time.strftime('%m%d-%H%M%S', time.localtime(now))}"


def build_command(request):
    commands = [f"{PYTHON} {suite}" for suite in SUITES]
    prompt = BUILD.format(request=request, protected=", ".join(PROTECTED), commands="; ".join(commands))
    # The shell is PowerShell on Windows (Bash elsewhere) - either one, only for exactly these test runs; edits only
    # inside the checkout, never the protected files
    return [executable(), "-p", prompt, "--restricted", "--tools", "Read,Glob,Grep,Edit,Write,PowerShell,Bash",
            "--allowedTools", "Read", "Glob", "Grep", "Edit", "Write",
            *[f"{shell}({c})" for c in commands for shell in ("PowerShell", "Bash")],
            "--disallowedTools", *[f"{tool}({path})" for path in PROTECTED for tool in ("Edit", "Write")],
            "--permission-mode", "dontAsk", "--strict-mcp-config", "--no-session-persistence",
            "--effort", "medium", "--output-format", "stream-json", "--verbose"]


def run_tests(folder):
    """The test files that failed in folder - every tests/test_*.py, the new ones too"""
    failed = []
    for suite in sorted(Path(folder).glob("tests/test_*.py")):
        try:
            done = subprocess.run([PYTHON, f"tests/{suite.name}"], cwd=folder, capture_output=True,
                                  timeout=TEST_TIMEOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            if done.returncode != 0:
                failed.append(suite.name)
        except subprocess.TimeoutExpired:
            failed.append(f"{suite.name} (timed out)")
    return failed


def improve(request, root=ROOT):
    """Start building a change on a new branch in the background; returns at once with what's happening"""
    if not executable():
        return "Not started: Claude Code isn't installed here"
    if not _job.acquire(blocking=False):
        return "Not started: Claude is already building a change - one at a time"
    branch = branch_name(request)
    threading.Thread(target=_build, args=(request, Path(root), branch), daemon=True).start()
    return (f"Started: Claude is building it on a new branch, {branch}, in a separate copy - my running code isn't "
            "touched. It takes a few minutes; I'll say when it's ready for review")


def _build(request, root, branch):
    """Claude makes the change in its own checkout; it's committed to the branch only if every test passes"""
    folder = CHANGES / branch.split("/", 1)[1]
    started = time.time()
    try:
        CHANGES.mkdir(exist_ok=True)
        made = _git(root, "worktree", "add", "-q", "-b", branch, str(folder), "HEAD")
        if made.returncode:
            return _done(f"I couldn't set up a branch for it, sir: {made.stderr.strip()[:150]}")
        logger.info(f"Building {request!r} on {branch} in {folder}")
        proc = subprocess.run(build_command(request), cwd=folder, env=clean_env(), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=BUILD_TIMEOUT, stdin=subprocess.DEVNULL,
                              creationflags=subprocess.CREATE_NO_WINDOW)
        result = next((e for e in events(proc.stdout) if e.get("type") == "result"), {})
        summary = str(result.get("result") or "").strip()
        logger.info(f"Claude finished in {time.time() - started:.0f}s ({len(tool_calls(proc.stdout))} tool calls, "
                    f"${result.get('total_cost_usd', 0):.2f} at list price): {summary[:300]}")
        if not _git(folder, "status", "--porcelain").stdout.strip():
            _discard(root, folder, branch)
            return _done(f"Claude made no change, sir. {summary}".strip())
        failed = run_tests(folder)
        if failed:
            _discard(root, folder, branch)
            return _done(f"Claude's change broke {', '.join(failed)}, sir, so I've discarded it.")
        _git(folder, "add", "-A")
        _git(folder, "commit", "-q", "-m", f"NIA: {request}\n\nBuilt by Claude Code at the user's request, through "
             "NIA. Every test passed; not merged - for review.")
        _git(root, "worktree", "remove", "--force", str(folder))  # the branch stays
        _done(f"Done, sir. {summary} Every test passed. It's on the branch {branch}, ready for you to review and merge.")
    except subprocess.TimeoutExpired:
        _discard(root, folder, branch)
        _done(f"Claude didn't finish within {BUILD_TIMEOUT // 60} minutes, sir, so I've discarded it.")
    except Exception as e:  # never leave a half-made checkout behind
        logger.exception("Building a change failed")
        _discard(root, folder, branch)
        _done(f"That didn't work, sir ({e.__class__.__name__}); nothing was changed.")
    finally:
        _job.release()


def _discard(root, folder, branch):
    """Remove the checkout and its branch: nothing of a failed change is left"""
    _git(root, "worktree", "remove", "--force", str(folder))
    _git(root, "branch", "-D", branch)


def _done(text):
    logger.info(f"Change: {text}")
    if ON_DONE:
        ON_DONE(text)


@tool
def improve_myself(request: str) -> str:
    """Have Claude Code build a change to your own code - add an ability, fix a problem with yourself - when the user
    asks you to ("ask Claude to integrate Discord into you", "teach yourself to..."). request: what to build, in plain
    words with the details given. Always asks the user first. It's built on a new branch in a separate copy - your
    running code never changes - and kept only if every test passes; the user reviews and merges it. Never use it to
    remove or weaken a safety check"""
    return improve(request)


TOOLS = [ask_claude, improve_myself] if executable() else []
