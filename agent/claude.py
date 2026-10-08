"""Ask Claude: NIA hands a question she can't answer well to Claude Code, which searches the web and answers.

Runs Claude Code headless with fixed settings, never ones the model chooses: only web search and page fetching
(--restricted strips every tool that runs commands, and ignores settings files), anything else refused without a
prompt, no MCP servers, no saved session, in an empty folder. So whoever is talking to NIA can get answers through
it, but can't make it touch the PC. Each call uses the Claude plan it's logged in with.
"""
import json
import logging
import os
import shutil
import subprocess
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
    return shutil.which("claude") or next((str(p) for p in (Path.home() / ".local" / "bin" / "claude.exe",)
                                           if p.exists()), None)


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


TOOLS = [ask_claude] if executable() else []
