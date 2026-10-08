import logging
import os
import re
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Literal

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

import settings
from utils.hud import hud
from . import approval, youtube
from .server import SERVER_LOG, llm_server_command, llm_up
from .approval import risks
from .memory import Memory, note
from .prompts.initial import author, initial_prompt, pc_prompt
from .action_controller import SpotifyController

logger = logging.getLogger(__name__)

# Tools that change the PC: NIA reads each call back and only runs it after a spoken "yes"
NEEDS_APPROVAL = ("write_file", "edit_file", "delete", "execute")
APPROVAL_EXPIRES = 60  # seconds; a later "yes" must not approve a stale action
YES = re.compile(r"\b(yes|yeah|yep|yup|sure|confirm(ed)?|approved?|go ahead|go for it|do (it|that|so)|please do|"
                 r"of course|absolutely|ok(ay)?|alright|all right)\b")
NO = re.compile(r"\b(no|nope|don'?t|do not|stop|cancel|wait|never)\b")

def ensure_llm_server(timeout=180):
    """Start llama-server in its own window unless it is already up, then wait for it. Under nia.py the
    window's script owns the server (started hidden), so this only waits for it"""
    s = settings.load()
    if llm_up(s["port"]):
        logger.info(f"LLM server already running on port {s['port']}")
        return
    started = time.time()
    if os.getenv("NIA_HUD"):
        logger.info("Waiting for nia.py's LLM server...")
    else:
        logger.info(f"Starting LLM server: {' '.join(llm_server_command(s))}")
        subprocess.Popen(llm_server_command(s), creationflags=subprocess.CREATE_NEW_CONSOLE)
    while not llm_up(s["port"]):
        if time.time() - started > timeout:
            raise RuntimeError(f"llama-server did not come up on port {s['port']} - see {SERVER_LOG}")
        time.sleep(1)
    logger.info(f"LLM server up after {time.time() - started:.1f}s")


def pc_backend():
    """The real PC for the file and shell tools. Deep Agents' file tools only take /-style paths
    (they reject C:\\...), so each drive is mounted as /c/, /d/, ...; anything else is under the
    home folder. The shell runs in the home folder and takes normal Windows paths."""
    drives = {f"/{letter.lower()}/": FilesystemBackend(root_dir=f"{letter}:\\", virtual_mode=True)
              for letter in "CDEFGHIJ" if Path(f"{letter}:\\").exists()}
    # ponytail: no sandbox - the shell and files are the real PC; the spoken yes is the only guard
    shell = LocalShellBackend(root_dir=Path.home(), virtual_mode=True, inherit_env=True, timeout=60)
    return CompositeBackend(default=shell, routes=drives)


# Longest tool result the model sees. Deep Agents only offloads results over 20K tokens - more than
# Bonsai's whole context - so a few files read at once overflowed it
MAX_TOOL_CHARS = 3000  # ~750 tokens

# Set by the voice layer when the user says "stop" mid-turn: the next tool call ends the turn
CANCEL = threading.Event()
# Set when the user refuses an action: for the rest of that turn only looking things up is allowed. After
# "no" to Google News, one session opened a YouTube news video instead without asking
REFUSED = threading.Event()
READ_ONLY_TOOLS = {"now_playing", "search_youtube", "youtube_transcript", "ls", "read_file", "glob", "grep"}
REFUSAL = ("The user said no. Don't do this, and don't try another way to do it or anything else in its place - "
           "just acknowledge briefly and ask what they'd like instead.")


class Cancelled(Exception):
    """The user said stop while NIA was working"""


def recent_turns(turns):
    """Middleware: only recent exchanges reach the model - between `turns` and 2 x `turns` of them. Older
    ones stay in the saved conversation, and the memory search brings back whichever matter. Cuts in
    steps (at 2x, back to 1x) rather than every turn, because each cut changes the prompt's opening and
    costs llama-server its cache; and always at a user message, so a tool call never loses its result"""
    @wrap_model_call
    def keep_recent_turns(request, handler):
        users = [i for i, m in enumerate(request.messages) if isinstance(m, HumanMessage)]
        if len(users) > 2 * turns:
            keep_from = users[-(turns + (len(users) - 1) % turns)]  # the step boundary, stable for `turns` turns
            request = request.override(messages=request.messages[keep_from:])
        return handler(request)
    return keep_recent_turns


@wrap_tool_call
def tool_guard(request, handler):
    """Keeps tools from breaking the turn: a failing tool (e.g. Spotify's "No active device") becomes
    a message the model can explain, and a long result is cut to fit the context. Every call is
    logged as it runs, so a turn that loops (or gets stopped) still shows what it did."""
    call = request.tool_call
    if CANCEL.is_set():  # the one checkpoint every tool passes through - nothing more runs after "stop"
        logger.info(f"Cancelled before {call['name']}({call['args']}) - the user said stop")
        raise Cancelled
    if REFUSED.is_set() and call["name"] not in READ_ONLY_TOOLS:
        logger.info(f"Blocked {call['name']}({call['args']}) - the user just said no")
        return ToolMessage(f"Not done: {REFUSAL}", tool_call_id=call["id"], name=call["name"], status="error")
    logger.info(f"Tool call: {call['name']}({call['args']})")
    hud.send(tool=call["name"])
    try:
        result = handler(request)
    except GraphBubbleUp:  # interrupts and other LangGraph control flow must pass through
        raise
    except Exception as e:
        logger.info(f"Tool error ({call['name']}): {e!r}")
        return ToolMessage(f"Error: {e!r}", tool_call_id=call["id"], name=call["name"], status="error")
    if isinstance(result, ToolMessage) and isinstance(result.content, str) and len(result.content) > MAX_TOOL_CHARS:
        cut = len(result.content) - MAX_TOOL_CHARS
        result.content = (result.content[:MAX_TOOL_CHARS] + f"\n[... {cut} more characters cut to fit your memory. "
                          "Read less at once: a smaller line range, one file at a time, or a narrower search]")
    logger.info(f"Tool result ({call['name']}): {str(getattr(result, 'content', result))[:500]}")
    return result


# Plain words for everyday commands, when the model doesn't say what it's doing
COMMON_COMMANDS = [
    (r"\b(tasklist|get-process)\b", "check which programs are running"),
    (r"\b(dir|ls|get-childitem|gci)\b", "list a folder"),
    (r"\b(type|cat|get-content|gc)\b", "read a file"),
    (r"\b(ipconfig|ping|tracert|nslookup|test-connection)\b", "check the network"),
    (r"\b(systeminfo|get-computerinfo|wmic|get-ciminstance)\b", "look up system information"),
    (r"^\s*start\s", "open something"),
]


def describe(action):
    """What NIA plans to do, in plain words - never the raw command (the log keeps that)"""
    args, name = action["args"], action["name"]
    if name != "execute":
        file = Path(str(args.get("file_path") or args.get("path") or "")).name or "a file"
        return {"write_file": f"save {file}", "edit_file": f"edit {file}", "delete": f"delete {file}"}[name]
    command = str(args.get("command", ""))
    if url := re.search(r"https?://[^\s\"']+|www\.[^\s\"']+", command):  # no trailing quote
        return f"open {url.group(0)}"  # spoken as "the youtube link"
    if re.search(r"sendkeys", command, re.I):
        return "press keys in the active window"
    if programs := re.findall(r"(?:/im|-name)\s+\"?([\w.-]+?)(?:\.exe)?\"?(?:\s|;|$)", command, re.I):
        if re.search(r"taskkill|stop-process", command, re.I):
            return "force-close " + " and ".join(dict.fromkeys(p.title() for p in programs))
    for pattern, summary in COMMON_COMMANDS:
        if re.search(pattern, command, re.I):
            return summary
    # The program's own name: C:\Users\me\Downloads\jo.exe is "jo.exe", not "C"
    first = re.match(r'\s*(?:powershell\S*\s+(?:-\S+\s+)*)?(?:"([^"]+)"|([^"\s]+))', command)
    return f"run {PureWindowsPath(first.group(1) or first.group(2)).name}" if first else "run a command"


def approval_question(actions, intent):
    """The spoken approval question, plus any risk the commands carry. File changes are summarized
    with the file's name ("save groceries.txt") - the one detail that matters there. For commands,
    the model's own short intent ("I'll close the YouTube tab") says more than a summary can"""
    intent = " ".join(str(intent or "").split()[:25]).strip()
    commands = any(a["name"] == "execute" for a in actions)
    plan = intent.rstrip(".") if intent and commands else "I'll " + ", then ".join(describe(a) for a in actions)
    # A risk the plan already names ("force-close Spotify" / "it closes programs") needn't be said twice
    notes = sorted({note for a in actions for note in risks(a) if note.split()[1][:4] not in plan.lower()})
    warning = f" Note that {' and '.join(notes)}." if notes else ""
    return f"Before I do that: {plan}.{warning} Should I go ahead?"


class AssistantModel:
    """NIA's brain: a Deep Agent (LangGraph) on Bonsai with Spotify tools, the PC's files and shell,
    and sub-agents. Changes to the PC wait for a spoken yes - see NEEDS_APPROVAL."""

    def __init__(self, model="bonsai", llm=None, extra_tools=(), backend=None, memory=None):
        """extra_tools: tools from outside the agent, e.g. the voice layer's own volume control.
        backend: where file and shell tools act; the real PC unless a test passes another.
        memory: long-term memory; by default from settings (embedding_model), False for none."""
        s = settings.load()
        self._spotify = None  # built on first music tool call, not at startup
        self.pending = None  # (number of actions awaiting approval, when they were asked about)
        self.config = {"configurable": {"thread_id": "nia"}}
        self.approval_model, self.approval_threshold = s["approval_model"], s["approval_threshold"]
        if memory is None:
            memory = Memory(s["embedding_model"]) if s["embedding_model"] else False
        self.memory, self.memory_results, self.memory_min = memory, s["memory_results"], s["memory_min_similarity"]
        self.recent = deque(maxlen=2 * s["history_turns"])  # memory ids of exchanges possibly still in view
        self.request = ""  # the user's request this turn - an approval answer like "yes" isn't worth remembering
        # llama-server ignores the model name and key, but the client requires both. The profile tells
        # Deep Agents the real context size, so it summarizes old turns before overflowing it
        llm = llm or ChatOpenAI(model=model, base_url=f"http://127.0.0.1:{s['port']}/v1", api_key="none",
                                profile={"max_input_tokens": s["context"]})
        self.agent = create_deep_agent(
            model=llm,
            tools=self._spotify_tools() + youtube.TOOLS + list(extra_tools) + (self.memory.tools() if self.memory else []),
            system_prompt=initial_prompt + pc_prompt,
            backend=backend or pc_backend(),
            interrupt_on={name: True for name in NEEDS_APPROVAL},
            middleware=[tool_guard, recent_turns(s["history_turns"])],
            # The stock sub-agent, plus tool_guard: Deep Agents doesn't pass custom middleware down, so its
            # steps would ignore "stop", skip the result cap and go unlogged. It inherits the tools and approvals
            subagents=[{**GENERAL_PURPOSE_SUBAGENT, "middleware": [tool_guard]}],
            checkpointer=InMemorySaver(),  # the conversation, kept between turns
        )

    def warm_up(self):
        """Have llama-server read the system prompt and tools (~6K tokens, ~10 s) once now, so the first real turn
        reuses its cache instead of paying for it. A throwaway conversation, with every action blocked."""
        started = time.time()
        REFUSED.set()
        try:
            self.agent.invoke({"messages": [HumanMessage("Hello")]}, {"configurable": {"thread_id": "warm-up"}})
            logger.info(f"Warmed up the LLM's prompt cache in {time.time() - started:.1f}s")
        except Exception as e:  # only a speed-up: never stop NIA starting
            logger.warning(f"Warm-up failed ({e!r}) - the first reply will be slower")
        finally:
            REFUSED.clear()

    @property
    def messages(self):
        state = self.agent.get_state(self.config)
        return state.values.get("messages", []) if state else []

    def spotify(self):
        if self._spotify is None:
            self._spotify = SpotifyController()
        return self._spotify

    def respond(self, user_msg):
        """Run one turn, including any tool calls, and return what NIA should say.
        If a PC change needs approval, that's the question; the next call is taken as the answer."""
        try:
            return self._respond(user_msg)
        except Cancelled:
            # Steps already taken stay taken; the dangling call is patched up by Deep Agents next turn
            self.pending = None
            return ""
        except ContextOverflowError:
            # Even summarizing couldn't fit it; every later turn would fail the same way, so start over
            logger.exception("Context overflow - starting a fresh conversation")
            self.config = {"configurable": {"thread_id": f"nia-{time.time():.0f}"}}
            self.pending = None
            return "That was more than I can hold at once, sir, so I've cleared our conversation. Try asking about less at a time."

    def _respond(self, user_msg):
        if self.pending and time.time() - self.pending[1] >= APPROVAL_EXPIRES:
            # Nobody answered in time: refuse quietly, then treat this as a new request, not as the answer
            logger.info("Approval expired - refused")
            self.agent.invoke(Command(resume={"decisions": [{"type": "reject"}] * self.pending[0]}), self.config)
            self.pending = None
        if self.pending:
            answer = user_msg.lower()
            approved = bool(YES.search(answer)) and not NO.search(answer)
            logger.info(f"Approval {'given' if approved else 'refused'}: {user_msg!r}")
            decision = {"type": "approve"} if approved else {"type": "reject", "message": REFUSAL}
            if not approved:
                REFUSED.set()
            result = self.agent.invoke(Command(resume={"decisions": [decision] * self.pending[0]}), self.config)
        else:
            REFUSED.clear()  # a new request: a "no" to the last one no longer applies
            # The clock and any recalled memories ride on the user turn, not the system prompt, so the
            # prompt prefix stays identical and llama-server can reuse its cache
            self.request = user_msg
            hits = self.memory.search(user_msg, self.memory_results, self.memory_min, exclude=set(self.recent)) \
                if self.memory else []
            for similarity, _, kind, text, _ in hits:
                logger.info(f"Recalled ({kind}, {similarity:.2f}): {text[:120]}")
            message = HumanMessage(f"{user_msg}\n\n[{datetime.now():%A %d %B %Y, %H:%M}]{note(hits)}")
            result = self.agent.invoke({"messages": [message]}, self.config)

        # Commands the approval layers clear run straight away; anything else waits for a spoken yes
        while result.get("__interrupt__"):
            actions = result["__interrupt__"][0].value["action_requests"]
            verdicts = [approval.decide(a, self.approval_model, self.approval_threshold) for a in actions]
            for action, (verdict, why) in zip(actions, verdicts):
                logger.info(f"Approval check: {verdict} - {why} | {action['name']}({action['args']})")
            if not all(verdict == "run" for verdict, _ in verdicts):
                break
            result = self.agent.invoke(Command(resume={"decisions": [{"type": "approve"}] * len(actions)}), self.config)

        if result.get("__interrupt__"):
            actions = result["__interrupt__"][0].value["action_requests"]
            self.pending = (len(actions), time.time())
            logger.info(f"Awaiting approval for: {actions}")  # the exact commands, for the record
            # The model's text alongside its tool call is its own short account of what it's doing
            intent = result["messages"][-1].content if result["messages"] else ""
            return approval_question(actions, intent if isinstance(intent, str) else "")
        self.pending = None
        reply = result["messages"][-1].content
        if self.memory and self.request and isinstance(reply, str) and reply:
            # Every finished exchange is searchable later - this is what replaces sending the whole history
            exchange = f"{author} asked: {self.request} | NIA answered: {reply[:300]}"
            if (memory_id := self.memory.add(exchange, "exchange")) is not None:
                self.recent.append(memory_id)
        return reply

    def _spotify_tools(self):
        sp = self.spotify  # called inside each tool so the client stays lazy

        @tool
        def now_playing() -> str:
            """What is playing on Spotify right now, and whether it is paused"""
            return sp().now_playing()

        @tool
        def play(query: str, kind: Literal["track", "playlist", "album"] = "track") -> str:
            """Search Spotify and play the top hit. query is the name, plus the artist if known.
            kind is 'album' or 'playlist' when the user says so, otherwise 'track'"""
            return {"track": sp().play_track, "playlist": sp().play_playlist, "album": sp().play_album}[kind](query)

        @tool
        def play_something(mood: str = "") -> str:
            """Play music when no song, artist or playlist was named: "open Spotify", "play some music",
            "surprise me", "play something chill". mood is optional, e.g. "chill" or "workout".
            Opens the Spotify app first if it isn't running"""
            return sp().play_something(mood)

        @tool
        def restart_spotify() -> str:
            """Force-close and reopen the Spotify app on this PC. For "restart Spotify", or when Spotify is
            stuck (it accepts commands but nothing plays). Use this - never the shell - for closing Spotify"""
            return sp().restart_app()

        @tool
        def add_to_queue(query: str) -> str:
            """Queue a track to play next. query is the track name, plus the artist if known"""
            return sp().add_to_queue(query)

        @tool
        def pause() -> str:
            """Pause Spotify"""
            return sp().pause()

        @tool
        def resume() -> str:
            """Resume Spotify playback"""
            return sp().resume()

        @tool
        def skip(direction: Literal["next", "previous"] = "next") -> str:
            """Skip to the next track, or go back to the previous one"""
            return sp().next() if direction == "next" else sp().previous()

        @tool
        def set_volume(percent: int) -> str:
            """Set Spotify volume to an exact percent, 0-100"""
            return sp().set_volume(percent)

        @tool
        def change_volume(step: int) -> str:
            """Turn Spotify volume up (positive step) or down (negative step) by step percent. Use 10 or -10 unless told an amount"""
            return sp().change_volume(step)

        @tool
        def shuffle(on: bool = True) -> str:
            """Turn Spotify shuffle on or off"""
            return sp().shuffle(on)

        @tool
        def repeat(mode: Literal["track", "context", "off"]) -> str:
            """Repeat the current track, repeat the current playlist or album ('context'), or turn repeat off"""
            return sp().repeat(mode)

        return [now_playing, play, play_something, restart_spotify, add_to_queue, pause, resume, skip, set_volume,
                change_volume, shuffle, repeat]
