import logging
import re
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Literal

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from langchain.agents.middleware import wrap_tool_call
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

import settings
from . import youtube
from .prompts.initial import initial_prompt, pc_prompt
from .action_controller import SpotifyController

logger = logging.getLogger(__name__)

# Tools that change the PC: NIA reads each call back and only runs it after a spoken "yes"
NEEDS_APPROVAL = ("write_file", "edit_file", "delete", "execute")
APPROVAL_EXPIRES = 60  # seconds; a later "yes" must not approve a stale action
YES = re.compile(r"\b(yes|yeah|yep|yup|sure|confirm(ed)?|approved?|go ahead|do it|ok(ay)?)\b")
NO = re.compile(r"\b(no|nope|don'?t|do not|stop|cancel|wait|never)\b")

# Server output also goes here, so it survives the server's console window closing
SERVER_LOG = Path(__file__).resolve().parent.parent / "logs" / "llama-server.log"


def llm_server_command(s=None):
    """llama-server (PrismML's fork) running Bonsai 2 27B fully on the GPU, from settings"""
    s = s or settings.load()
    bonsai = Path(s["bonsai_dir"])
    SERVER_LOG.parent.mkdir(exist_ok=True)
    return [str(bonsai / "llama-prism" / "llama-server.exe"), "-m", str(bonsai / s["model_file"]),
            "-ngl", "99", "-fa", "on", "-c", str(s["context"]), "-np", "1",
            "--reasoning", "on" if s["thinking"] else "off",
            "--temp", str(s["temperature"]), "--top-p", str(s["top_p"]), "--top-k", str(s["top_k"]),
            "--presence-penalty", str(s["presence_penalty"]),
            "--host", "127.0.0.1", "--port", str(s["port"]), "--log-file", str(SERVER_LOG), "--log-colors", "off"]


def llm_up(port):
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200
    except OSError:  # refused while starting, 503 while the model loads
        return False


def ensure_llm_server(timeout=180):
    """Start llama-server in its own window unless it is already up, then wait for it"""
    s = settings.load()
    if llm_up(s["port"]):
        logger.info(f"LLM server already running on port {s['port']}")
        return
    logger.info(f"Starting LLM server: {' '.join(llm_server_command(s))}")
    started = time.time()
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


@wrap_tool_call
def tool_guard(request, handler):
    """Keeps tools from breaking the turn: a failing tool (e.g. Spotify's "No active device") becomes
    a message the model can explain, and a long result is cut to fit the context. Every call is
    logged as it runs, so a turn that loops (or gets stopped) still shows what it did."""
    call = request.tool_call
    logger.info(f"Tool call: {call['name']}({call['args']})")
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


def describe(action):
    """One spoken line for a pending tool call, so the user knows exactly what they're approving"""
    args = action["args"]
    if action["name"] == "execute":
        return f"run the command: {args.get('command')}"
    verb = {"write_file": "write the file", "edit_file": "edit the file", "delete": "delete"}[action["name"]]
    return f"{verb} {args.get('file_path') or args.get('path')}"


class AssistantModel:
    """NIA's brain: a Deep Agent (LangGraph) on Bonsai with Spotify tools, the PC's files and shell,
    and sub-agents. Changes to the PC wait for a spoken yes - see NEEDS_APPROVAL."""

    def __init__(self, model="bonsai", llm=None, extra_tools=(), backend=None):
        """extra_tools: tools from outside the agent, e.g. the voice layer's own volume control.
        backend: where file and shell tools act; the real PC unless a test passes another."""
        s = settings.load()
        self._spotify = None  # built on first music tool call, not at startup
        self.pending = None  # (number of actions awaiting approval, when they were asked about)
        self.config = {"configurable": {"thread_id": "nia"}}
        # llama-server ignores the model name and key, but the client requires both. The profile tells
        # Deep Agents the real context size, so it summarizes old turns before overflowing it
        llm = llm or ChatOpenAI(model=model, base_url=f"http://127.0.0.1:{s['port']}/v1", api_key="none",
                                profile={"max_input_tokens": s["context"]})
        self.agent = create_deep_agent(
            model=llm,
            tools=self._spotify_tools() + youtube.TOOLS + list(extra_tools),
            system_prompt=initial_prompt + pc_prompt,
            backend=backend or pc_backend(),
            interrupt_on={name: True for name in NEEDS_APPROVAL},
            middleware=[tool_guard],
            checkpointer=InMemorySaver(),  # the conversation, kept between turns
        )

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
            decision = {"type": "approve"} if approved else {"type": "reject"}
            result = self.agent.invoke(Command(resume={"decisions": [decision] * self.pending[0]}), self.config)
        else:
            # The clock rides on the user turn, not the system prompt, so the prompt prefix stays
            # identical and llama-server can reuse its cache
            message = HumanMessage(f"{user_msg}\n\n[{datetime.now():%A %d %B %Y, %H:%M}]")
            result = self.agent.invoke({"messages": [message]}, self.config)

        if result.get("__interrupt__"):
            actions = result["__interrupt__"][0].value["action_requests"]
            self.pending = (len(actions), time.time())
            logger.info(f"Awaiting approval for: {actions}")
            return "Before I do that: I'll " + ", then ".join(describe(a) for a in actions) + ". Should I go ahead?"
        self.pending = None
        return result["messages"][-1].content

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

        return [now_playing, play, play_something, add_to_queue, pause, resume, skip, set_volume, change_volume,
                shuffle, repeat]
