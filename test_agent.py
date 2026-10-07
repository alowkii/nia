"""Offline checks for the agent: scripted model, fake Spotify, in-memory files (never the real PC).
Run: python test_agent.py"""
import time

from deepagents.backends import StateBackend
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import chat
from agent.chat import AssistantModel
from agent.action_controller import SpotifyController


class FakeSpotify(SpotifyController):
    """Records calls instead of hitting the Spotify API."""
    def __init__(self):
        self.calls = []

    def __getattribute__(self, name):
        if name in ("calls", "__class__", "__dict__"):
            return object.__getattribute__(self, name)
        def record(*args):
            self.calls.append((name, args))
            return f"did {name}"
        return record


class ScriptedLLM(GenericFakeChatModel):
    """Replays canned AIMessages; tools are already wired by the graph."""
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, **args):
    """A model turn that calls one tool"""
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": name}])


def assistant(*replies, spotify=None, extra_tools=()):
    a = AssistantModel(llm=ScriptedLLM(messages=iter(replies)), extra_tools=extra_tools, backend=StateBackend())
    a._spotify = spotify
    return a


def test_tools_reach_spotify():
    cases = [
        (call("play", query="Back in Black", kind="album"), ("play_album", ("Back in Black",))),
        (call("play", query="Highway to Hell"), ("play_track", ("Highway to Hell",))),
        (call("skip", direction="previous"), ("previous", ())),
        (call("repeat", mode="off"), ("repeat", ("off",))),
        (call("change_volume", step=-10), ("change_volume", (-10,))),
    ]
    for tool_call, expected in cases:
        a = assistant(tool_call, AIMessage("Done, sir."), spotify=FakeSpotify())
        assert a.respond("do it") == "Done, sir."
        assert a._spotify.calls == [expected], (tool_call.tool_calls, a._spotify.calls)


def test_spotify_errors_go_back_to_the_model():
    class Boom:
        def __getattr__(self, name):
            def fail(*args):
                raise RuntimeError("no active device")
            return fail
    a = assistant(call("pause"), AIMessage("No device is active, sir."), spotify=Boom())
    assert a.respond("pause") == "No device is active, sir."
    tool_msg = next(m for m in a.messages if isinstance(m, ToolMessage))
    assert "no active device" in tool_msg.content


def test_chat_does_not_build_spotify():
    a = assistant(AIMessage("Evening, sir."))
    assert a.respond("Hey Nia!") == "Evening, sir."
    assert a._spotify is None, "built a Spotify client for a plain chat turn"


def test_conversation_carries_over_turns():
    a = assistant(AIMessage("Evening, sir."), AIMessage("You said hello, sir."))
    a.respond("hello")
    a.respond("what did I just say?")
    humans = [m for m in a.messages if isinstance(m, HumanMessage)]
    assert len(humans) == 2 and humans[0].content.startswith("hello")
    assert "[" in humans[0].content, "the user turn carries the current time"


def written(a, path):
    return path in (a.agent.get_state(a.config).values.get("files") or {})


def test_pc_changes_need_a_spoken_yes():
    # Approved: the read-back names the file, and the write happens only after "yes"
    a = assistant(call("write_file", file_path="/notes.txt", content="milk"), AIMessage("Saved, sir."))
    question = a.respond("note down milk")
    assert "write the file /notes.txt" in question and "go ahead" in question.lower()
    assert not written(a, "/notes.txt"), "wrote before the user said yes"
    assert a.respond("yes, go ahead") == "Saved, sir."
    assert written(a, "/notes.txt")

    # Refused: anything but a clear yes, including "yes... no wait"
    for answer in ("no", "nope, cancel that", "yes - no wait", "what?"):
        a = assistant(call("execute", command="del C:\\stuff"), AIMessage("Okay, I won't, sir."))
        assert "run the command: del C:\\stuff" in a.respond("clean up")
        assert a.respond(answer) == "Okay, I won't, sir.", answer
        tool_msg = next(m for m in a.messages if isinstance(m, ToolMessage))
        assert tool_msg.status == "error" or "reject" in str(tool_msg.content).lower(), tool_msg

    # Stale: once the question expires, the next sentence is a new request, not the answer
    a = assistant(call("write_file", file_path="/old.txt", content="x"), AIMessage("Left it alone, sir."),
                  AIMessage("Evening, sir."))
    a.respond("write old.txt")
    count, asked = a.pending
    a.pending = (count, asked - chat.APPROVAL_EXPIRES - 1)
    assert a.respond("yes, hello") == "Evening, sir."  # answered as a new turn
    assert not written(a, "/old.txt"), "a late yes approved a stale action"
    assert a.pending is None

    # Reading needs no approval
    a = assistant(call("ls", path="/"), AIMessage("Nothing there, sir."))
    assert a.respond("what's in my files?") == "Nothing there, sir."


def test_extra_tools_reach_the_agent():
    """The voice layer hands in its own tools (e.g. voice volume) next to Spotify's"""
    from langchain_core.tools import tool
    heard = []

    @tool
    def set_voice_volume(percent: int) -> str:
        """Set your own voice volume"""
        heard.append(percent)
        return "ok"

    a = assistant(call("set_voice_volume", percent=60), AIMessage("Quieter now, sir."), extra_tools=[set_voice_volume])
    assert a.respond("talk at 60 percent") == "Quieter now, sir."
    assert heard == [60]


def test_playback_wakes_an_idle_device():
    class StubAPI:
        def __init__(self, devices):
            self.devices_list, self.started = devices, None
        def devices(self):
            return {"devices": self.devices_list}
        def start_playback(self, **kwargs):
            self.started = kwargs

    def start(devices):
        c = object.__new__(SpotifyController)  # skip OAuth
        c.sp = StubAPI(devices)
        c._start(context_uri="spotify:playlist:x")
        return c.sp.started

    idle = {"id": "laptop", "is_active": False}
    # Only an idle device: target it, or Spotify answers "No active device found"
    assert start([idle]) == {"context_uri": "spotify:playlist:x", "device_id": "laptop"}
    # Something already active: leave the choice to Spotify
    assert start([idle, {"id": "phone", "is_active": True}]) == {"context_uri": "spotify:playlist:x"}
    # No devices at all: Spotify's own error goes back to the model
    assert start([]) == {"context_uri": "spotify:playlist:x"}


if __name__ == "__main__":
    test_tools_reach_spotify()
    test_spotify_errors_go_back_to_the_model()
    test_chat_does_not_build_spotify()
    test_conversation_carries_over_turns()
    test_pc_changes_need_a_spoken_yes()
    test_extra_tools_reach_the_agent()
    test_playback_wakes_an_idle_device()
    print("ok")
