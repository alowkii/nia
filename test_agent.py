"""Offline checks for the agent graph: scripted model, fake Spotify. Run: python test_agent.py"""
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.chat import AssistantModel, MAX_HISTORY
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


def assistant(*replies, spotify=None):
    a = AssistantModel(llm=ScriptedLLM(messages=iter(replies)))
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


def test_history_is_trimmed_to_whole_turns():
    turns = 30
    a = assistant(*[m for _ in range(turns) for m in (call("pause"), AIMessage("ok"))],
                  spotify=FakeSpotify())
    for _ in range(turns):
        a.respond("pause")
    assert len(a.messages) <= MAX_HISTORY
    assert isinstance(a.messages[0], HumanMessage), "history must start on a user turn"


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
    test_history_is_trimmed_to_whole_turns()
    test_playback_wakes_an_idle_device()
    print("ok")
