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
        (call("play_something"), ("play_something", ("",))),
        (call("play_something", mood="chill"), ("play_something", ("chill",))),
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


def test_approval_question_says_what_not_how():
    q = chat.approval_question
    close_tab = {"name": "execute", "args": {"command": "powershell -NoProfile -Command \"...SendKeys('%w')\""}}
    # The model's own words are what's heard - never the command
    assert q([close_tab], "I'll close the YouTube tab.") == "Before I do that: I'll close the YouTube tab. Should I go ahead?"
    # Without them, a plain summary
    assert q([close_tab], "") == "Before I do that: I'll press keys in the active window. Should I go ahead?"
    assert q([{"name": "execute", "args": {"command": "start https://www.youtube.com/watch?v=x"}}], "") == \
        "Before I do that: I'll open https://www.youtube.com/watch?v=x. Should I go ahead?"  # spoken as "the youtube link"
    assert "edit notes.txt" in q([{"name": "edit_file", "args": {"file_path": "/c/Users/me/notes.txt"}}], "")
    # For file changes the file's name beats a vague "I'll create that file for you"
    assert q([{"name": "write_file", "args": {"file_path": "/c/Users/me/groceries.txt"}}], "I'll create that file for you now.") == \
        "Before I do that: I'll save groceries.txt. Should I go ahead?"
    # Without the model's words, everyday commands still get plain descriptions - never "run a command"
    run = lambda cmd: q([{"name": "execute", "args": {"command": cmd}}], "")
    assert run("tasklist | findstr /i spotify") == "Before I do that: I'll check which programs are running. Should I go ahead?"
    assert run("taskkill /F /IM Spotify.exe; taskkill /F /IM SpotifyLauncher.exe") == \
        "Before I do that: I'll force-close Spotify and Spotifylauncher. Should I go ahead?"  # no repeated warning
    assert run("python backup.py") == "Before I do that: I'll run python. Should I go ahead?"
    # Every way people say yes counts; anything hedged doesn't
    for answer in ("Do that.", "Do it.", "Sure, go ahead", "Yes please", "Go for it", "Okay", "Alright", "Of course"):
        assert chat.YES.search(answer.lower()) and not chat.NO.search(answer.lower()), answer
    for answer in ("No, don't", "Wait", "Yes, no wait", "Hmm"):
        assert not (chat.YES.search(answer.lower()) and not chat.NO.search(answer.lower())), answer
    # A harmless-sounding intent can't hide a dangerous command
    wipe = {"name": "execute", "args": {"command": "Remove-Item C:\\Users\\me\\Documents -Recurse; shutdown /s"}}
    asked = q([wipe], "I'll tidy up your documents.")
    assert asked.startswith("Before I do that: I'll tidy up your documents.")
    assert "it deletes files" in asked and "it shuts down or restarts the PC" in asked


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
    assert "save notes.txt" in question and "go ahead" in question.lower() and "/notes" not in question
    assert not written(a, "/notes.txt"), "wrote before the user said yes"
    assert a.respond("yes, go ahead") == "Saved, sir."
    assert written(a, "/notes.txt")

    # Refused: anything but a clear yes, including "yes... no wait"
    for answer in ("no", "nope, cancel that", "yes - no wait", "what?"):
        a = assistant(call("execute", command="del C:\\stuff"), AIMessage("Okay, I won't, sir."))
        question = a.respond("clean up")
        assert "del C:" not in question and "it deletes files" in question, question
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


def test_stop_ends_the_turn_before_the_next_tool():
    from langchain_core.tools import tool
    ran = []

    @tool
    def first_step() -> str:
        """Step one"""
        ran.append("first")
        chat.CANCEL.set()  # the user says "stop" while this step runs
        return "done"

    @tool
    def second_step() -> str:
        """Step two"""
        ran.append("second")
        return "done"

    a = assistant(call("first_step"), call("second_step"), AIMessage("All done, sir."),
                  AIMessage("Evening, sir."), extra_tools=[first_step, second_step])
    chat.CANCEL.clear()
    assert a.respond("do both steps") == "", "a stopped turn has nothing to say"
    assert ran == ["first"], "a step ran after the user said stop"
    chat.CANCEL.clear()  # the voice layer clears it when the next turn starts
    assert a.respond("hello") in ("All done, sir.", "Evening, sir."), "the conversation must go on after a stop"


def test_long_tool_results_are_cut_to_fit():
    from langchain_core.tools import tool

    @tool
    def read_big_file() -> str:
        """Returns a huge file"""
        return "x" * 20000

    a = assistant(call("read_big_file"), AIMessage("It's mostly x, sir."), extra_tools=[read_big_file])
    a.respond("what's in the big file?")
    result = next(m for m in a.messages if isinstance(m, ToolMessage))
    assert len(result.content) < chat.MAX_TOOL_CHARS + 300 and "cut to fit" in result.content


def test_context_overflow_starts_a_fresh_conversation():
    from langchain_core.exceptions import ContextOverflowError

    class Overflowing(ScriptedLLM):
        def _generate(self, *args, **kwargs):
            raise ContextOverflowError("too much")

    a = AssistantModel(llm=Overflowing(messages=iter([])), backend=StateBackend())
    old_thread = a.config["configurable"]["thread_id"]
    assert "cleared our conversation" in a.respond("read every file I own")
    assert a.config["configurable"]["thread_id"] != old_thread


def test_youtube_plays_one_video_and_reads_its_transcript():
    from agent import youtube
    opened, fetched = [], []
    youtube.search = lambda q, count=1: [(f"vid{i:08d}", f"Video {i} for {q}", "Some Channel") for i in range(count)]
    youtube.webbrowser.open = opened.append
    youtube.last_opened = 0.0

    class FakeTranscripts:
        def fetch(self, video_id, languages):
            fetched.append(video_id)
            return [type("Snippet", (), {"text": "hello"})(), type("Snippet", (), {"text": "world"})()]
    youtube.YouTubeTranscriptApi = FakeTranscripts

    a = assistant(call("play_youtube", query="lofi"), AIMessage("Playing it, sir."),
                  call("youtube_transcript"), AIMessage("They say hello world, sir."))
    assert a.respond("play lofi on youtube") == "Playing it, sir."
    assert opened == ["https://www.youtube.com/watch?v=vid00000000"], "opened something other than a watch link"
    assert a.pending is None, "YouTube must not need approval"
    assert a.respond("summarize this video") == "They say hello world, sir."
    assert fetched == ["vid00000000"], "the transcript should default to the video just played"

    # A model that keeps "trying" gets one tab, not a screenful
    refused = youtube.play_youtube.invoke({"query": "lofi again"})
    assert "Not opened" in refused and len(opened) == 1
    # Searching lists results and opens nothing
    listed = youtube.search_youtube.invoke({"query": "stradman"})
    assert len(listed.splitlines()) == 5 and "vid00000004" in listed and len(opened) == 1
    # After the cooldown, a chosen id or link opens exactly that video
    youtube.last_opened -= youtube.OPEN_COOLDOWN
    youtube.play_youtube.invoke({"query": "https://youtu.be/dQw4w9WgXcQ?t=5"})
    assert opened[-1] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert "hello world" in youtube.youtube_transcript.invoke({"video": "dQw4w9WgXcQ"})


def test_play_something_opens_spotify_and_picks_from_preferences():
    import tempfile
    from pathlib import Path
    import settings
    from agent import action_controller
    action_controller.time.sleep = lambda s: None
    launched = []
    action_controller.os.startfile = launched.append

    class StubAPI:
        """Spotify with no app running until it's launched; then the laptop shows up and plays"""
        def __init__(self):
            self.searched, self.started, self.shuffled = None, None, None
        def devices(self):
            return {"devices": [{"id": "laptop", "type": "Computer", "is_active": False}] if launched else []}
        def search(self, q, limit, type):
            self.searched = q
            return {"playlists": {"items": [None, {"name": f"{q} mix", "uri": "spotify:playlist:p",
                                                   "tracks": {"total": 40}}]}}
        def start_playback(self, **kwargs):
            self.started = kwargs
        def current_playback(self):
            return {"item": {"name": "x"}, "is_playing": bool(self.started), "device": {"name": "ALOKLT"}}
        def shuffle(self, state):
            self.shuffled = state

    with tempfile.TemporaryDirectory() as tmp:
        settings.PATH = Path(tmp) / "settings.json"  # never touch the real settings.json
        settings.save({**settings.DEFAULTS, "music_moods": "rainy day jazz, gym bangers"})
        c = object.__new__(SpotifyController)  # skip OAuth
        c.sp = StubAPI()
        reply = c.play_something()
        assert launched == ["spotify:"], "should open the Spotify app when it isn't running"
        assert c.sp.searched in ("rainy day jazz", "gym bangers"), "the random pick must come from the setting"
        assert c.sp.started["device_id"] == "laptop" and 0 <= c.sp.started["offset"]["position"] < 40
        assert c.sp.shuffled is True and "shuffled" in reply
        assert c.play_something("chill") and c.sp.searched == "chill"  # a named mood wins
        assert launched == ["spotify:"], "an app that's already open isn't launched again"


def test_restart_spotify_only_touches_spotify():
    from agent import action_controller
    action_controller.time.sleep = lambda s: None
    killed, launched = [], []
    action_controller.subprocess.run = lambda cmd, **kw: killed.append(cmd)
    action_controller.os.startfile = launched.append

    class StubAPI:
        def devices(self):  # the app is gone until it's relaunched
            return {"devices": [{"id": "laptop", "type": "Computer", "is_active": False}] if launched else []}

    c = object.__new__(SpotifyController)  # skip OAuth
    c.sp = StubAPI()
    assert "back online" in c.restart_app()
    assert killed == [["taskkill", "/F", "/IM", "Spotify.exe"], ["taskkill", "/F", "/IM", "SpotifyLauncher.exe"]]
    assert launched == ["spotify:"]


def test_playback_wakes_an_idle_device_and_checks_it_started():
    from agent import action_controller
    action_controller.time.sleep = lambda s: None  # no real waiting for the start check
    action_controller.os.startfile = lambda uri: None  # never launch the real app

    class StubAPI:
        def __init__(self, devices, plays=True):
            self.devices_list, self.started, self.plays = devices, None, plays
        def devices(self):
            return {"devices": self.devices_list}
        def start_playback(self, **kwargs):
            self.started = kwargs
        def current_playback(self):  # a stuck app accepts the command but loads nothing
            item = {"name": "x"} if self.plays else None
            return {"item": item, "is_playing": self.plays, "device": {"name": "ALOKLT"}}

    def start(devices, plays=True):
        c = object.__new__(SpotifyController)  # skip OAuth
        c.sp = StubAPI(devices, plays)
        c._start(context_uri="spotify:playlist:x")
        return c.sp.started

    laptop = {"id": "laptop", "type": "Computer", "is_active": False}
    phone = {"id": "phone", "type": "Smartphone", "is_active": False}
    # Only idle devices: target one, or Spotify answers "No active device found" - the computer
    # NIA runs on, even when a phone is listed first
    assert start([laptop]) == {"context_uri": "spotify:playlist:x", "device_id": "laptop"}
    assert start([phone, laptop]) == {"context_uri": "spotify:playlist:x", "device_id": "laptop"}
    # Something already active: leave the choice to Spotify
    assert start([laptop, dict(phone, is_active=True)]) == {"context_uri": "spotify:playlist:x"}
    # No devices at all: the app is opened; if it never comes online, that error goes back to the model
    try:
        start([])
        raise AssertionError("played with no device at all")
    except RuntimeError as e:
        assert "didn't come online" in str(e)
    # Accepted but nothing started (stuck app): an error, never "Successfully playing"
    try:
        start([laptop], plays=False)
        raise AssertionError("a stuck Spotify app was reported as playing")
    except RuntimeError as e:
        assert "nothing started playing on ALOKLT" in str(e)


if __name__ == "__main__":
    test_tools_reach_spotify()
    test_spotify_errors_go_back_to_the_model()
    test_chat_does_not_build_spotify()
    test_approval_question_says_what_not_how()
    test_conversation_carries_over_turns()
    test_pc_changes_need_a_spoken_yes()
    test_extra_tools_reach_the_agent()
    test_stop_ends_the_turn_before_the_next_tool()
    test_long_tool_results_are_cut_to_fit()
    test_context_overflow_starts_a_fresh_conversation()
    test_youtube_plays_one_video_and_reads_its_transcript()
    test_play_something_opens_spotify_and_picks_from_preferences()
    test_restart_spotify_only_touches_spotify()
    test_playback_wakes_an_idle_device_and_checks_it_started()
    print("ok")
