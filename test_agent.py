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


def without_waiting(test):
    """Spotify's waits skipped for one test. time.sleep is one function shared by every module, so it's put back
    afterwards - left patched, it once made a later test's 60-second wait finish at once"""
    def wrapped():
        real = time.sleep
        time.sleep = lambda seconds: None
        try:
            test()
        finally:
            time.sleep = real
    wrapped.__name__ = test.__name__
    return wrapped


def assistant(*replies, spotify=None, extra_tools=(), approval_model=None, memory=False, llm=None):
    a = AssistantModel(llm=llm or ScriptedLLM(messages=iter(replies)), extra_tools=extra_tools,
                       backend=StateBackend(), memory=memory)
    a._spotify = spotify
    a.approval_model = approval_model  # no decision model unless a test stubs one in - never real Ollama
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


def word_embed(texts):
    """Stand-in for EmbeddingGemma: one dimension per word, so similarity = shared words. Never Ollama."""
    import re
    import zlib
    import numpy as np
    vectors = np.zeros((len(texts), 512), dtype=np.float32)
    for row, text in enumerate(texts):
        text = text.split(": ", 2)[-1]  # drop the task prefix
        for word in re.findall(r"[a-z]+", text.lower()):
            vectors[row, zlib.crc32(word.encode()) % 512] += 1  # not hash(): that's randomized per run
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)


def test_memory_finds_what_matters_and_persists():
    import tempfile
    from pathlib import Path
    from agent.memory import Memory, note
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = Path(tmp) / "memory.sqlite"
        m = Memory("stub", db, embed=word_embed)
        m.add("Priya's birthday is on June 3", "fact")
        m.add("Aalok's gym days are Tuesday and Saturday", "fact")
        m.add("Aalok asked to play lofi | NIA played lofi beats", "exchange")
        hits = m.search("when is priya's birthday", k=4, min_similarity=0.3)
        assert [h[3] for h in hits] == ["Priya's birthday is on June 3"], hits
        assert m.search("how far is the moon", min_similarity=0.3) == [], "off-topic turns get no memories"
        assert "June 3" in note(hits) and note([]) == ""
        m.db.close()

        m = Memory("stub", db, embed=word_embed)  # a restart: memories are still there
        assert m.search("priya birthday", min_similarity=0.3)
        assert m.forget("lofi") == [], "forget only removes facts, never the conversation log"
        assert m.forget("priya's birthday") == ["Priya's birthday is on June 3"]
        assert not m.search("priya birthday", min_similarity=0.3) and len(m.rows) == 2
        m.db.close()

        # The same question asked five times is recalled once, leaving room for something else relevant
        m = Memory("stub", Path(tmp) / "dupes.sqlite", embed=word_embed)
        for answer in ("standing by sir", "standing by sir all quiet", "standing by sir nothing much",
                       "standing by sir ready", "standing by sir as ever"):
            m.add(f"Aalok asked what are you doing right now | NIA answered {answer}", "exchange")
        m.add("Aalok asked what are you doing tonight | NIA answered your gym day is Tuesday", "exchange")
        hits = m.search("what are you doing right now", k=4, min_similarity=0.2)
        assert sum("right now" in h[3] for h in hits) == 1 and any("tonight" in h[3] for h in hits), hits
        m.db.close()

        # Exchanges that only made sense in the moment aren't kept
        from agent.chat import worth_remembering
        assert worth_remembering("What are you doing right now?")
        assert not worth_remembering("I guess so.") and not worth_remembering("I mean, could you...")

        tries = []
        m = Memory("stub", db, embed=lambda texts: tries.append(1) or (_ for _ in ()).throw(OSError("ollama down")))
        assert m.search("gym days") == [] and m.add("x", "fact") is None, "no embeddings: memory steps aside"
        assert len(tries) == 1, "after one refusal it stops asking for a while - each try cost ~2 s per turn"
        m.down_until = 0  # a minute later
        m.search("gym days")
        assert len(tries) == 2
        m.db.close()


def test_skills_are_saved_recalled_and_followed():
    # "Save that as my evening routine": the steps of a multi-step job, by name, to repeat it later
    import tempfile
    from pathlib import Path
    from agent.memory import Memory
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        m = Memory("stub", Path(tmp) / "memory.sqlite", embed=word_embed)
        save_skill, list_skills, forget = (next(t for t in m.tools() if t.name == n)
                                           for n in ("save_skill", "list_skills", "forget"))
        steps = ["Look up tonight's weather in Mumbai", "Play a chill playlist that suits it",
                 "Set the Spotify volume to 30 percent"]
        assert save_skill.invoke({"name": "evening routine", "when": "Aalok asks for his evening routine",
                                  "steps": steps}) == 'Saved the skill "evening routine" (3 steps).'
        assert m.skills() == ['Skill "evening routine" - use when Aalok asks for his evening routine. Steps: '
                              "1. Look up tonight's weather in Mumbai. 2. Play a chill playlist that suits it. "
                              "3. Set the Spotify volume to 30 percent."]
        # Saving under the same name updates it, not a second copy
        save_skill.invoke({"name": "Evening Routine", "when": "Aalok asks for his evening routine",
                           "steps": steps + ["Save a note evening.txt saying what was played"]})
        assert len(m.skills()) == 1 and "4. Save a note" in m.skills()[0]
        save_skill.invoke({"name": "morning news", "when": "Aalok asks for the morning news",
                           "steps": ["Search the web for today's top headlines", "Read the top two"]})
        assert list_skills.invoke({}) == "Saved skills: evening routine, morning news"
        assert "needs its steps" in save_skill.invoke({"name": "empty", "when": "never", "steps": []})

        # "Do my evening routine": the skill comes back with the request, for her to follow as her plan
        seen = []

        class Recording(ScriptedLLM):
            def _generate(self, messages, *args, **kwargs):
                seen.append(messages)
                return super()._generate(messages, *args, **kwargs)
        a = assistant(memory=m, llm=Recording(messages=iter([AIMessage("Right away, sir.")])))
        a.memory_min = 0.2
        a.respond("do my evening routine")
        request = [msg for msg in seen[-1] if isinstance(msg, HumanMessage)][-1].content
        assert "[From memory" in request and 'skill "evening routine"' in request.lower() and "morning news" not in request

        assert forget.invoke({"about": "evening routine"}).lower().startswith('forgot: skill "evening routine"')
        assert list_skills.invoke({}) == "Saved skills: morning news"
        m.db.close()


def test_only_recent_turns_reach_the_model_and_memory_fills_in():
    import tempfile
    from pathlib import Path
    import settings
    from agent.memory import Memory
    seen = []

    class Recording(ScriptedLLM):
        def _generate(self, messages, *args, **kwargs):
            seen.append(messages)
            return super()._generate(messages, *args, **kwargs)

    turns = settings.load()["history_turns"]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        m = Memory("stub", Path(tmp) / "memory.sqlite", embed=word_embed)
        total = 2 * turns + 2  # enough for one step cut
        replies = [AIMessage(f"reply {i}") for i in range(total)]
        a = assistant(memory=m, llm=Recording(messages=iter(replies)))
        a.memory_min = 0.3
        a.respond("my favourite colour is teal")
        for i in range(total - 2):
            a.respond(f"filler question number {i}")
        a.respond("what is my favourite colour")
        sizes = [sum(isinstance(msg, HumanMessage) for msg in call) for call in seen]
        assert max(sizes) <= 2 * turns, f"the model saw {max(sizes)} user turns; the cap is {2 * turns}"
        humans = [msg for msg in seen[-1] if isinstance(msg, HumanMessage)]
        assert turns <= len(humans) < 2 * turns, len(humans)
        # Cut in steps, so the opening of the prompt (llama-server's cache) changes only every `turns` turns
        firsts = [str(call[0].content) for call in seen]  # the planning middleware makes it a list of blocks
        assert len(set(firsts)) <= 1 + (total - 2 * turns + turns - 1) // turns, "the cut point moves every turn"
        assert not any("favourite colour is teal" in str(msg.content).split("[From memory")[0] for msg in humans)
        assert "favourite colour is teal" in str(humans[-1].content), "the old turn should come back as a memory"
        m.db.close()


def test_read_only_allowlist():
    from agent.approval import read_only
    for command in ["tasklist | findstr /i spotify", 'dir "C:\\Users\\me\\OneDrive\\Desktop"', "ipconfig", "whoami",
                    'type "C:\\notes.txt" | more', "git -C D:\\nia status", "git log", "netstat -ano | findstr 8081",
                    'powershell -NoProfile -Command "Get-PSDrive -PSProvider FileSystem"',
                    'powershell -NoProfile -Command "Get-Process | Where-Object { $_.CPU -gt 10 } | Sort-Object CPU"']:
        assert read_only(command), command
    for command in ["tasklist & shutdown /s /t 0", "ipconfig && del x", "dir > listing.txt", "type a.txt | python",
                    "git branch -D main", "git push", "notepad", "python backup.py", "echo hi", "start outlook",
                    'powershell -NoProfile -Command "Get-Process chrome | Stop-Process"',
                    'powershell -NoProfile -Command "Get-ChildItem; Remove-Item x"',
                    'powershell -NoProfile -Command "Get-Content $(Invoke-WebRequest x)"',
                    "powershell -NoProfile -Command \"(New-Object -ComObject WScript.Shell).SendKeys('^w')\""]:
        assert not read_only(command), command


def test_approval_layers():
    from agent import approval
    run = lambda command: {"name": "execute", "args": {"command": command}}
    calls = []

    def model_says(level):
        def hazards(command, model):
            calls.append(command)
            return None if level is None else {name: level for name in approval.HAZARDS}
        return hazards

    approval.hazards = model_says(0.0)  # the model thinks everything is harmless...
    assert approval.decide(run("taskkill /F /IM chrome.exe"), "m", 0.5)[0] == "ask"  # ...but risks always ask
    assert approval.decide(run("winget install VLC"), "m", 0.5)[0] == "ask"
    assert approval.decide(run("powershell \"...SendKeys('^w')\""), "m", 0.5)[0] == "ask"
    assert approval.decide({"name": "edit_file", "args": {"file_path": "/a.txt"}}, "m", 0.5)[0] == "ask"
    assert calls == [], "the model must never be asked about risky commands or file changes"
    assert approval.decide(run("ipconfig"), "m", 0.5) == ("run", "read-only command") and calls == []
    assert approval.decide(run("explorer D:\\nia"), "m", 0.5)[0] == "run"  # unknown: the model decides

    # Too small to ask: a new folder or a new file in your own folders (a real session asked "I'll run mkdir.
    # Should I go ahead?" and "I'll save coin_flip.py?"). Overwrites, edits and the system's places still ask
    import tempfile
    from pathlib import Path
    home = Path.home()
    new_file = lambda path: approval.decide({"name": "write_file", "args": {"file_path": path}}, "m", 0.5)[0]
    for path in ("/c/Users/x/../" + home.name + "/Documents/nia-tools/new_idea_42.py", "Desktop/shopping_42.txt",
                 "/d/projects/notes_42.txt"):
        assert new_file(path) == "run", path
    with tempfile.NamedTemporaryFile(dir=home, suffix=".txt", delete=False) as existing:
        pass
    try:
        assert new_file(existing.name) == "ask", "overwriting an existing file asks"
    finally:
        Path(existing.name).unlink()
    own = approval.NIA_ROOT  # wherever NIA's code lives (D:\nia here, elsewhere in a test checkout)
    for path in ("/c/Users/" + home.name + "/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup/x.bat",
                 "/c/Windows/x.txt", "/c/Program Files/x.txt", ".ssh/authorized_keys", "/c/temp/x.txt",
                 str(own / "agent" / "new_tool.py"), str(own.parent / "nia-changes" / "x" / "y.py"),
                 str(own / ".." / own.name / "agent" / "sneaky.py")):
        assert new_file(path) == "ask", path
    assert approval.decide(run(r"mkdir C:\Users\%s\Documents\nia-tools 2>&1 && echo created" % home.name),
                           "m", 0.5) == ("run", "a new folder in your folders")
    assert approval.decide(run(r'md "D:\new projects"'), "m", 0.5)[0] == "run"
    for command in (r"mkdir C:\Windows\x", r"mkdir C:\Users\x\AppData\y", r"mkdir D:\x && del D:\y",
                    r"mkdir D:\x & shutdown /s"):
        assert approval.decide(run(command), None, 0.5)[0] == "ask", command
    approval.hazards = model_says(0.9)
    assert approval.decide(run("explorer D:\\nia"), "m", 0.5)[0] == "ask"
    approval.hazards = model_says(None)  # Ollama down
    assert approval.decide(run("explorer D:\\nia"), "m", 0.5) == ("ask", "decision model unavailable")
    assert approval.decide(run("explorer D:\\nia"), "", 0.5) == ("ask", "no decision model set")

    # In the agent: a cleared command runs with no question; an uncleared one is asked about
    a = assistant(call("execute", command="tasklist"), AIMessage("Spotify isn't running, sir."))
    assert a.respond("is spotify running?") == "Spotify isn't running, sir." and a.pending is None
    approval.hazards = model_says(0.9)
    a = assistant(call("execute", command="explorer D:\\nia"), AIMessage("Opened it, sir."), approval_model="m")
    assert "Should I go ahead?" in a.respond("open the nia folder") and a.pending


def test_approval_question_says_what_not_how():
    q = chat.approval_question
    close_tab = {"name": "execute", "args": {"command": "powershell -NoProfile -Command \"...SendKeys('%w')\""}}
    # The model's own words are what's heard - never the command
    assert q([close_tab], "I'll close the YouTube tab.") == \
        "Before I do that: I'll close the YouTube tab. Note that it presses keys in another window. Should I go ahead?"
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
    assert run("C:\\Users\\me\\Downloads\\jo.exe") == "Before I do that: I'll run jo.exe. Should I go ahead?"  # was "run C"
    assert run('"C:\\Program Files\\App\\app.exe" --flag').endswith("I'll run app.exe. Should I go ahead?")
    assert run('start "https://news.google.com/"') == "Before I do that: I'll open https://news.google.com/. Should I go ahead?"
    # Every way people say yes counts; anything hedged doesn't
    for answer in ("Do that.", "Do it.", "Sure, go ahead", "Yes please", "Go for it", "Okay", "Alright", "Of course",
                   "Go on", "Continue", "Proceed", "Carry on", "Sounds good", "Fine", "Why not"):  # "Go on" was a no once
        assert chat.YES.search(answer.lower()) and not chat.NO.search(answer.lower()), answer
    for answer in ("No, don't", "Wait", "Yes, no wait", "Hmm", "Not fine", "I'm not sure", "Please don't"):
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


def test_warm_up_leaves_no_trace():
    # At startup, so the first real reply doesn't spend ~10 s reading the prompt: it may not act, or join the chat
    from langchain_core.tools import tool
    done = []

    @tool
    def open_news() -> str:
        """Open a news video"""
        done.append(1)
        return "opened"

    a = assistant(call("open_news"), AIMessage("Hello, sir."), AIMessage("Evening, sir."), extra_tools=[open_news])
    a.warm_up()
    assert done == [], "the warm-up must not do anything"
    assert a.messages == [] and not chat.REFUSED.is_set()
    assert a.respond("hello") == "Evening, sir." and done == []


def written(a, path):
    return path in (a.agent.get_state(a.config).values.get("files") or {})


def test_pc_changes_need_a_spoken_yes():
    # Approved: the read-back names the file, and the write happens only after "yes"
    # (outside your own folders - AppData - since a new file in them no longer asks)
    a = assistant(call("write_file", file_path="/AppData/notes.txt", content="milk"), AIMessage("Saved, sir."))
    question = a.respond("note down milk")
    assert "save notes.txt" in question and "go ahead" in question.lower() and "/AppData" not in question
    assert not written(a, "/AppData/notes.txt"), "wrote before the user said yes"
    assert a.respond("yes, go ahead") == "Saved, sir."
    assert written(a, "/AppData/notes.txt")

    # Refused: anything but a clear yes, including "yes... no wait"
    for answer in ("no", "nope, cancel that", "yes - no wait", "what?"):
        a = assistant(call("execute", command="del C:\\stuff"), AIMessage("Okay, I won't, sir."))
        question = a.respond("clean up")
        assert "del C:" not in question and "it deletes files" in question, question
        assert a.respond(answer) == "Okay, I won't, sir.", answer
        tool_msg = next(m for m in a.messages if isinstance(m, ToolMessage))
        assert tool_msg.status == "error" or "reject" in str(tool_msg.content).lower(), tool_msg
        # A no is reported as a no; an unclear answer as "held off" - never "you declined" when nobody did
        if answer == "what?":
            assert "wasn't a clear yes" in str(tool_msg.content) and "said no" not in str(tool_msg.content)
        else:
            assert "The user said no" in str(tool_msg.content)

    # "Go on" is a yes: it runs (a real session heard it as "you declined")
    a = assistant(call("execute", command="del C:\\stuff"), AIMessage("Done, sir."))
    a.respond("clean up")
    assert a.respond("Go on") == "Done, sir." and a.pending is None

    # Stale: once the question expires, the next sentence is a new request, not the answer
    a = assistant(call("write_file", file_path="/AppData/old.txt", content="x"), AIMessage("Left it alone, sir."),
                  AIMessage("Evening, sir."))
    a.respond("write old.txt")
    count, asked = a.pending
    a.pending = (count, asked - chat.APPROVAL_EXPIRES - 1)
    assert a.respond("yes, hello") == "Evening, sir."  # answered as a new turn
    assert not written(a, "/AppData/old.txt"), "a late yes approved a stale action"
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


@without_waiting
def test_play_something_opens_spotify_and_picks_from_preferences():
    import tempfile
    from pathlib import Path
    import settings
    from agent import action_controller
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

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
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


@without_waiting
def test_spotify_playback_fixes_from_a_real_session():
    import spotipy
    from agent import action_controller

    class StubAPI:
        def __init__(self, plays=True, ghost_device=False):
            self.calls, self.plays, self.ghost = [], plays, ghost_device
        def devices(self):
            return {"devices": [{"id": "fresh-laptop", "type": "Computer", "is_active": False}]}
        def search(self, q, limit, type):
            found = {"tracks": {"items": [{"name": "Thunderstruck", "uri": "spotify:track:t", "artists": [{"name": "AC/DC"}],
                                           "album": {"uri": "spotify:album:a"}}]},
                     "playlists": {"items": [None] if q == "empty mood" else [{"name": f"{q} mix", "uri": "spotify:playlist:p",
                                                                                  "tracks": {"total": 10}}]}}
            return found
        def start_playback(self, **kwargs):
            self.calls.append(kwargs)
            if self.ghost and len(self.calls) == 1:
                raise spotipy.SpotifyException(404, -1, "Device not found")
        def current_playback(self):
            return {"item": {"name": "x"} if self.plays else None, "is_playing": self.plays, "device": {"name": "ALOKLT"}}
        def shuffle(self, state):
            pass

    def controller(api):
        c = object.__new__(SpotifyController)  # skip OAuth
        c.sp = api
        return c

    # A track plays inside its album, starting at that track - a lone track stopped starting on this PC
    c = controller(StubAPI())
    assert c.play_track("thunderstruck") == "Successfully playing: Thunderstruck by AC/DC"
    assert c.sp.calls[0]["context_uri"] == "spotify:album:a" and c.sp.calls[0]["offset"] == {"uri": "spotify:track:t"}
    assert "uris" not in c.sp.calls[0]

    # Nothing starts: offer a restart once - but never again after one (the session restarted five times)
    c = controller(StubAPI(plays=False))
    for already_restarted, expect in ((False, "offer to restart"), (True, "Don't restart it again")):
        if already_restarted:
            c.restarted = action_controller.time.time()
        try:
            c.play_track("thunderstruck")
            raise AssertionError("claimed success")
        except RuntimeError as e:
            assert expect in str(e), str(e)

    # A device id that went stale with a restart (404) is looked up again, once
    c = controller(StubAPI(ghost_device=True))
    c.play_track("thunderstruck")
    assert len(c.sp.calls) == 2 and c.sp.calls[1]["device_id"] == "fresh-laptop"

    # play_something: an empty search for one mood falls back to another instead of giving up
    import tempfile
    from pathlib import Path
    import settings
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings.PATH = Path(tmp) / "settings.json"
        settings.save({**settings.DEFAULTS, "music_moods": "empty mood, rainy jazz"})
        c = controller(StubAPI())
        assert "rainy jazz mix" in c.play_something()

        # Playing, but Spotify 404s the shuffle (a real session): still a success, not "Spotify isn't responding"
        class NoShuffle(StubAPI):
            def shuffle(self, state):
                raise spotipy.SpotifyException(404, -1, "Not found.")
        reply = controller(NoShuffle()).play_something()
        assert reply.startswith("Playing the playlist") and "random track" in reply


def test_a_no_stops_other_actions_that_turn():
    from langchain_core.tools import tool
    done = []

    @tool
    def open_news() -> str:
        """Open a news video"""
        done.append("news video")
        return "opened"

    a = assistant(call("execute", command="start https://news.google.com"), call("open_news"),
                  AIMessage("Okay, sir. What would you like instead?"), call("open_news"), AIMessage("Here you go."),
                  extra_tools=[open_news])
    assert "Should I go ahead?" in a.respond("what's the news")
    assert a.respond("no") == "Okay, sir. What would you like instead?"
    assert done == [], "after a no, she opened something else instead"
    blocked = [m for m in a.messages if isinstance(m, ToolMessage) and m.name == "open_news"]
    assert blocked and "the user said no" in blocked[0].content.lower()
    assert a.respond("ok, open the news video") == "Here you go." and done == ["news video"], "a new request is fine"


def test_sub_agents_follow_the_same_rules():
    # Live, Bonsai handed "close the browser" to a sub-agent: its steps need the same yes, and the same no
    from langchain_core.tools import tool
    done = []

    @tool
    def open_news() -> str:
        """Open a news video"""
        done.append("news video")
        return "opened"

    def delegate(*sub_steps, after):
        return [call("task", description="Save a note saying hi", subagent_type="general-purpose"), *sub_steps, after]

    # Its file change asks first, and a yes lets it finish
    a = assistant(*delegate(call("write_file", file_path="/AppData/notes.txt", content="hi"), AIMessage("Saved."),
                            after=AIMessage("Done, sir.")))
    assert "Should I go ahead?" in a.respond("save a note saying hi"), "a sub-agent's file change must ask first"
    assert a.respond("yes") == "Done, sir."
    assert any(isinstance(m, ToolMessage) and m.name == "task" and m.content == "Saved." for m in a.messages)

    # A no: the paused sub-agent never resumes, and nothing else runs in its place
    a = assistant(*delegate(call("write_file", file_path="/AppData/notes.txt", content="hi"),
                            after=call("open_news")), AIMessage("Okay, sir."), extra_tools=[open_news])
    a.respond("save a note saying hi")
    assert a.respond("no") == "Okay, sir." and done == []

    # "Stop" halts it between its own steps too, not only between the main agent's
    @tool
    def slow_step() -> str:
        """A step that's under way when the user says stop"""
        chat.CANCEL.set()
        return "halfway"

    a = assistant(*delegate(call("slow_step"), call("open_news"), AIMessage("Finished."), after=AIMessage("Done.")),
                  extra_tools=[slow_step, open_news])
    try:
        a.respond("save a note saying hi")
    finally:
        chat.CANCEL.clear()
    assert done == [], "the sub-agent kept going after stop"


def test_web_search_and_reading():
    from agent import web
    # DuckDuckGo's lite page, as it answers (trimmed): an ad, then results behind its redirect links
    lite = """
      <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fduckduckgo.com%2Fy.js%3Fad%3D1&amp;rut=x" class='result-link'>Buy octopus plushies</a>
      <td class='result-snippet'>Sponsored.</td>
      <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.scienceabc.com%2Foctopus&amp;rut=y" class='result-link'>Octopus Hearts &amp; Blood</a>
      <td class='result-snippet'>Its systemic <b>heart</b> stops beating when it swims.</td>
      <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fb&amp;rut=z" class='result-link'>Second</a>
      <td class='result-snippet'>Two branchial hearts keep going.</td>"""
    real_get = web._get
    web._get = lambda url, **kw: ("text/html", lite)
    try:
        assert web.search("octopus hearts") == [
            ("Octopus Hearts & Blood", "https://www.scienceabc.com/octopus", "Its systemic heart stops beating when it swims."),
            ("Second", "https://example.org/b", "Two branchial hearts keep going.")], "ads skipped, real addresses decoded"
        listing = web.web_search.invoke({"query": "octopus hearts"})
        assert listing.startswith("1. Octopus Hearts & Blood - https://www.scienceabc.com/octopus")

        # A page: its prose, without the menu, scripts and footer around it
        page = """<html><head><title>Octopus hearts</title><script>var tracking = 1;</script></head><body>
          <nav><a href="/">Home</a> <a href="/animals">Animals</a></nav>
          <h1>Why one heart stops</h1>
          <p>An octopus has three hearts, and its main heart stops beating when it swims by jet propulsion.</p>
          <p>The two branchial hearts keep pumping blood through the gills the whole time.</p>
          <footer>Copyright 2026 - all rights reserved - privacy policy - cookie settings</footer></body></html>"""
        web._get = lambda url, **kw: ("text/html", page)
        text = web.read_page.invoke({"url": "https://example.org/octopus"})
        assert text.startswith("Octopus hearts") and "three hearts" in text and "branchial" in text
        assert "tracking" not in text and "Animals" not in text and "Copyright" not in text
        web._get = lambda url, **kw: ("application/pdf", "")
        assert "not a page" in web.read_page.invoke({"url": "https://example.org/paper.pdf"})
        for bad in ["file:///C:/secret.txt", "C:\\notes.txt", "javascript:alert(1)"]:
            assert "Only http" in web.read_page.invoke({"url": bad}), bad
        web._get = lambda url, **kw: (_ for _ in ()).throw(OSError("offline"))
        assert "internet may be down" in web.web_search.invoke({"query": "x"})
    finally:
        web._get = real_get

    # The shell is no longer the way to the web: fetching from it always asks first (curl once ran unasked)
    from agent.approval import risks
    for command in ['curl -s "https://www.ncbi.nlm.nih.gov/pubmed/?term=octopus"', "wget http://x.org/a",
                    'powershell -NoProfile -Command "Invoke-WebRequest https://x.org"']:
        assert "it reaches the internet" in risks({"name": "execute", "args": {"command": command}}), command


def test_ask_claude_is_safe_and_reports_back():
    import json
    import time as clock
    from agent import claude
    from agent.approval import risks

    # Always the same locked-down call: web search and fetch only, anything else refused, nothing carried over
    cmd = claude.command("Who won the last Grand Prix?", "sonnet")
    assert cmd[cmd.index("--effort") + 1] == "low", "low effort: 17 s instead of 46 s for the same answer"
    assert "Always browse first" in cmd[2], "never an answer from memory alone"
    assert cmd[cmd.index("--output-format") + 1] == "stream-json" and "--verbose" in cmd
    for flag in ("-p", "--restricted", "--strict-mcp-config", "--no-session-persistence"):
        assert flag in cmd, flag
    assert cmd[cmd.index("--tools") + 1] == "WebSearch,WebFetch" and cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    assert not any("dangerously" in part or "bypass" in part for part in cmd)
    assert cmd[-2:] == ["--model", "sonnet"] and "--model" not in claude.command("q", "")
    real_env = dict(claude.os.environ)
    claude.os.environ.update({"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "parent"})
    try:
        assert not any(k == "CLAUDECODE" or k.startswith("CLAUDE_") for k in claude.clean_env()), \
            "a call made from inside a Claude Code session must be a session of its own"
    finally:
        claude.os.environ.clear()
        claude.os.environ.update(real_env)
    # The shell can't start Claude Code on its own terms
    for command in ["claude -p hi --dangerously-skip-permissions", r"C:\Users\x\.local\bin\claude.exe -p x"]:
        assert "it runs Claude Code" in risks({"name": "execute", "args": {"command": command}}), command
    assert not risks({"name": "execute", "args": {"command": "dir claudette"}})

    class FakeClaude:
        """Stands in for the claude process: answers with out after delay seconds, unless killed"""
        def __init__(self, out, delay=0.0):
            self.out, self.delay, self.killed, self.returncode = out, delay, False, 0
        def __call__(self, *args, **kwargs):
            return self
        def communicate(self):
            end = clock.time() + self.delay
            while clock.time() < end and not self.killed:
                clock.sleep(0.05)
            return self.out, ""
        def kill(self):
            self.killed = True

    real_popen, real_exe, real_timeout = claude.subprocess.Popen, claude.executable, claude.TIMEOUT
    claude.executable = lambda: "claude.exe"
    try:
        # Claude Code's event stream: the tools it used, then the result
        stream = [{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "WebSearch"}]}},
                  {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "WebFetch"}]}},
                  {"type": "result", "result": "Verstappen won in Bahrain, sir.\nSources: formula1.com",
                   "is_error": False, "total_cost_usd": 0.05}]
        out = "\n".join(json.dumps(event) for event in stream)
        assert [b["name"] for b in claude.tool_calls(out)] == ["WebSearch", "WebFetch"], "it really browsed"
        claude.subprocess.Popen = FakeClaude(out)
        announced = []
        claude.ANNOUNCE = lambda: announced.append(1)
        answer = claude.ask_claude.invoke({"question": "Who won?"})
        assert "pass it on faithfully" in answer and "Verstappen won in Bahrain" in answer and announced == [1]
        claude.ANNOUNCE = None
        claude.subprocess.Popen = FakeClaude(json.dumps({"type": "result", "result": "Not logged in", "is_error": True}))
        assert "couldn't answer" in claude.ask("Who won?") and "web_search" in claude.ask("Who won?")
        claude.subprocess.Popen = FakeClaude("Error: please run /login")
        assert "couldn't be reached" in claude.ask("Who won?")
        # "Stop" mid-call kills it; so does running out of time
        slow = FakeClaude("{}", delay=30)
        claude.subprocess.Popen = slow
        threading_timer = __import__("threading").Timer(0.3, claude.STOP.set)
        threading_timer.start()
        started = clock.time()
        assert claude.ask("Who won?") == "Stopped - the user said stop" and slow.killed and clock.time() - started < 3
        claude.STOP.clear()
        claude.TIMEOUT = 0.3
        claude.subprocess.Popen = FakeClaude("{}", delay=30)
        assert "didn't answer within" in claude.ask("Who won?")
        claude.executable = lambda: None
        assert "isn't installed" in claude.ask("Who won?")
    finally:
        claude.subprocess.Popen, claude.executable, claude.TIMEOUT = real_popen, real_exe, real_timeout
        claude.STOP.clear()
    assert claude.STOP is chat.CANCEL, "NIA's stop is the same flag"
    assert "ask_claude" in chat.READ_ONLY_TOOLS


def test_changes_to_herself_go_to_a_branch_and_only_if_tests_pass():
    import re
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    from agent import claude

    # Claude may only edit inside its checkout, never the safety files, and run nothing but the tests
    cmd = claude.build_command("add a joke tool")
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk" and "--restricted" in cmd
    shells = [c for c in cmd if c.startswith(("Bash(", "PowerShell("))]  # PowerShell is the shell on Windows
    assert len(shells) == 6 and all(c.endswith((" test_agent.py)", " test_voice.py)", " test_nia.py)")) for c in shells)
    for path in ("agent/approval.py", "agent/claude.py", "nia.py", "test_agent.py"):
        assert f"Edit({path})" in cmd and f"Write({path})" in cmd, path
    assert claude.branch_name("Integrate Discord into you, please!", 0).startswith("nia/integrate-discord-into-you-please-")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp) / "nia"
        root.mkdir()
        git = lambda *args, cwd=root: subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
        git("init", "-q")
        git("config", "user.email", "test@example.com")
        git("config", "user.name", "Test")
        (root / "app.py").write_text("print('nia')\n")
        git("add", "-A")
        git("commit", "-q", "-m", "start")
        (root / "notes.txt").write_text("the user's own uncommitted work")  # must stay exactly where it is

        said, reported = [], __import__("threading").Event()
        real = (claude.build_command, claude.run_tests, claude.CHANGES, claude.executable, claude.ON_DONE)
        claude.CHANGES, claude.executable = Path(tmp) / "changes", lambda: "claude"
        claude.ON_DONE = lambda text: (said.append(text), reported.set())

        def fake_claude(writes):
            """Stands in for Claude Code: writes a file (or not) in its checkout and reports back"""
            script = (f"import json, pathlib\nif {writes!r}: pathlib.Path('joke.py').write_text('JOKE = 1')\n"
                      "print(json.dumps({'type': 'result', 'result': 'I added a joke tool.'}))")
            return lambda request: [sys.executable, "-c", script]

        def run(request, writes, failing):
            claude.build_command, claude.run_tests = fake_claude(writes), lambda folder: failing
            said.clear()
            reported.clear()
            started = claude.improve(request, root=root)
            assert started.startswith("Started") and "nia/" in started
            assert reported.wait(60), "the job never reported back"  # it runs in the background
            return said[0]

        try:
            # Tests pass: committed to a new branch; the running checkout and the user's work untouched
            done = run("add a joke tool", writes=True, failing=[])
            branch = re.search(r"nia/[\w-]+", done).group(0)
            assert "ready for you to review" in done and "I added a joke tool." in done
            assert git("log", "-1", "--format=%s", branch).stdout.strip() == "NIA: add a joke tool"
            assert "joke.py" in git("show", "--name-only", "--format=", branch).stdout
            assert git("branch", "--show-current").stdout.strip() in ("master", "main"), "never switched"
            assert not (root / "joke.py").exists() and (root / "notes.txt").exists()
            assert not list((Path(tmp) / "changes").iterdir()), "its checkout is cleaned up; the branch stays"

            # Tests fail: discarded - no branch, no checkout
            done = run("add a broken tool", writes=True, failing=["test_agent.py"])
            assert "broke test_agent.py" in done and "nia/add-a-broken-tool" not in git("branch").stdout
            # Nothing changed: nothing kept
            done = run("do nothing", writes=False, failing=[])
            assert done.startswith("Claude made no change") and "nia/do-nothing" not in git("branch").stdout
        finally:
            claude.build_command, claude.run_tests, claude.CHANGES, claude.executable, claude.ON_DONE = real

    # The agent never starts one without a spoken yes, and says plainly what it will do
    started = []
    real_improve = claude.improve
    claude.improve = lambda request, root=None: started.append(request) or "Started: on a branch"
    try:
        a = assistant(call("improve_myself", request="integrate Discord"), AIMessage("It's under way, sir."))
        question = a.respond("ask Claude to integrate Discord into you")
        assert "have Claude integrate Discord, on a new branch for you to review" in question
        # A long, detailed request is read back as its first clause, not in full
        from agent.chat import short_request
        assert short_request('Add a "flip_coin" tool to NIA. When Aalok asks to flip a coin (or similar phrasing '
                             'like "heads or tails"), the tool uses random...') == "add a flip coin tool to me"
        assert short_request("Integrate Discord into NIA so she can read messages") ==             "integrate Discord into me so I can read messages"
        assert short_request("a weather widget") == "build a weather widget into me"
        assert "Should I go ahead?" in question and started == []
        assert a.respond("yes") == "It's under way, sir." and started == ["integrate Discord"]
    finally:
        claude.improve = real_improve


def test_a_plan_is_tracked_and_shown():
    # A multi-step job: she writes her plan, ticks it off, and the window gets each version of it
    sent = []
    real_send = chat.hud.send
    chat.hud.send = lambda **message: sent.append(message)
    steps = [{"content": "Search the web", "status": "completed"}, {"content": "Read two articles", "status": "in_progress"},
             {"content": "Save a summary", "status": "pending"}]
    try:
        a = assistant(call("write_todos", todos=steps), AIMessage("Done, sir. The summary is on your Desktop."))
        assert a.respond("research octopus hearts and save a summary") == "Done, sir. The summary is on your Desktop."
    finally:
        chat.hud.send = real_send
    plans = [m["plan"] for m in sent if "plan" in m]
    assert plans == [[{"step": "Search the web", "status": "completed"},
                      {"step": "Read two articles", "status": "in_progress"},
                      {"step": "Save a summary", "status": "pending"}]], sent
    assert "write_todos" in chat.READ_ONLY_TOOLS, "planning isn't an action: a no doesn't block it"


def test_open_link_only_opens_web_addresses():
    from agent import youtube
    opened = []
    youtube.webbrowser.open = opened.append
    youtube.last_opened = 0.0
    assert "Opened www.google.com" in youtube.open_link.invoke({"url": "https://www.google.com"})
    assert opened == ["https://www.google.com"]
    for bad in ["file:///C:/Windows/System32/cmd.exe", "C:\\evil.exe", "javascript:alert(1)", "notepad"]:
        youtube.last_opened = 0.0
        assert "only http" in youtube.open_link.invoke({"url": bad}) and len(opened) == 1, bad
    # One tab per 15 s, shared with YouTube: a link and then another link or a video inside it are refused
    youtube.last_opened = 0.0
    assert "Opened a.com" in youtube.open_link.invoke({"url": "https://a.com"})
    assert "Not opened" in youtube.open_link.invoke({"url": "https://b.com"})
    youtube.search = lambda q, count=1: [("vid00000000", "A video", "A channel")]
    assert "Not opened" in youtube.play_youtube.invoke({"query": "lofi"})
    assert opened == ["https://www.google.com", "https://a.com"]


@without_waiting
def test_restart_spotify_only_touches_spotify():
    from agent import action_controller
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


@without_waiting
def test_playback_wakes_an_idle_device_and_checks_it_started():
    from agent import action_controller
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
    # Every test_ function, in the order defined - a list kept by hand once left six of them out
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
    print("ok")
