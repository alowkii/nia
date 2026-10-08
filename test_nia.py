"""Checks nia.py, NIA's window: the page and its live state stream, typed commands and mic presses reaching the
assistant, settings saved and applied (at once, or by restarting what needs it), and that no other web page can
drive her. The LLM server and assistant are stand-ins, settings go to a temporary file. Run: python test_nia.py"""
import json
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

import nia
import settings


class Recorder:
    """Stands in for nia.Server / nia.Assistant: records what it was asked to do"""
    def __init__(self):
        self.sent, self.restarts = [], 0

    def send(self, message):
        self.sent.append(message)

    def restart(self):
        self.restarts += 1


with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    settings.PATH = Path(tmp) / "settings.json"  # never touch the real settings.json
    hub, done = nia.Hub(), threading.Event()
    assistant, server = Recorder(), Recorder()
    httpd = nia.HudServer(("127.0.0.1", 0), nia.handler(hub, assistant, server, done))  # a free port
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def post(path, data=None, header=True):
        request = urllib.request.Request(base + path, json.dumps(data or {}).encode(), method="POST",
                                         headers={"X-NIA": "1"} if header else {})
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read() or b"null")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"null")

    # The page, and its stream: the latest state on connecting, then each change as it happens
    assert b'id="orb"' in urllib.request.urlopen(base + "/").read()
    hub.send(state="thinking", you="play lofi", tool="play")
    stream = urllib.request.urlopen(base + "/events", timeout=5)

    def next_event():
        while not (line := stream.readline()).startswith(b"data: "):
            pass
        return json.loads(line[6:])

    first = next_event()
    assert first["state"] == "thinking" and first["you"] == "play lofi" and first["tool"] == "play"
    assert first["config"]["theme"] == "cyan" and first["config"]["wake_word"] is True
    hub.send(state="speaking", envelope=[0.5, 1.0], fps=60)
    assert next_event() == {"state": "speaking", "envelope": [0.5, 1.0], "fps": 60}
    assert "envelope" not in hub.latest, "a page opening mid-sentence mustn't replay an old sentence's loudness"

    # Typing and the mic button reach the assistant
    assert post("/command", {"text": "what time is it"})[0] == 204
    assert post("/mic", {"action": "wake"})[0] == 204
    assert assistant.sent == [{"type": "text", "text": "what time is it"}, {"type": "wake"}]
    post("/mic", {"action": "restart"})

    # No other web page can drive her: they can't send the header without a preflight that's never answered
    for path in ("/command", "/mic", "/settings", "/quit", "/server/restart", "/spotify/login"):
        assert post(path, {"text": "delete everything"}, header=False)[0] == 403, path

    # Settings: the form's strings become the right types; bad values are refused with a reason
    values = json.load(urllib.request.urlopen(base + "/settings"))
    assert values["values"]["voice_speed"] == settings.DEFAULTS["voice_speed"] and values["fields"]
    assert post("/settings", {"voice_speed": "fast"}) == (400, {"error": "voice_speed must be a number, got 'fast'"})
    assert post("/settings", {"voice_speed": "5"})[1]["error"] == "voice_speed must be between 0.5 and 2"
    assert post("/settings", {"no_such": 1})[0] == 400
    assert settings.load()["voice_speed"] == settings.DEFAULTS["voice_speed"], "a refused save changes nothing"

    # Live settings apply at once, with no restart
    assistant.sent.clear()
    assert post("/settings", {"voice_speed": "1.4", "spoken_replies": False}) == (200, {"restarting": []})
    assert settings.load()["voice_speed"] == 1.4 and settings.load()["spoken_replies"] is False
    assert assistant.sent == [{"type": "settings", "values": {"voice_speed": 1.4, "spoken_replies": False}}]
    assert hub.latest["config"]["spoken_replies"] is False, "every open page hears about it"
    # Others restart the assistant; server ones restart the server too
    assert post("/settings", {"voice": "kokoro_bm_fable"}) == (200, {"restarting": ["assistant"]})
    assert post("/settings", {"context": "8192"}) == (200, {"restarting": ["server", "assistant"]})
    assert post("/settings", {"context": "8192"}) == (200, {"restarting": []}), "saving unchanged values restarts nothing"
    import time
    time.sleep(0.2)  # restarts run on their own threads
    assert server.restarts == 1 and assistant.restarts == 3  # the mic button's restart, the voice, the context

    # Only NIA's own llama-server counts: Ollama runs its own llama-server.exe, and waiting on that one
    # left a real launch stuck on "Booting" (and Restart server would have killed it)
    from types import SimpleNamespace
    ours = SimpleNamespace(info={"name": "llama-server.exe", "exe": nia.llm_server_command()[0]})
    ollamas = SimpleNamespace(info={"name": "llama-server.exe",
                                    "exe": r"C:\Users\x\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe"})
    real = nia.psutil.process_iter
    nia.psutil.process_iter = lambda attrs: [ollamas]
    assert nia.bonsai_servers() == []
    nia.psutil.process_iter = lambda attrs: [ollamas, ours]
    assert nia.bonsai_servers() == [ours]
    nia.psutil.process_iter = real

    # The window's quit
    assert post("/quit")[0] == 204 and done.is_set()
    httpd.shutdown()

print("ok")
