"""Checks nia.py, NIA's window: the page and its live state stream, typed commands and mic presses reaching the
assistant, settings saved and applied (at once, or by restarting what needs it), and that no other web page can
drive her. The LLM server and assistant are stand-ins, settings go to a temporary file. Run: python tests/test_nia.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # NIA's code, one folder up

import json
import tempfile
import threading
import urllib.error
import urllib.request

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

    # The microphones on offer, for the picker beside the mic button and the settings form
    listed = json.load(urllib.request.urlopen(base + "/mics"))
    assert listed["mics"] == nia.mics.names() and listed["current"] == ""
    assert json.load(urllib.request.urlopen(base + "/settings"))["mics"] == listed["mics"]
    assistant.sent.clear()
    assert post("/settings", {"mic_device": "Microphone (Test)"}) == (200, {"restarting": []}), "switches live"
    assert assistant.sent == [{"type": "settings", "values": {"mic_device": "Microphone (Test)"}}]
    assert hub.latest["config"]["mic_device"] == "Microphone (Test)"

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

    # She asks to be restarted (restart_myself): just her...
    import threading
    import time
    reboot, restarted = threading.Event(), []
    child = nia.Assistant(hub, server=Recorder(), done=reboot)
    child.restart = lambda: restarted.append(1)
    assert child.handle({"restart": "assistant", "state": "speaking"}) == {"state": "speaking"}, "the rest goes on"
    for _ in range(50):
        if restarted:
            break
        time.sleep(0.02)
    assert restarted == [1] and not reboot.is_set() and not child.rebooting
    # ...or a reboot: main() shuts all of NIA down and hands over to a fresh copy of nia.py
    assert child.handle({"restart": "all"}) == {}
    assert reboot.is_set() and child.rebooting and restarted == [1]
    assert hub.latest["rebooting"] is True, "the page shows booting, not offline, while nia.py is away"
    # ...or shut down: main() stops everything, starts no new copy, and the page closes its own window
    off = threading.Event()
    stopped = nia.Assistant(hub, server=Recorder(), done=off)
    assert stopped.handle({"restart": "off"}) == {} and off.is_set() and not stopped.rebooting
    assert hub.latest["state"] == "offline" and hub.latest["closing"] is True
    assert nia.Assistant(hub).handle({"state": "awake"}) == {"state": "awake"}

    # The logo: the window's icon and header, served from assets/logo only
    page = urllib.request.urlopen(base + "/").read().decode("utf-8")
    assert 'href="/logo/svg/nia-favicon.svg"' in page and 'href="/favicon.ico"' in page
    assert 'class="brand" role="img" aria-label="NIA"' in page
    for path, kind in (("/favicon.ico", "image/x-icon"), ("/logo/svg/nia-favicon.svg", "image/svg+xml"),
                       ("/logo/png/apple-touch-icon-180.png", "image/png")):
        with urllib.request.urlopen(base + path) as response:
            assert response.headers["Content-Type"] == kind and response.read() == nia.logo_file(path).read_bytes()
    # ...and nothing else on the PC, however the path is dressed up
    for path in ("/logo/../nia.py", "/logo/../../.env", "/logo/svg/../../../settings.json", "/logo/%2e%2e/nia.py",
                 "/logo/nope.svg", "/logo/"):
        try:
            urllib.request.urlopen(base + path)
            raise AssertionError(f"served {path}")
        except urllib.error.HTTPError as e:
            assert e.code == 404, path

    # The window's quit
    assert post("/quit")[0] == 204 and done.is_set()
    httpd.shutdown()

# Closing NIA closes everything she started - real processes here, not stand-ins. After one Ctrl+C an assistant
# lingered 10+ minutes and the model server (~6 GB of GPU memory) never stopped
import subprocess
import sys
import time
import psutil

SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]
FAMILY = [sys.executable, "-c", "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', "
          "'import time; time.sleep(60)']); time.sleep(60)"]


def gone_soon(pid, seconds=10):
    end = time.time() + seconds
    while time.time() < end:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.1)
    return False


# kill_tree: the process and everything under it (.venv's python.exe is a launcher with the real Python under it)
parent = subprocess.Popen(FAMILY)
for _ in range(100):
    family = psutil.Process(parent.pid).children(recursive=True)
    if len(family) >= 2 if "venv" in sys.executable.lower() else family:
        break
    time.sleep(0.05)
family = [parent.pid] + [c.pid for c in psutil.Process(parent.pid).children(recursive=True)]
nia.kill_tree(parent.pid)
assert all(gone_soon(pid) for pid in family), "the whole tree, not just the top"

# bound_to_me: a process that owns a server dies abruptly - no cleanup at all - and the server goes with it
owner = subprocess.Popen([sys.executable, "-c", f"""
import os, subprocess, sys
sys.path.insert(0, {str(nia.ROOT)!r})
import nia
server = subprocess.Popen({SLEEPER!r})
nia.bound_to_me(server)
print(server.pid, flush=True)
os._exit(0)  # like a closed console or Task Manager
"""], stdout=subprocess.PIPE, text=True, cwd=nia.ROOT)
server_pid = int(owner.stdout.readline())
owner.wait(30)
try:
    assert gone_soon(server_pid), "the server outlived the NIA that started it"
finally:
    nia.kill_tree(server_pid)

# leftovers: an assistant whose nia.py is gone is found; one with a living parent (run by hand) is left alone
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    (Path(tmp) / "main.py").write_text("import time; time.sleep(60)")
    real_root, nia.ROOT = nia.ROOT, Path(tmp).resolve()
    kept = subprocess.Popen([sys.executable, "main.py"], cwd=tmp)  # its parent - this test - is alive
    gone_parent = subprocess.Popen([sys.executable, "-c", "import subprocess, sys; "
                                    "print(subprocess.Popen([sys.executable, 'main.py']).pid, flush=True)"],
                                   cwd=tmp, stdout=subprocess.PIPE, text=True)
    orphan_pid = int(gone_parent.stdout.readline())
    gone_parent.wait(30)
    try:
        time.sleep(1)  # the launchers start their real Pythons
        found = [proc.pid for proc in nia.leftovers()]
        assert orphan_pid in found, "an assistant whose parent is gone"
        assert kept.pid not in found, "never one with a living parent"
        assert not any(c.pid in found for c in psutil.Process(kept.pid).children()), "nor its real Python"
    finally:
        nia.ROOT = real_root
        nia.kill_tree(orphan_pid)
        nia.kill_tree(kept.pid)

print("ok")
