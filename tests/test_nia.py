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
    ours = SimpleNamespace(info={"name": f"llama-server{nia.EXE}", "exe": nia.llm_server_command()[0]})
    ollamas = SimpleNamespace(info={"name": f"llama-server{nia.EXE}",
                                    "exe": r"C:\Users\x\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe"
                                    if nia.WINDOWS else "/usr/lib/ollama/llama-server"})
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
    assert hub.latest["state"] == "offline"
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

    # Spotify login with no Spotify app in .env: the panel is told why, instead of a button that does nothing
    import os
    real_id = os.environ.get("SPOTIFY_ID")
    os.environ["SPOTIFY_ID"] = ""  # kept over whatever .env says (load_dotenv never overrides)
    try:
        code, body = post("/spotify/login")
        assert code == 400 and "SPOTIFY_ID and SPOTIFY_SECRET in .env" in body["error"], (code, body)
    finally:
        if real_id is None:
            os.environ.pop("SPOTIFY_ID")
        else:
            os.environ["SPOTIFY_ID"] = real_id

    # Spotify sends the browser back to NIA's own server, which finishes the login - only one started from her panel
    from agent import action_controller
    assert action_controller.REDIRECT_URI == f"http://127.0.0.1:{nia.PORT}/spotify/callback"
    traded, opened = [], []

    class StubLogin:  # stands in for Spotify: never the real login, nor a real browser tab
        def get_authorize_url(self, state=None):
            return f"https://accounts.spotify.com/authorize?state={state}"
        def get_access_token(self, code, as_dict=True, check_cache=True):
            traded.append(code)
    real_oauth, real_open, real_account = action_controller.oauth, nia.webbrowser.open, nia.spotify_account
    action_controller.oauth, nia.webbrowser.open = StubLogin, opened.append
    nia.spotify_account = lambda login: "<b>alowki</b>"  # never asks the real Spotify who's signed in
    pages = []

    def callback(query):
        try:
            with urllib.request.urlopen(f"{base}/spotify/callback?{query}") as response:
                pages.append(response.read().decode())
                return response.status
        except urllib.error.HTTPError as e:
            pages.append(e.read().decode())
            return e.code
    try:
        assert callback("code=evil&state=x") == 400 and traded == [], "no login was started: refused"
        state = post("/spotify/login")[1]["url"].split("state=")[1]
        assert opened and callback("code=evil&state=wrong") == 400 and traded == [], "another page's code: refused"
        state = post("/spotify/login")[1]["url"].split("state=")[1]
        assert callback(f"code=good&state={state}") == 200 and traded == ["good"]
        # The tab she sends back: NIA's look, who's signed in (escaped - it's Spotify's text, not hers), what to say
        assert "Spotify is connected" in pages[-1] and 'data-state="ok"' in pages[-1]
        assert "&lt;b&gt;alowki&lt;/b&gt;" in pages[-1] and "<b>alowki" not in pages[-1]
        assert "Hey Nia, play some lofi" in pages[-1] and "{{" not in pages[-1].split("-->", 1)[1]
        assert callback(f"code=again&state={state}") == 400 and traded == ["good"], "each login is used once"
        assert "Spotify login didn&#x27;t finish" in pages[-1] and 'data-state="error"' in pages[-1]
        assert "wasn&#x27;t started from NIA&#x27;s settings panel" in pages[-1], "the failed tab says why"
        assert "spotify" not in hub.latest, "a reloaded page doesn't replay an old login's outcome"
    finally:
        action_controller.oauth, nia.webbrowser.open, nia.spotify_account = real_oauth, real_open, real_account

    # The window's quit
    assert post("/quit")[0] == 204 and done.is_set()
    httpd.shutdown()

# Her own taskbar button and pin: a Start-menu shortcut with her icon, how to start her, and the window's app ID
# (a folder outside AppData: Store Python sees its own private copy of AppData, the copy step the real one)
with tempfile.TemporaryDirectory(dir=nia.LOG_DIR, ignore_cleanup_errors=True) as tmp:
    if not nia.WINDOWS:  # Linux: her launcher entry, with her icon, how to start her and her window's app ID
        real_entry, nia.DESKTOP_ENTRY = nia.DESKTOP_ENTRY, Path(tmp) / "applications" / "nia.desktop"
        try:
            nia.taskbar_identity()
            entry = nia.DESKTOP_ENTRY.read_text()
            assert f'Exec="{sys.executable}" "{nia.ROOT / "nia.py"}"' in entry and f"Icon={nia.ICON}" in entry
            assert nia.ICON.is_file() and "StartupWMClass=nia" in entry
            from gi.repository import GLib
            assert GLib.get_prgname() == "nia", "the window's app ID matches the entry's name"
        finally:
            nia.DESKTOP_ENTRY = real_entry
    else:
        real_shortcut, nia.SHORTCUT = nia.SHORTCUT, Path(tmp) / "NIA.lnk"  # never the real Start menu here
        try:
            nia.taskbar_identity()
            import pythoncom
            from win32com.propsys import propsys, pscon
            from win32com.shell import shell
            link = pythoncom.CoCreateInstance(shell.CLSID_ShellLink, None, pythoncom.CLSCTX_INPROC_SERVER,
                                              shell.IID_IShellLink)
            link.QueryInterface(pythoncom.IID_IPersistFile).Load(str(nia.SHORTCUT))
            assert link.GetPath(0)[0].lower().endswith("pythonw.exe"), "starts her without a console"
            assert link.GetArguments() == f'"{nia.ROOT / "nia.py"}"' and link.GetIconLocation()[0] == str(nia.ICON)
            store = propsys.SHGetPropertyStoreFromParsingName(str(nia.SHORTCUT))
            assert store.GetValue(pscon.PKEY_AppUserModel_ID).GetValue() == nia.APP_ID, "the same ID as her window"
            assert not (nia.LOG_DIR / "NIA.lnk").exists(), "the staged copy is cleaned up"
        finally:
            nia.SHORTCUT = real_shortcut

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

# popen_bound: a process that owns a server dies abruptly - no cleanup at all - and the server goes with it. Started
# from a thread that ends first, as a settings save restarts it: on Linux that must not end the server early
owner = subprocess.Popen([sys.executable, "-c", f"""
import os, subprocess, sys, threading, time
sys.path.insert(0, {str(nia.ROOT)!r})
import nia
started = []
starter = threading.Thread(target=lambda: started.append(nia.popen_bound({SLEEPER!r})))
starter.start()
starter.join()
time.sleep(1)
assert started[0].poll() is None, "the server outlived the thread that started it"
print(started[0].pid, flush=True)
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

# Only NIA's own turns go in nia.log: her assistant and a typed session. A live test's turns once landed there
for script, log_file in (("main", "nia.log"), ("text_chat", "nia.log"), ("nia", "window.log"),
                         ("test_agent", "test.log"), ("live_play_test", "test.log"), ("scratch", "test.log")):
    chosen = subprocess.run([sys.executable, "-c", f"import sys; sys.argv = [{script + '.py'!r}]; "
                             "sys.path.insert(0, '.'); import utils.logger as l; print(l.LOG_FILE.name)"],
                            cwd=nia.ROOT, capture_output=True, text=True).stdout.strip()
    assert chosen == log_file, (script, chosen)

print("ok")
